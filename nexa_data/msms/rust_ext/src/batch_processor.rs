use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::Bound;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::mem;

const DEFAULT_MAX_INPUT_PEAKS: usize = 1_000_000;

#[pyclass(name = "RustBatchProcessor")]
pub struct RustBatchProcessor {
    normalize: bool,
    sort_mz: bool,
    min_peaks: usize,
    max_peaks: usize,
    max_precursor_mz: f32,
    filter_nonfinite: bool,
    max_input_peaks: usize,
}

#[pymethods]
impl RustBatchProcessor {
    #[new]
    #[pyo3(signature=(
        normalize=true,
        sort_mz=true,
        min_peaks=1,
        max_peaks=4096,
        max_precursor_mz=2000.0,
        filter_nonfinite=true,
        max_input_peaks=DEFAULT_MAX_INPUT_PEAKS
    ))]
    pub fn new(
        normalize: bool,
        sort_mz: bool,
        min_peaks: usize,
        max_peaks: usize,
        max_precursor_mz: f32,
        filter_nonfinite: bool,
        max_input_peaks: usize,
    ) -> Self {
        let guarded_max_peaks = max_peaks.clamp(1, max_input_peaks.max(1));
        let input_guard = max_input_peaks.max(guarded_max_peaks);
        Self {
            normalize,
            sort_mz,
            min_peaks,
            max_peaks: guarded_max_peaks,
            max_precursor_mz,
            filter_nonfinite,
            max_input_peaks: input_guard,
        }
    }

    pub fn process(
        &self,
        py: Python<'_>,
        batch: Bound<'_, PyList>,
    ) -> PyResult<(Vec<PyObject>, Py<PyDict>)> {
        let processed = batch.len();
        let batch_ref = batch.as_ref();
        let mut spectra = Vec::with_capacity(processed);
        let iterator = batch_ref.iter()?;
        for item in iterator {
            let item = item?;
            let dict = item.downcast::<PyDict>()?;
            let record = dict.to_object(py);
            let precursor_mz: f32 = get_required_item(&dict, "precursor_mz")?.extract()?;
            let mzs_any = get_required_item(&dict, "mzs")?;
            let ints_any = match dict.get_item("ints")? {
                Some(value) => value,
                None => get_required_item(&dict, "intensities")?,
            };

            let mzs = extract_f32_vec(&mzs_any)?;
            let intensities = extract_f32_vec(&ints_any)?;

            spectra.push(SpectrumWorkItem {
                record,
                precursor_mz,
                mzs,
                intensities,
                status: SpectrumStatus::Pending,
            });
        }

        let cfg = ProcessorConfig {
            normalize: self.normalize,
            sort_mz: self.sort_mz,
            min_peaks: self.min_peaks,
            max_peaks: self.max_peaks,
            max_precursor_mz: self.max_precursor_mz,
            filter_nonfinite: self.filter_nonfinite,
            max_input_peaks: self.max_input_peaks,
        };

        py.allow_threads(|| {
            spectra
                .par_iter_mut()
                .for_each(|spec| process_spectrum(spec, &cfg));
        });

        let mut integrity_counts: HashMap<&'static str, usize> = HashMap::new();
        let mut attrition_counts: HashMap<&'static str, usize> = HashMap::new();
        let mut dropped_peaks_total = 0usize;
        let mut sanitized_records: Vec<PyObject> = Vec::new();

        for mut item in spectra.into_iter() {
            match item.status {
                SpectrumStatus::Valid {
                    peak_count,
                    dropped_peaks,
                } => {
                    dropped_peaks_total += dropped_peaks;
                    let dict_bound = item.record.bind(py);
                    let dict = dict_bound.downcast::<PyDict>()?;
                    let mzs_vec = mem::take(&mut item.mzs);
                    let ints_vec = mem::take(&mut item.intensities);
                    let mzs_array = mzs_vec.into_pyarray_bound(py);
                    dict.set_item("mzs", mzs_array.as_any())?;
                    let ints_array = ints_vec.into_pyarray_bound(py);
                    dict.set_item("ints", ints_array.as_any())?;
                    dict.set_item("intensities", ints_array.as_any())?;
                    dict.set_item("processing_engine", "rust")?;
                    dict.set_item("peak_count", peak_count)?;
                    dict.set_item("dropped_peaks", dropped_peaks)?;
                    sanitized_records.push(dict.to_object(py));
                }
                SpectrumStatus::Integrity(reason) => {
                    *integrity_counts.entry(reason).or_insert(0) += 1;
                }
                SpectrumStatus::Attrition(reason) => {
                    *attrition_counts.entry(reason).or_insert(0) += 1;
                }
                SpectrumStatus::Pending => {}
            }
        }

        let summary = build_summary_dict(
            py,
            processed,
            sanitized_records.len(),
            dropped_peaks_total,
            integrity_counts,
            attrition_counts,
        )?;

        Ok((sanitized_records, summary))
    }
}

fn extract_f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(array) = obj.downcast::<PyArray1<f32>>() {
        unsafe {
            return Ok(array.as_slice()?.to_vec());
        }
    }

    if let Ok(array64) = obj.downcast::<PyArray1<f64>>() {
        unsafe {
            return Ok(array64.as_slice()?.iter().map(|v| *v as f32).collect());
        }
    }

    if let Ok(seq) = obj.extract::<Vec<f32>>() {
        return Ok(seq);
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported array type for m/z or intensities",
    ))
}

#[derive(Clone, Copy)]
struct ProcessorConfig {
    normalize: bool,
    sort_mz: bool,
    min_peaks: usize,
    max_peaks: usize,
    max_precursor_mz: f32,
    filter_nonfinite: bool,
    max_input_peaks: usize,
}

struct SpectrumWorkItem {
    record: PyObject,
    precursor_mz: f32,
    mzs: Vec<f32>,
    intensities: Vec<f32>,
    status: SpectrumStatus,
}

enum SpectrumStatus {
    Pending,
    Valid {
        peak_count: usize,
        dropped_peaks: usize,
    },
    Integrity(&'static str),
    Attrition(&'static str),
}

#[derive(Clone, Copy)]
struct Peak {
    mz: f32,
    intensity: f32,
    index: usize,
}

fn process_spectrum(item: &mut SpectrumWorkItem, cfg: &ProcessorConfig) {
    if item.mzs.is_empty() || item.mzs.len() != item.intensities.len() {
        item.status = SpectrumStatus::Integrity("shape_mismatch");
        return;
    }

    if item.mzs.len() > cfg.max_input_peaks {
        item.status = SpectrumStatus::Attrition("excessive_peaks");
        item.mzs.clear();
        item.intensities.clear();
        return;
    }

    if !(0.0 < item.precursor_mz && item.precursor_mz < cfg.max_precursor_mz) {
        item.status = SpectrumStatus::Integrity("invalid_precursor_mz");
        return;
    }

    let mut peaks: Vec<Peak> = Vec::with_capacity(item.mzs.len().min(cfg.max_peaks));
    for (idx, (&mz, &intensity)) in item.mzs.iter().zip(item.intensities.iter()).enumerate() {
        if cfg.filter_nonfinite && (!mz.is_finite() || !intensity.is_finite()) {
            item.status = SpectrumStatus::Integrity("nonfinite_values");
            return;
        }

        if mz <= 0.0 {
            item.status = SpectrumStatus::Integrity("negative_mz");
            return;
        }

        if intensity > 0.0 {
            peaks.push(Peak {
                mz,
                intensity,
                index: idx,
            });
        }
    }

    if peaks.len() < cfg.min_peaks {
        item.status = SpectrumStatus::Attrition("too_few_peaks");
        return;
    }

    let mut dropped = 0usize;
    if cfg.max_peaks > 0 && peaks.len() > cfg.max_peaks {
        dropped = peaks.len() - cfg.max_peaks;
        let target = cfg.max_peaks;
        peaks.select_nth_unstable_by(target, |a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(Ordering::Equal)
        });
        peaks.truncate(target);
    }

    if cfg.sort_mz {
        peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap_or(Ordering::Equal));
    } else {
        peaks.sort_by(|a, b| a.index.cmp(&b.index));
    }

    if cfg.normalize {
        if let Some(max_intensity) =
            peaks
                .iter()
                .map(|peak| peak.intensity)
                .fold(None, |acc: Option<f32>, val| match acc {
                    Some(current) if current > val => Some(current),
                    Some(_) => Some(val),
                    None => Some(val),
                })
        {
            if max_intensity > 0.0 {
                for peak in peaks.iter_mut() {
                    peak.intensity /= max_intensity;
                }
            }
        }
    }

    item.mzs = peaks.iter().map(|peak| peak.mz).collect();
    item.intensities = peaks.iter().map(|peak| peak.intensity).collect();
    item.status = SpectrumStatus::Valid {
        peak_count: item.mzs.len(),
        dropped_peaks: dropped,
    };
}

fn build_summary_dict<'py>(
    py: Python<'py>,
    processed: usize,
    retained: usize,
    dropped_peaks: usize,
    integrity: HashMap<&'static str, usize>,
    attrition: HashMap<&'static str, usize>,
) -> PyResult<Py<PyDict>> {
    let summary = PyDict::new_bound(py);
    summary.set_item("processed", processed)?;
    summary.set_item("retained", retained)?;
    summary.set_item("dropped_peaks", dropped_peaks)?;

    let integrity_dict = PyDict::new_bound(py);
    for (key, value) in integrity {
        integrity_dict.set_item(key, value)?;
    }
    summary.set_item("integrity_errors", integrity_dict)?;

    let attrition_dict = PyDict::new_bound(py);
    for (key, value) in attrition {
        attrition_dict.set_item(key, value)?;
    }
    summary.set_item("attrition", attrition_dict)?;

    Ok(summary.unbind())
}

fn get_required_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    match dict.get_item(key)? {
        Some(value) => Ok(value),
        None => Err(PyErr::new::<PyKeyError, _>(format!("{key} missing"))),
    }
}
