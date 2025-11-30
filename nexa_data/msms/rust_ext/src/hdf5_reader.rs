/// Fast HDF5 reading routines - Python wrapper for now.
/// Full Rust implementation can be added later for maximum performance.
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn read_hdf5_spectrum(py: Python, path: &str, index: usize) -> PyResult<PyObject> {
    // Use Python's h5py for now (Rust hdf5 crate has different API)
    // This is a wrapper that can be optimized later with pure Rust
    let h5py = py.import_bound("h5py")?;
    let file = h5py.call_method1("File", (path, "r"))?;

    // Read spectrum data
    let spectrum_ds = file.get_item("spectrum")?;
    let spectrum_array = spectrum_ds.call_method1("__getitem__", (index,))?;

    // Extract m/z and intensities
    let mzs_slice = spectrum_array.call_method1("__getitem__", (0,))?;
    let ints_slice = spectrum_array.call_method1("__getitem__", (1,))?;

    // Convert to numpy arrays and filter zeros
    let np = py.import_bound("numpy")?;
    let mzs_array = np.call_method1("array", (mzs_slice,))?;
    let ints_array = np.call_method1("array", (ints_slice,))?;

    // Filter: keep only where both mz > 0 and intensity > 0
    let mask_mzs = np.call_method1("greater", (mzs_array.clone(), 0.0))?;
    let mask_ints = np.call_method1("greater", (ints_array.clone(), 0.0))?;
    let mask = np.call_method1("logical_and", (mask_mzs, mask_ints))?;

    let filtered_mzs = mzs_array.call_method1("__getitem__", (mask.clone(),))?;
    let filtered_ints = ints_array.call_method1("__getitem__", (mask,))?;

    // Read metadata
    let names_ds = file.get_item("name")?;
    let name_obj = names_ds.call_method1("__getitem__", (index,))?;
    let name: String = if name_obj.is_instance_of::<pyo3::types::PyBytes>() {
        String::from_utf8(name_obj.extract::<&[u8]>()?.to_vec()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyUnicodeDecodeError, _>(format!("Invalid UTF-8: {}", e))
        })?
    } else {
        name_obj.str()?.to_string()
    };

    let precursor_mz_ds = file.get_item("precursor_mz")?;
    let precursor_mz: f64 = precursor_mz_ds
        .call_method1("__getitem__", (index,))?
        .extract()?;

    let charge_ds = file.get_item("charge")?;
    let charge: i8 = charge_ds.call_method1("__getitem__", (index,))?.extract()?;

    // Build Python dictionary
    let sample_id = format!("{}_{}", name, index);
    let record = PyDict::new_bound(py);
    record.set_item("sample_id", sample_id)?;
    record.set_item("mzs", filtered_mzs)?;
    record.set_item("intensities", filtered_ints)?;
    record.set_item("precursor_mz", precursor_mz)?;
    record.set_item("charge", charge)?;
    record.set_item("collision_energy", 0.0f64)?;
    record.set_item("adduct", py.None())?;
    record.set_item("instrument_type", py.None())?;
    record.set_item("smiles", py.None())?;
    record.set_item("inchikey", py.None())?;
    record.set_item("formula", py.None())?;

    Ok(record.into())
}

#[pyfunction]
pub fn read_hdf5_batch(py: Python, path: &str, indices: Vec<usize>) -> PyResult<Vec<PyObject>> {
    let mut results = Vec::new();

    for index in indices {
        match read_hdf5_spectrum(py, path, index) {
            Ok(record) => results.push(record),
            Err(e) => {
                eprintln!("Error reading spectrum at index {}: {}", index, e);
            }
        }
    }

    Ok(results)
}
