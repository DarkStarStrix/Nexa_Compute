use arrow::record_batch::RecordBatch;
use arrow::csv;
use arrow::json;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::{self, File};
use std::sync::Arc;
use std::io::{BufReader, Seek};
use pyo3.prelude::*;
use crate::config::PreprocessingConfig;
use crate::schema::parse_schema;
use crate::stats::CorpusStats;
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use std::collections::HashMap;
use arrow::pyarrow::ToPyArrow;
use arrow::pyarrow::FromPyArrow;

fn get_writer(output_path: &Path, schema: Arc<arrow::datatypes::Schema>) -> PyResult<ArrowWriter<File>> {
    let file = File::create(output_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let props = WriterProperties::builder().build();
    ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn process_single_csv(file_path: &Path, output_dir: &Path, config: &PreprocessingConfig, schema: Arc<arrow::datatypes::Schema>) -> PyResult<CorpusStats> {
    let file = File::open(file_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    let builder = csv::ReaderBuilder::new(schema.clone())
        .has_header(true)
        .with_batch_size(config.batch_size);
    
    let mut reader = builder.build(file).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    
    let file_stem = file_path.file_stem().unwrap().to_string_lossy();
    let output_path = output_dir.join(format!("{}.parquet", file_stem));
    let mut writer = get_writer(&output_path, schema)?;

    let mut row_count = 0;
    while let Some(batch_result) = reader.next() {
        let batch = batch_result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        row_count += batch.num_rows();
        writer.write(&batch).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    }
    
    writer.close().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let mut stats = CorpusStats::default();
    stats.total_rows = row_count;
    stats.file_stats.insert(file_stem.to_string(), row_count);
    
    Ok(stats)
}

fn process_single_jsonl(file_path: &Path, output_dir: &Path, config: &PreprocessingConfig, schema: Arc<arrow::datatypes::Schema>) -> PyResult<CorpusStats> {
    let file = File::open(file_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let reader = BufReader::new(file);
    
    let builder = json::ReaderBuilder::new(schema.clone())
        .with_batch_size(config.batch_size);
        
    let mut json_reader = builder.build(reader).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let file_stem = file_path.file_stem().unwrap().to_string_lossy();
    let output_path = output_dir.join(format!("{}.parquet", file_stem));
    let mut writer = get_writer(&output_path, schema)?;

    let mut row_count = 0;
    while let Some(batch_result) = json_reader.next() {
        let batch = batch_result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        row_count += batch.num_rows();
        writer.write(&batch).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    }
    
    writer.close().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let mut stats = CorpusStats::default();
    stats.total_rows = row_count;
    stats.file_stats.insert(file_stem.to_string(), row_count);

    Ok(stats)
}

fn get_input_files(input_path: &Path, extension: &str) -> PyResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    if input_path.is_dir() {
        for entry in fs::read_dir(input_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))? {
            let entry = entry.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(extension) {
                files.push(path);
            }
        }
    } else if input_path.is_file() {
        files.push(input_path.to_path_buf());
    }
    Ok(files)
}

pub fn process_csv(config: &PreprocessingConfig) -> PyResult<CorpusStats> {
    let input_path = Path::new(&config.input_path);
    let output_dir = Path::new(&config.output_dir);
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }

    let files = get_input_files(input_path, "csv")?;
    if files.is_empty() {
        return Ok(CorpusStats::default());
    }

    let schema = if let Some(schema_str) = &config.schema {
        parse_schema(schema_str)?
    } else {
        let file_for_schema = File::open(&files[0]).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let (inferred_schema, _) = csv::ReaderBuilder::new(Arc::new(arrow::datatypes::Schema::empty()))
            .has_header(true)
            .infer_schema_from_reader(file_for_schema, Some(100), None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Arc::new(inferred_schema)
    };

    // Parallel processing with reduce
    let stats = files.par_iter()
        .map(|file_path| process_single_csv(file_path, output_dir, config, schema.clone()))
        .reduce(
            || Ok(CorpusStats::default()),
            |a, b| {
                let mut acc = a?;
                let other = b?;
                acc.merge(&other);
                Ok(acc)
            }
        )?;
        
    Ok(stats)
}

pub fn process_jsonl(config: &PreprocessingConfig) -> PyResult<CorpusStats> {
    let input_path = Path::new(&config.input_path);
    let output_dir = Path::new(&config.output_dir);
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }

    let files = get_input_files(input_path, "jsonl")?;
    if files.is_empty() {
        return Ok(CorpusStats::default());
    }

    let schema = if let Some(schema_str) = &config.schema {
        parse_schema(schema_str)?
    } else {
        let file = File::open(&files[0]).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut reader = BufReader::new(file);
        let inferred = arrow::json::reader::infer_json_schema_from_seekable(&mut reader, Some(100))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Arc::new(inferred)
    };
    
    let stats = files.par_iter()
        .map(|file_path| process_single_jsonl(file_path, output_dir, config, schema.clone()))
        .reduce(
            || Ok(CorpusStats::default()),
            |a, b| {
                let mut acc = a?;
                let other = b?;
                acc.merge(&other);
                Ok(acc)
            }
        )?;
        
    Ok(stats)
}

// --- Stream Records Implementation ---

enum BatchReader {
    Csv(csv::Reader<File>),
    Json(json::Reader<BufReader<File>>),
}

#[pyclass]
pub struct RecordBatchIterator {
    reader: BatchReader,
}

#[pymethods]
impl RecordBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python) -> PyResult<Option<PyObject>> {
        let batch_option = match &mut slf.reader {
            BatchReader::Csv(r) => r.next().transpose().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
            BatchReader::Json(r) => r.next().transpose().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
        };

        match batch_option {
            Some(batch) => {
                // Convert RecordBatch to PyObject (pyarrow.RecordBatch)
                // We use ToPyArrow from arrow crate
                let pyarrow_batch = batch.to_pyarrow(py)?;
                Ok(Some(pyarrow_batch))
            }
            None => Ok(None),
        }
    }
}

pub fn stream_records(path: String, batch_size: usize) -> PyResult<RecordBatchIterator> {
    let file_path = Path::new(&path);
    let file = File::open(file_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");

    let reader = match extension {
        "csv" => {
             // Schema inference for streaming? 
             // For simplicity, assume header exists and let Reader infer.
             // Ideally we pass schema. But function sig is just path + batch_size.
             // We'll do dynamic inference or default.
             let (schema, _) = csv::ReaderBuilder::new(Arc::new(arrow::datatypes::Schema::empty()))
                .has_header(true)
                .infer_schema_from_reader(File::open(file_path).unwrap(), Some(100), None)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

             let r = csv::ReaderBuilder::new(Arc::new(schema))
                .has_header(true)
                .with_batch_size(batch_size)
                .build(file)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
             BatchReader::Csv(r)
        },
        "json" | "jsonl" => {
            let buf_reader = BufReader::new(file);
            // Schema inference
             let mut schema_reader = BufReader::new(File::open(file_path).unwrap());
             let schema = arrow::json::reader::infer_json_schema_from_seekable(&mut schema_reader, Some(100))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let r = json::ReaderBuilder::new(Arc::new(schema))
                .with_batch_size(batch_size)
                .build(buf_reader)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            BatchReader::Json(r)
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unsupported file extension")),
    };

    Ok(RecordBatchIterator { reader })
}

pub fn write_parquet_shards(records_iter: &PyAny, output_dir: String) -> PyResult<()> {
    let out_path = Path::new(&output_dir);
    if !out_path.exists() {
         fs::create_dir_all(out_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }

    // Iterate over python iterable
    let mut iter = records_iter.iter()?;
    
    // Strategy: write all to one file for now, or split? 
    // "write_parquet_shards" implies sharding.
    // We can shard by row count (e.g. 1M rows per file).
    
    let mut shard_idx = 0;
    let rows_per_shard = 1_000_000;
    let mut current_shard_rows = 0;
    
    let mut current_writer: Option<ArrowWriter<File>> = None;
    let mut current_schema: Option<Arc<arrow::datatypes::Schema>> = None;

    while let Some(item_result) = iter.next() {
        let item = item_result?;
        // item should be a pyarrow.RecordBatch
        let batch: RecordBatch = RecordBatch::from_pyarrow(item)?;
        
        if current_schema.is_none() {
             current_schema = Some(batch.schema());
        }
        
        if current_writer.is_none() {
            let path = out_path.join(format!("shard_{:04}.parquet", shard_idx));
            current_writer = Some(get_writer(&path, current_schema.clone().unwrap())?);
        }
        
        if let Some(writer) = &mut current_writer {
            writer.write(&batch).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            current_shard_rows += batch.num_rows();
            
            if current_shard_rows >= rows_per_shard {
                writer.close().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                current_writer = None;
                shard_idx += 1;
                current_shard_rows = 0;
            }
        }
    }
    
    if let Some(writer) = current_writer {
        writer.close().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    }

    Ok(())
}
