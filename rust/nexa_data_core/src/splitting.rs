use pyo3.prelude::*;
use std::path::Path;
use std::fs::File;
use std::sync::Arc;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::arrow::ArrowReader;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use arrow::record_batch::RecordBatch;
use arrow::compute::concat_batches;
use crate::config::SplitConfig;

pub fn split_dataset(config: &SplitConfig) -> PyResult<()> {
    // Basic in-memory shuffle and split for now.
    // For larger datasets, we'd need out-of-core shuffle or sharded shuffle.
    // Assuming input is a single parquet file for MVP or we read it all.
    
    let file = File::open(&config.input_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let reader = SerializedFileReader::new(file).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut arrow_reader = parquet::arrow::ParquetFileArrowReader::new(Arc::new(reader));
    
    // Read all batches
    let record_reader = arrow_reader.get_record_reader(1024).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut batches = Vec::new();
    for batch in record_reader {
        batches.push(batch.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?);
    }
    
    if batches.is_empty() {
        return Ok(());
    }
    
    let schema = batches[0].schema();
    // Concatenate all batches to shuffle indices
    let combined_batch = concat_batches(&schema, &batches).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let num_rows = combined_batch.num_rows();
    
    let mut indices: Vec<usize> = (0..num_rows).collect();
    let mut rng = StdRng::seed_from_u64(config.seed);
    indices.shuffle(&mut rng);
    
    // Determine split points
    let train_count = (num_rows as f64 * config.train_ratio) as usize;
    let val_count = (num_rows as f64 * config.val_ratio) as usize;
    // test gets the rest
    
    let train_indices = &indices[0..train_count];
    let val_indices = &indices[train_count..train_count + val_count];
    let test_indices = &indices[train_count + val_count..];
    
    // Helper to slice batch by indices
    // Since arrow doesn't have "take" by array of indices easily on RecordBatch directly without some compute kernels,
    // we utilize arrow::compute::take.
    
    fn write_indices(
        indices: &[usize],
        batch: &RecordBatch,
        output_dir: &str,
        name: &str,
        schema: Arc<arrow::datatypes::Schema>
    ) -> PyResult<()> {
        if indices.is_empty() { return Ok(()); }
        
        let indices_array = arrow::array::UInt32Array::from(indices.iter().map(|&i| i as u32).collect::<Vec<u32>>());
        let indices_dyn = Arc::new(indices_array) as arrow::array::ArrayRef;
        
        // Take rows for each column
        let mut columns = Vec::new();
        for col in batch.columns() {
            let new_col = arrow::compute::take(col, &indices_dyn, None)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            columns.push(new_col);
        }
        
        let new_batch = RecordBatch::try_new(schema.clone(), columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        let output_path = Path::new(output_dir).join(format!("{}.parquet", name));
        let file = File::create(output_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        writer.write(&new_batch).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        writer.close().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }
    
    write_indices(train_indices, &combined_batch, &config.output_dir, "train", schema.clone())?;
    write_indices(val_indices, &combined_batch, &config.output_dir, "val", schema.clone())?;
    write_indices(test_indices, &combined_batch, &config.output_dir, "test", schema.clone())?;
    
    Ok(())
}

