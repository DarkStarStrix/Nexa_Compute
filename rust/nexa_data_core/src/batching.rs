use rayon::prelude::*;
use pyo3::prelude::*;

pub fn process_files_parallel<F>(files: Vec<String>, processor: F) -> Result<(), String> 
where F: Fn(&str) -> Result<(), String> + Sync + Send {
    files.par_iter().try_for_each(|path| {
        processor(path)
    })
}

#[pyfunction]
pub fn parallel_process_files(files: Vec<String>) -> PyResult<()> {
    // Example placeholder for parallel processing logic
    // In a real implementation, we would pass a callback or configuration object
    // to define what processing happens (e.g., convert to parquet, validate schema)
    
    let result = process_files_parallel(files, |path| {
        // Placeholder processing
        // e.g. transform(path)
        Ok(())
    });
    
    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}
