use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};

#[pyfunction]
pub fn write_shards_parallel(
    shards: Vec<(String, Vec<String>)>, // list of (output_path, list_of_records_as_json_str)
) -> PyResult<Vec<String>> { // returns list of checksums
    let results = Arc::new(Mutex::new(Vec::with_capacity(shards.len())));
    
    // Pre-fill results to maintain order? 
    // For simplicity, let's just return checksums in no guaranteed order or map them.
    // Actually, parallel iterator can map and collect.
    
    let checksums: Vec<String> = shards.into_par_iter().map(|(path, records)| {
        write_single_shard(&path, &records).unwrap_or_else(|_| "error".to_string())
    }).collect();

    Ok(checksums)
}

fn write_single_shard(path: &str, records: &[String]) -> std::io::Result<String> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut hasher = Sha256::new();
    
    for record in records {
        let bytes = record.as_bytes();
        writer.write_all(bytes)?;
        writer.write_all(b"\n")?;
        hasher.update(bytes);
        hasher.update(b"\n");
    }
    
    writer.flush()?;
    let result = hasher.finalize();
    Ok(hex::encode(result))
}
