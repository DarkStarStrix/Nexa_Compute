use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use pyo3::prelude::*;

pub fn deterministic_shuffle(indices: &mut [usize], seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
}

pub fn split_indices(indices: &[usize], weights: &[f64]) -> Vec<Vec<usize>> {
    let total_weight: f64 = weights.iter().sum();
    let total_items = indices.len();
    let mut result = Vec::with_capacity(weights.len());
    
    let mut start = 0;
    for weight in weights {
        let count = ((weight / total_weight) * total_items as f64).round() as usize;
        let end = (start + count).min(total_items);
        result.push(indices[start..end].to_vec());
        start = end;
    }
    
    // If any items remain (due to rounding), add to last split
    if start < total_items {
        if let Some(last) = result.last_mut() {
            last.extend_from_slice(&indices[start..]);
        }
    }
    
    result
}

#[pyfunction]
pub fn shuffle_and_split(num_items: usize, weights: Vec<f64>, seed: u64) -> PyResult<Vec<Vec<usize>>> {
    let mut indices: Vec<usize> = (0..num_items).collect();
    deterministic_shuffle(&mut indices, seed);
    Ok(split_indices(&indices, &weights))
}
