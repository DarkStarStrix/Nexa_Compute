use serde::Serialize;

#[derive(Serialize)]
pub struct Metrics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

pub fn reduce(data: &[f64]) -> Metrics {
    if data.is_empty() {
        return Metrics { mean: 0.0, std: 0.0, min: 0.0, max: 0.0, p50: 0.0, p95: 0.0, p99: 0.0 };
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[((sorted.len() as f64 * 0.95) as usize).min(sorted.len()-1)];
    let p99 = sorted[((sorted.len() as f64 * 0.99) as usize).min(sorted.len()-1)];
    
    Metrics {
        mean,
        std,
        min: sorted[0],
        max: *sorted.last().unwrap(),
        p50,
        p95,
        p99,
    }
}
