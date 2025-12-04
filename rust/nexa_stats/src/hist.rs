use serde::Serialize;

#[derive(Serialize)]
pub struct Histogram {
    pub bins: Vec<f64>,
    pub counts: Vec<usize>,
}

pub fn compute(data: &[f64], num_bins: usize) -> Histogram {
    if data.is_empty() {
        return Histogram { bins: vec![], counts: vec![] };
    }
    
    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if (max - min).abs() < f64::EPSILON {
        return Histogram { bins: vec![min, max], counts: vec![data.len()] };
    }
    
    let step = (max - min) / num_bins as f64;
    let mut counts = vec![0; num_bins];
    let mut bins = Vec::with_capacity(num_bins + 1);
    
    for i in 0..=num_bins {
        bins.push(min + i as f64 * step);
    }
    
    for &val in data {
        let idx = ((val - min) / step).floor() as usize;
        let idx = idx.min(num_bins - 1);
        counts[idx] += 1;
    }
    
    Histogram { bins, counts }
}

