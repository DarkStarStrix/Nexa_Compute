use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

pub struct Sampler {
    rng: ChaCha8Rng,
    weights: Vec<f64>,
    cumulative_weights: Vec<f64>,
    total_weight: f64,
}

impl Sampler {
    pub fn new(weights: Vec<f64>, seed: u64) -> Self {
        let mut cumulative_weights = Vec::with_capacity(weights.len());
        let mut total = 0.0;
        for w in &weights {
            total += w;
            cumulative_weights.push(total);
        }

        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            weights,
            cumulative_weights,
            total_weight: total,
        }
    }

    pub fn sample_index(&mut self) -> usize {
        let target = self.rng.gen::<f64>() * self.total_weight;
        match self.cumulative_weights.binary_search_by(|w| w.partial_cmp(&target).unwrap()) {
            Ok(index) => index,
            Err(index) => index.min(self.weights.len() - 1),
        }
    }
}

