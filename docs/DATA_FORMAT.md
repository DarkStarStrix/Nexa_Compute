# Data Format

All built-in pipelines operate on tabular tensors:

- **Features**: `float32` tensor shaped `(N, num_features)`.
- **Targets**: `int64` tensor shaped `(N,)` representing class indices.
- **Synthetic dataset**: generated deterministically via `nexa_compute.data.dataset.SyntheticClassificationDataset` with configurable feature/label counts.

Processed datasets may also be exported to Parquet via `scripts/preprocess_data.py` with the following schema:

| Column       | Type    | Description                     |
|--------------|---------|---------------------------------|
| feature_0..N | float64 | Normalised feature columns      |
| label        | int64   | Target class label              |

Distilled datasets store teacher probabilities in `distilled_dataset.json`:

```json
{
  "teacher_probs": [0.1, 0.9],
  "target": 1
}
```
