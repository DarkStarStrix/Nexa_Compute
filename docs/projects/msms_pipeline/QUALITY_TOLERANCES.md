# Quality Tolerances

## Stage 1: Canonicalization (HDF5 → canonical)

- **Integrity errors**: ≤0.01% (NaN/inf/mismatch)
- **Attrition**: 1-10% acceptable if logged
- Alert if >0.1% unexplained failures

## Stage 2: Shard Construction (canonical → Parquet)

- **Missing samples**: 0%
- **Duplicates**: 0%
- **Checksum mismatches**: 0
- **Schema inconsistencies**: 0

## Stage 3: Training Loader (Arrow → tensors)

- **Runtime failures**: 0 per epoch
- **NaN/inf in tensors**: 0%
- **Batches skipped**: 0

## Stage 4: Metadata Quality

- **Training**: ≥90-95% fully resolved, ≤5-10% unknown
- **Eval/Calibration**: ≥99% resolved, <1% unknown

