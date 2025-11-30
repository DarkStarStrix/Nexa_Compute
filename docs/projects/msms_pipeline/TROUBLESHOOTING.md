# Troubleshooting

## Common Issues

### Integrity Error Rate > 0.1%
- Check HDF5 file format
- Verify preprocessing settings
- Review error types in quality report

### Checksum Mismatches
- Indicates data corruption
- Regenerate affected shards
- Check disk health

### Duplicate Sample IDs
- Critical bug in pipeline
- Check shard writer logic
- Verify HDF5 source files

### High Attrition Rate
- Review attrition reasons in quality report
- Adjust preprocessing thresholds if needed
- Verify data quality expectations

