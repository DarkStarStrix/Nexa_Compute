# NexaData TUI

Real-time terminal UI for monitoring the NexaData MS/MS pipeline using Golang and Bubble Tea.

## Building

```bash
cd nexa_data/msms/tui
go build -o nexa-data-tui .
```

Or install dependencies and build:
```bash
go mod tidy
go build -o nexa-data-tui .
```

## Usage

The TUI is automatically started by the pipeline when `tui` display mode is selected. You can also run it manually:

```bash
./nexa-data-tui --run-dir <run_directory> --run-id <run_id>
```

## Features

- Real-time progress bar
- Metrics table (total spectra, samples written, shards written, throughput, integrity errors)
- Performance stats (throughput, elapsed time, bytes written)
- Integrity status indicators
- Auto-updates every second from `metrics.json` file

## Keyboard Controls

- `q` - Quit
- `Ctrl+C` - Quit

## Integration

The TUI reads metrics from the pipeline's run directory:
- `metrics.json` - Main metrics (updated continuously)
- `resource_timeseries.json` - Time series data (optional)

