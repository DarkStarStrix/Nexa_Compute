package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/table"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	titleStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
	headerStyle   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("39"))
	successStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("46"))
	warningStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("226"))
	errorStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	dimStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	highlightStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("39")).Bold(true)
)

type Metrics struct {
	TotalSpectra        int     `json:"total_spectra"`
	SamplesWritten      int     `json:"samples_written"`
	ShardsWritten       int     `json:"shards_written"`
	BytesWritten        int64   `json:"bytes_written"`
	ElapsedSeconds      float64 `json:"elapsed_seconds"`
	SamplesPerSecond    float64 `json:"samples_per_second"`
	IntegrityErrorCount int     `json:"integrity_error_count"`
	IntegrityErrors     map[string]int `json:"integrity_errors"`
	AttritionCount      int     `json:"attrition_count"`
}

type TimeSeriesPoint struct {
	Timestamp          float64  `json:"timestamp"`
	SamplesPerSecond   float64  `json:"samples_per_second"`
	BytesPerSecond     float64  `json:"bytes_per_second"`
	CPUPercent         *float64 `json:"cpu_percent"`
	MemoryPercent      *float64 `json:"memory_percent"`
	MemoryUsedGB       *float64 `json:"memory_used_gb"`
	DaskWorkerSat      *float64 `json:"dask_worker_saturation"`
	ShardsWritten      int      `json:"shards_written"`
	IntegrityErrorRate float64  `json:"integrity_error_rate"`
}

type model struct {
	runDir        string
	runID         string
	metrics       Metrics
	timeSeries    []TimeSeriesPoint
	progressBar   progress.Model
	spinner       spinner.Model
	metricsTable  table.Model
	width         int
	height        int
	lastUpdate    time.Time
	err           error
	loading       bool
}

func initialModel(runDir, runID string) model {
	m := model{
		runDir:     runDir,
		runID:      runID,
		loading:    true,
		lastUpdate: time.Now(),
	}

	// Initialize spinner
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("63"))
	m.spinner = s

	// Initialize progress bar
	prog := progress.New(progress.WithScaledGradient("#FF7CCB", "#FDFF8C"))
	prog.Width = 60
	m.progressBar = prog

	// Initialize metrics table
	columns := []table.Column{
		{Title: "Metric", Width: 25},
		{Title: "Value", Width: 20},
	}

	rows := []table.Row{
		{"Total Spectra", "0"},
		{"Samples Written", "0"},
		{"Shards Written", "0"},
		{"Throughput", "0 samples/s"},
		{"Integrity Errors", "0"},
	}

	t := table.New(
		table.WithColumns(columns),
		table.WithRows(rows),
		table.WithFocused(true),
		table.WithHeight(5),
	)

	tableStyles := table.DefaultStyles()
	tableStyles.Header = table.DefaultStyles().Header.
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("240")).
		BorderBottom(true).
		Bold(true)
	tableStyles.Selected = table.DefaultStyles().Selected.
		Foreground(lipgloss.Color("229")).
		Background(lipgloss.Color("57")).
		Bold(false)
	t.SetStyles(tableStyles)

	m.metricsTable = t

	return m
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		m.spinner.Tick,
		loadMetrics(m.runDir),
		tickMetrics(m.runDir),
	)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.metricsTable.SetWidth(msg.Width - 4)
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		}
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case metricsMsg:
		m.metrics = msg.metrics
		m.loading = false
		m.lastUpdate = time.Now()
		m.updateTable()
		m.updateProgress()
		return m, tickMetrics(m.runDir)

	case timeSeriesMsg:
		m.timeSeries = msg.data
		return m, nil

	case error:
		m.err = msg
		return m, nil
	}

	return m, nil
}

func (m *model) updateTable() {
	var progressPct float64
	if m.metrics.TotalSpectra > 0 && m.metrics.SamplesWritten > 0 {
		progressPct = float64(m.metrics.SamplesWritten) / float64(m.metrics.TotalSpectra)
	}

	rows := []table.Row{
		{"Total Spectra", fmt.Sprintf("%d", m.metrics.TotalSpectra)},
		{"Samples Written", fmt.Sprintf("%d", m.metrics.SamplesWritten)},
		{"Shards Written", fmt.Sprintf("%d", m.metrics.ShardsWritten)},
		{"Throughput", fmt.Sprintf("%.1f samples/s", m.metrics.SamplesPerSecond)},
		{"Integrity Errors", fmt.Sprintf("%d", m.metrics.IntegrityErrorCount)},
	}

	m.metricsTable.SetRows(rows)

	if progressPct > 0 {
		m.progressBar.SetPercent(progressPct)
	}
}

func (m *model) updateProgress() {
	if m.metrics.TotalSpectra > 0 {
		progressPct := float64(m.metrics.SamplesWritten) / float64(m.metrics.TotalSpectra)
		if progressPct > 1.0 {
			progressPct = 1.0
		}
		m.progressBar.SetPercent(progressPct)
	}
}

func (m model) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	var s string

	// Title
	title := titleStyle.Render("NexaData Pipeline Monitor")
	runInfo := dimStyle.Render(fmt.Sprintf("Run: %s", m.runID))
	header := lipgloss.JoinHorizontal(lipgloss.Left, title, "  ", runInfo)
	s += header + "\n\n"

	// Status
	if m.loading {
		s += m.spinner.View() + " Loading metrics...\n\n"
	} else {
		lastUpdate := time.Since(m.lastUpdate).Round(time.Second)
		s += successStyle.Render("●") + " " + dimStyle.Render(fmt.Sprintf("Updated %s ago", lastUpdate)) + "\n\n"
	}

	// Progress bar
	s += headerStyle.Render("Progress") + "\n"
	if m.metrics.TotalSpectra > 0 {
		progressText := fmt.Sprintf("%d / %d samples (%.1f%%)", 
			m.metrics.SamplesWritten, 
			m.metrics.TotalSpectra,
			float64(m.metrics.SamplesWritten)/float64(m.metrics.TotalSpectra)*100)
		s += m.progressBar.View() + " " + progressText + "\n\n"
	} else {
		s += dimStyle.Render("Waiting for data...") + "\n\n"
	}

	// Metrics table
	s += headerStyle.Render("Metrics") + "\n"
	s += m.metricsTable.View() + "\n\n"

	// Throughput
	s += headerStyle.Render("Performance") + "\n"
	s += fmt.Sprintf("Throughput: %s\n", highlightStyle.Render(fmt.Sprintf("%.1f samples/sec", m.metrics.SamplesPerSecond)))
	if m.metrics.ElapsedSeconds > 0 {
		s += fmt.Sprintf("Elapsed: %s\n", dimStyle.Render(formatDuration(m.metrics.ElapsedSeconds)))
	}
	s += fmt.Sprintf("Bytes Written: %s\n", dimStyle.Render(formatBytes(m.metrics.BytesWritten)))
	s += "\n"

	// Integrity status
	if m.metrics.IntegrityErrorCount > 0 {
		s += errorStyle.Render(fmt.Sprintf("⚠ Integrity Errors: %d", m.metrics.IntegrityErrorCount)) + "\n"
	} else {
		s += successStyle.Render("✓ No integrity errors") + "\n"
	}

	// Help
	s += "\n" + dimStyle.Render("Press 'q' to quit")

	return lipgloss.NewStyle().Width(m.width - 2).Padding(1, 2).Render(s)
}

type metricsMsg struct {
	metrics Metrics
}

type timeSeriesMsg struct {
	data []TimeSeriesPoint
}

func loadMetrics(runDir string) tea.Cmd {
	return func() tea.Msg {
		metricsPath := filepath.Join(runDir, "metrics.json")
		data, err := os.ReadFile(metricsPath)
		if err != nil {
			return error(err)
		}

		var metrics Metrics
		if err := json.Unmarshal(data, &metrics); err != nil {
			return error(err)
		}

		return metricsMsg{metrics: metrics}
	}
}

func loadTimeSeries(runDir string) tea.Cmd {
	return func() tea.Msg {
		tsPath := filepath.Join(runDir, "resource_timeseries.json")
		data, err := os.ReadFile(tsPath)
		if err != nil {
			return timeSeriesMsg{data: []TimeSeriesPoint{}}
		}

		var timeSeries []TimeSeriesPoint
		if err := json.Unmarshal(data, &timeSeries); err != nil {
			return timeSeriesMsg{data: []TimeSeriesPoint{}}
		}

		return timeSeriesMsg{data: timeSeries}
	}
}

func tickMetrics(runDir string) tea.Cmd {
	return tea.Tick(1*time.Second, func(t time.Time) tea.Msg {
		return loadMetrics(runDir)()
	})
}

func formatDuration(seconds float64) string {
	d := time.Duration(seconds * float64(time.Second))
	return d.Round(time.Second).String()
}

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

func main() {
	var runDir string
	var runID string

	flag.StringVar(&runDir, "run-dir", "", "Run directory containing metrics.json")
	flag.StringVar(&runID, "run-id", "", "Run ID")
	flag.Parse()

	if runDir == "" {
		fmt.Fprintf(os.Stderr, "Error: --run-dir is required\n")
		os.Exit(1)
	}

	if runID == "" {
		runID = "unknown"
	}

	p := tea.NewProgram(initialModel(runDir, runID), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

