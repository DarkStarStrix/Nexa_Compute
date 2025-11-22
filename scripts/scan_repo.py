import os
import re
from pathlib import Path

# Define the spec sections (keywords) to look for
KEYWORDS = [
    "introduction",
    "data quality",
    "distillation process",
    "model inference",
    "training pipeline",
    "evaluation metrics",
    "conclusion",
]

# File extensions to scan
EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}

# Root directory (project root)
ROOT = Path(__file__).resolve().parents[1]  # assuming script is in scripts/

report_lines = []

def scan_file(file_path: Path):
    try:
        text = file_path.read_text(errors="ignore")
    except Exception:
        return
    for keyword in KEYWORDS:
        # case‑insensitive search
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for match in pattern.finditer(text):
            # capture a short snippet around the match
            start = max(match.start() - 30, 0)
            end = min(match.end() + 30, len(text))
            snippet = text[start:end].replace("\n", " ")
            report_lines.append(
                f"- {file_path.relative_to(ROOT)}:{match.start()} – `{keyword}` → {snippet.strip()}"
            )

def main():
    for dirpath, _, filenames in os.walk(ROOT):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() in EXTENSIONS:
                scan_file(fpath)
    # Write report
    report_path = ROOT / "docs" / "scan_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Repository Scan Report for Spec_Nexa_2.md\n\n")
        if report_lines:
            f.write("## References found\n\n")
            f.write("\n".join(report_lines))
        else:
            f.write("No references to spec sections were found.\n")
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
