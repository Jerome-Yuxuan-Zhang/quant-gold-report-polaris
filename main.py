from __future__ import annotations

from pathlib import Path

from src.pipeline import run_pipeline


def main() -> Path:
    result = run_pipeline(generate_report=True)
    report_path = result["artifacts"]["report_path"]
    print(f"Report generated: {report_path}")
    return Path(report_path)


if __name__ == "__main__":
    main()

