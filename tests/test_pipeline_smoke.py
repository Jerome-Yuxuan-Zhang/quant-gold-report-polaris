from __future__ import annotations

from pathlib import Path

from src.pipeline import run_pipeline


def test_pipeline_smoke(sample_dataset, tmp_path):
    output_root = tmp_path / "out"
    overrides = {
        "output": {
            "raw_dir": str(output_root / "raw"),
            "processed_dir": str(output_root / "processed"),
            "charts_dir": str(output_root / "charts"),
            "reports_dir": str(output_root / "reports"),
        }
    }
    result = run_pipeline(overrides=overrides, dataset_override=sample_dataset, generate_report=True, run_id="test_run")
    assert result["artifacts"]["report_path"] is not None
    assert Path(result["artifacts"]["report_path"]).exists()
    assert "trend_momentum" in result["figures"]["plotly"]
    assert "macro_correlation" in result["figures"]["artifacts"]

