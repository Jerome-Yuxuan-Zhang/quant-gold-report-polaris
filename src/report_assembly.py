from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.utils import format_num, format_pct


def build_cover_preview(settings: dict, dataset: dict, run_id: str) -> Path:
    preview_path = Path(settings["output"]["charts_dir"]) / f"cover_preview_{run_id}.png"
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.5, 0.82, settings["report"]["title"], ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0.5, 0.74, settings["project"]["tagline"], ha="center", va="center", fontsize=12, wrap=True)
    ax.text(0.5, 0.56, f"Asset: {settings['asset']['primary_symbol']}", ha="center", fontsize=16)
    ax.text(0.5, 0.50, f"Coverage: {dataset['summary']['start']} to {dataset['summary']['end']}", ha="center", fontsize=14)
    ax.text(0.5, 0.44, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha="center", fontsize=14)
    ax.text(0.5, 0.30, f"Author: {settings['report']['author']}", ha="center", fontsize=14)
    fig.tight_layout()
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    shutil.copyfile(preview_path, Path(settings["output"]["charts_dir"]) / "latest_cover_preview.png")
    return preview_path


def paragraph(text: str, style_name: str, styles) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), styles[style_name])


def chart_image(path: str | Path, width: float = 17 * cm) -> Image:
    image = Image(str(path))
    image._restrictSize(width, 12 * cm)
    return image


def build_metric_table(rows: list[list[str]]) -> Table:
    table = Table(rows, hAlign="LEFT", colWidths=[6 * cm, 4 * cm, 6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b4d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#edf2f7")]),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ]
        )
    )
    return table


def build(*, settings: dict, run_id: str, dataset: dict, trend: dict, volatility: dict, macro: dict, backtest: dict, narrative: dict) -> Path:
    report_path = Path(settings["output"]["reports_dir"]) / f"quant_gold_report_{run_id}.pdf"
    cover_preview_path = build_cover_preview(settings, dataset, run_id)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading1"], fontSize=18, textColor=colors.HexColor("#1f3b4d")))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=10.5, leading=15))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))

    story = [
        chart_image(cover_preview_path, width=16 * cm),
        Spacer(1, 0.5 * cm),
        paragraph(settings["project"]["tagline"], "Body", styles),
        Spacer(1, 0.3 * cm),
        paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "Small", styles),
        PageBreak(),
        paragraph("Executive Summary", "SectionTitle", styles),
        paragraph(narrative["summary"]["executive_summary"], "Body", styles),
        Spacer(1, 0.3 * cm),
        paragraph("Market Overview", "SectionTitle", styles),
        chart_image(trend["artifacts"]["trend_chart"]),
        Spacer(1, 0.2 * cm),
        chart_image(volatility["artifacts"]["vol_chart"]),
        PageBreak(),
        paragraph("Macro Factor Analysis", "SectionTitle", styles),
        paragraph(narrative["summary"]["macro_analysis"], "Body", styles),
        Spacer(1, 0.2 * cm),
        chart_image(macro["artifacts"]["macro_chart"]),
        Spacer(1, 0.2 * cm),
    ]

    regression_rows = [["Term", "Coefficient", "t-stat"]]
    regression_table = macro["artifacts"]["regression_table"]
    if regression_table.empty:
        regression_rows.append(["Unavailable", "N/A", "N/A"])
    else:
        for row in regression_table.itertuples(index=False):
            regression_rows.append([row.term, format_num(row.coefficient), format_num(row.t_stat)])
    story.extend(
        [
            build_metric_table(regression_rows),
            PageBreak(),
            paragraph("Quantitative Signal Review", "SectionTitle", styles),
            paragraph(narrative["summary"]["signal_review"], "Body", styles),
            Spacer(1, 0.2 * cm),
            chart_image(backtest["artifacts"]["equity_curve_chart"]),
            Spacer(1, 0.2 * cm),
        ]
    )

    performance_rows = [
        ["Metric", "Value", "Comment"],
        ["Annualized Return", format_pct(backtest["summary"]["annualized_return"]), "Walk-forward out-of-sample"],
        ["Sharpe Ratio", format_num(backtest["summary"]["sharpe_ratio"]), "Daily returns annualized"],
        ["Maximum Drawdown", format_pct(backtest["summary"]["max_drawdown"]), "Peak-to-trough decline"],
        ["Monthly Hit Rate", format_pct(backtest["summary"]["hit_rate"]), "Share of positive months"],
    ]
    story.extend(
        [
            build_metric_table(performance_rows),
            PageBreak(),
            paragraph("Risk & Limitations", "SectionTitle", styles),
            paragraph(
                "This is a single-asset, single-strategy backtest with no diversification benefit.\n"
                "Dual moving-average crossovers are widely known and unlikely to represent durable alpha.\n"
                "No transaction costs or slippage are modeled, so realized implementation performance may be worse.\n"
                "Signal parameters were selected with historical knowledge, so data-snooping risk remains.\n"
                "The LLM layer describes structured outputs and does not make forecasts.\n"
                "Free data sources can contain revisions, gaps, or symbol-specific inconsistencies.",
                "Body",
                styles,
            ),
            Spacer(1, 0.2 * cm),
            paragraph(narrative["summary"]["limitations_note"], "Small", styles),
            Spacer(1, 0.3 * cm),
            paragraph("Appendix", "SectionTitle", styles),
            paragraph(
                f"Data sources include Yahoo Finance, FRED, and optional AkShare SGE data.\n"
                f"Primary asset: {settings['asset']['primary_symbol']}; flow proxy: {settings['asset']['flow_proxy_symbol']}.\n"
                f"Latest volatility regime: {volatility['summary']['latest_vol_regime'] or 'N/A'}.\n"
                f"LLM narrative mode: {narrative['summary']['mode']}.\n"
                f"Shanghai gold extension status: {macro['summary']['shanghai_analysis'].get('status', 'unavailable')}.",
                "Body",
                styles,
            ),
        ]
    )

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title=settings["report"]["title"],
        author=settings["report"]["author"],
    )
    doc.build(story)
    shutil.copyfile(report_path, Path(settings["output"]["reports_dir"]) / "latest_quant_gold_report.pdf")
    return report_path
