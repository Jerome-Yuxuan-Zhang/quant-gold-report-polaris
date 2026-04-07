from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.utils import format_num, format_pct


def paragraph(text: str, style_name: str, styles) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), styles[style_name])


def chart_image(path: str | Path, width: float = 17 * cm) -> Image:
    image = Image(str(path))
    image._restrictSize(width, 11.8 * cm)
    return image


def build_metric_table(rows: list[list[str]], col_widths: list[float] | None = None) -> Table:
    table = Table(rows, hAlign="LEFT", colWidths=col_widths or [6 * cm, 4 * cm, 6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#edf2f7")]),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def build(*, settings: dict, run_id: str, dataset: dict, trend: dict, volatility: dict, macro: dict, backtest: dict, figures: dict, narrative: dict) -> Path:
    report_path = Path(settings["output"]["reports_dir"]) / f"quant_gold_report_{run_id}.pdf"
    chart_artifacts = figures["artifacts"]

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading1"], fontSize=18, textColor=colors.HexColor("#1d3557"), spaceAfter=8))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=10.5, leading=15))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))

    regression_rows = [["Term", "Coefficient", "t-stat", "p-value"]]
    regression_table = macro["artifacts"]["regression_table"]
    if regression_table.empty:
        regression_rows.append(["Unavailable", "N/A", "N/A", "N/A"])
    else:
        for row in regression_table.itertuples(index=False):
            regression_rows.append([row.term, format_num(row.coefficient), format_num(row.t_stat), format_num(row.p_value)])

    performance_rows = [
        ["Metric", "Value", "Comment"],
        ["Annualized Return (Net)", format_pct(backtest["summary"]["annualized_return"]), "Walk-forward, after transaction costs"],
        ["Annualized Return (Gross)", format_pct(backtest["summary"]["annualized_return_gross"]), "Before transaction costs"],
        ["Sharpe Ratio (Net)", format_num(backtest["summary"]["sharpe_ratio"]), "Daily returns annualized"],
        ["Maximum Drawdown (Net)", format_pct(backtest["summary"]["max_drawdown"]), "Peak-to-trough decline"],
        ["Monthly Hit Rate", format_pct(backtest["summary"]["hit_rate"]), "Share of positive months"],
    ]

    methodology_rows = [
        ["Research Question", settings["project"]["research_question"]],
        ["Primary Asset", dataset["summary"]["selected_symbol"]],
        ["As-of Macro Handling", dataset["metadata"]["macro_assumptions"]["cpi_index"]],
        ["Transaction Costs", f"{backtest['summary']['transaction_cost_bps']:.1f} bps per unit turnover"],
    ]

    story = [
        chart_image(chart_artifacts["cover_preview"], width=15.8 * cm),
        Spacer(1, 0.4 * cm),
        paragraph(settings["report"]["subtitle"], "Body", styles),
        Spacer(1, 0.2 * cm),
        paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "Small", styles),
        PageBreak(),
        paragraph("Research Objective", "SectionTitle", styles),
        paragraph(settings["project"]["research_question"], "Body", styles),
        Spacer(1, 0.2 * cm),
        paragraph(narrative["summary"]["executive_summary"], "Body", styles),
        Spacer(1, 0.3 * cm),
        paragraph("Market Context", "SectionTitle", styles),
        chart_image(chart_artifacts["trend_momentum"]),
        Spacer(1, 0.2 * cm),
        chart_image(chart_artifacts["volatility_regime"]),
        PageBreak(),
        paragraph("Factor Exposure", "SectionTitle", styles),
        paragraph(narrative["summary"]["macro_analysis"], "Body", styles),
        Spacer(1, 0.2 * cm),
        chart_image(chart_artifacts["macro_correlation"]),
        Spacer(1, 0.2 * cm),
        build_metric_table(regression_rows, col_widths=[5.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm]),
        PageBreak(),
        paragraph("Signal and Strategy Review", "SectionTitle", styles),
        paragraph(narrative["summary"]["signal_review"], "Body", styles),
        Spacer(1, 0.2 * cm),
        chart_image(chart_artifacts["backtest_equity"]),
        Spacer(1, 0.2 * cm),
        build_metric_table(performance_rows),
        PageBreak(),
        paragraph("Risk and Limitations", "SectionTitle", styles),
        paragraph(
            "The report evaluates one primary gold-linked asset at a time, so conclusions are not a diversification claim.\n"
            "Dual moving-average signals are deliberately simple and should be interpreted as baseline filters rather than proprietary alpha.\n"
            "Monthly CPI is mapped into the daily panel using a release-lag convention, but revised macro history can still make ex-post analysis cleaner than real-time conditions.\n"
            "Transaction costs are modeled with a simple basis-point assumption; slippage, taxes, and market impact are still omitted.\n"
            "The LLM layer is descriptive only and is constrained to statistics generated by the analytical modules.",
            "Body",
            styles,
        ),
        Spacer(1, 0.2 * cm),
        paragraph(narrative["summary"]["limitations_note"], "Small", styles),
        Spacer(1, 0.3 * cm),
        paragraph("Appendix", "SectionTitle", styles),
        build_metric_table([["Item", "Detail"]] + methodology_rows, col_widths=[4.5 * cm, 10.5 * cm]),
        Spacer(1, 0.25 * cm),
        chart_image(chart_artifacts["monthly_box"]),
        PageBreak(),
        paragraph("Distribution and Robustness", "SectionTitle", styles),
        chart_image(chart_artifacts["regime_box"]),
        Spacer(1, 0.2 * cm),
        chart_image(chart_artifacts["signal_box"]),
    ]

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title=settings["report"]["title"],
        author=settings["report"]["author"],
    )
    doc.build(story)
    shutil.copyfile(report_path, Path(settings["output"]["reports_dir"]) / "latest_quant_gold_report.pdf")
    return report_path

