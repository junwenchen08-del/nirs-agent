"""Comparison gallery generator.

Produces a self-contained HTML page comparing multiple calibration models,
including a metrics table and embedded predicted-vs-reference plots.
"""

from __future__ import annotations

from nir_core.models import ModelResult
from nir_core.plotting.model_diag import plot_predicted_vs_reference


def _metric_value(metrics: dict, key: str, default: str = "-") -> str:
    """Format a metric value for display in the HTML table."""
    val = metrics.get(key)
    if val is None:
        return default
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def generate_comparison_gallery(results: list[ModelResult]) -> str:
    """Generate a multi-model comparison HTML page.

    Includes:
    - A page title.
    - A metrics comparison table (method, n_components, RMSEP, R2, RPD).
    - One section per model with an embedded predicted-vs-reference plot
      (placeholder text if the model has no y_ref/y_pred stored).

    Args:
        results: List of ModelResult instances to compare.

    Returns:
        Complete HTML document string (including <html>...</html>).
    """
    rows_html: list[str] = []
    sections_html: list[str] = []

    for i, res in enumerate(results):
        method = res.method
        n_comp = res.n_components if res.n_components is not None else "-"
        rmsep = _metric_value(res.metrics, "RMSEP")
        r2 = _metric_value(res.metrics, "R2")
        rpd = _metric_value(res.metrics, "RPD")

        rows_html.append(
            "<tr>"
            f"<td>{i + 1}</td>"
            f"<td>{method}</td>"
            f"<td>{n_comp}</td>"
            f"<td>{rmsep}</td>"
            f"<td>{r2}</td>"
            f"<td>{rpd}</td>"
            "</tr>"
        )

        # Attempt to embed a predicted-vs-reference plot if y arrays are present.
        y_ref = res.metrics.get("y_ref")
        y_pred = res.metrics.get("y_pred")
        plot_html: str
        if y_ref is not None and y_pred is not None:
            try:
                import numpy as np

                png_b64 = plot_predicted_vs_reference(
                    np.asarray(y_ref),
                    np.asarray(y_pred),
                    title=f"{method} - Predicted vs Reference",
                )
                plot_html = (
                    f'<img src="data:image/png;base64,{png_b64}" '
                    f'alt="{method} predicted vs reference" '
                    f'style="max-width:100%;border:1px solid #ccc;"/>'
                )
            except Exception as exc:  # noqa: BLE001 - gallery must fail open
                plot_html = f"<p>无法生成预测图 / Plot unavailable: {exc}</p>"
        else:
            plot_html = (
                "<p style='color:#888;'>无 y_ref/y_pred 数据，"
                "无法生成预测vs参考图 / No prediction data available.</p>"
            )

        sections_html.append(
            "<section>"
            f"<h3>模型 {i + 1}: {method}</h3>"
            f"<p>主成分数 / Components: {n_comp} | "
            f"RMSEP: {rmsep} | R²: {r2} | RPD: {rpd}</p>"
            f"{plot_html}"
            "</section>"
        )

    table_html = (
        "<table>"
        "<thead><tr>"
        "<th>#</th><th>方法 Method</th><th>主成分 Components</th>"
        "<th>RMSEP</th><th>R²</th><th>RPD</th>"
        "</tr></thead>"
        "<tbody>" + "\n".join(rows_html) + "</tbody>"
        "</table>"
    )

    if not results:
        table_html = (
            "<p style='color:#888;'>无模型结果 / No model results to display.</p>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>模型对比 / Model Comparison Gallery</title>
<style>
  body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 24px; color: #222; }}
  h1 {{ color: #1f3a5f; }}
  h3 {{ color: #2c5282; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: center; }}
  th {{ background-color: #edf2f7; }}
  tr:nth-child(even) {{ background-color: #f7fafc; }}
  section {{ margin: 24px 0; padding: 12px; background: #fafafa; border-radius: 6px; }}
</style>
</head>
<body>
<h1>模型对比 / Model Comparison Gallery</h1>
<p>共 {len(results)} 个模型 / {len(results)} model(s) compared.</p>
{table_html}
{"".join(sections_html)}
</body>
</html>
"""
    return html
