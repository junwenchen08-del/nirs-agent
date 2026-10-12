"""Chinese report plots from persisted, bounded execution evidence only.

No dataset, estimator, fitting or Test-evaluation API is accepted here. The
caller supplies hash-verified Cal curves and already finalized Test points.
"""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from .report_labels import label

FONT = Path(__file__).with_name("report_assets") / "NotoSansSC-VF.ttf"


def _figure(title, xlabel, ylabel, *, panels=1):
    fig = Figure(figsize=(11, 4.4), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, panels))
    font = FontProperties(fname=str(FONT), size=10)
    for ax in axes:
        ax.set_title(title, fontproperties=font, pad=14)
        ax.set_xlabel(xlabel, fontproperties=font)
        ax.set_ylabel(ylabel, fontproperties=font)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            tick.set_fontproperties(font)
    return fig, axes, font


def _finite(values, *, maximum):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("图表数值不合法") from None
    if array.ndim != 1 or not 0 < len(array) <= maximum or not np.all(np.isfinite(array)):
        raise ValueError("图表数值必须有限，且符合缓存采样上限")
    return array


def build_report_figures(root, audit, *, cal_spectra=None, final_chart=None, target_unit=None):
    """Return explicit filenames/captions for the existing offline packager."""
    root = Path(root)
    figures = []

    def save(fig, filename, title, explanation):
        fig.savefig(root / filename, dpi=150, format="png")
        fig.clear()
        figures.append({"filename": filename, "title": title, "explanation": explanation})

    cal = cal_spectra or {}
    if cal.get("evidence_status") == "available" and cal.get("scope") == "calibration" and cal.get("source") == "actual_cal_fitted_preprocessing_state" and cal.get("experiment_id") == audit.get("retained_experiment_id"):
        before, after = cal["before"], cal["after"]
        if before.get("scope") != "calibration_before" or after.get("scope") != "calibration_after":
            raise ValueError("光谱图必须来自 Cal 缓存")
        x = _finite(before.get("x"), maximum=512)
        if not np.array_equal(x, _finite(after.get("x"), maximum=512)):
            raise ValueError("处理前后光谱轴不一致")
        delta = np.diff(x)
        if len(x) > 1 and not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError("光谱轴顺序不合法")
        raw, processed = before.get("series", []), after.get("series", [])
        if not 0 < len(raw) == len(processed) <= 12 or [item["id"] for item in raw] != [item["id"] for item in processed]:
            raise ValueError("处理前后光谱样本不一致")
        axis_unit = before.get("x_unit") or "单位未记录"
        xlabel = f"光谱轴（{axis_unit}）" if before.get("x_label") == "wavelength" else "变量序号"
        fig, axes, font = _figure("", xlabel, "光谱数值", panels=2)
        for ax, series, title in zip(axes, (raw, processed), ("原始光谱 · 校准集（Cal）", "预处理后光谱 · 校准集（Cal）"), strict=True):
            ax.set_title(title, fontproperties=font, pad=14)
            for item in series:
                y = _finite(item.get("values"), maximum=512)
                if len(x) != len(y):
                    raise ValueError("光谱曲线长度不一致")
                ax.plot(x, y, linewidth=0.9, alpha=0.7)
            ax.set_xlim(x[0], x[-1]) if len(x) > 1 else None
        save(
            fig,
            "cal_spectra.png",
            "原始与预处理后光谱（Cal）",
            f"展示 {len(raw)}/{before.get('n_total', '未记录')} 个 Cal 样本、{len(x)}/{before.get('axis_total', '未记录')} 个原始轴点；光谱轴：{axis_unit}。"
            "使用保留实验实际拟合的预处理状态与已保存曲线，保留波长顺序，未重新拟合；不包含 Test 光谱。",
        )

    records = [item for item in audit.get("decisions", []) if item.get("status") == "succeeded" and (item.get("execution") or {}).get("metrics", {}).get("val", {}).get("RMSE") is not None]
    if records:
        x = np.arange(1, len(records) + 1)
        y = _finite([item["execution"]["metrics"]["val"]["RMSE"] for item in records], maximum=1000)
        fig, (ax,), font = _figure("验证误差趋势 · 验证集（Val）", "实际成功实验序号", "均方根误差（RMSE，越低越好）")
        ax.plot(x, y, "o-", color="#0f766e", label="验证集误差（Val）")
        ax.set_xticks(x if len(x) <= 24 else x[:: max(1, len(x) // 24)])
        for i, item in enumerate(records):
            if item.get("experiment_id") == audit.get("retained_experiment_id"):
                ax.scatter(x[i], y[i], s=140, facecolor="none", edgecolor="#d97706", linewidth=2, label="LLM 最终保留实验", zorder=5)
        ax.legend(prop=font)
        save(fig, "validation_trend.png", "各轮验证误差趋势（Val）", "只显示本会话真实执行成功的实验，不包含未执行候选；黄色圆圈表示 LLM 保留选择，最低误差不会自动替代该选择。失败实验另见决策表。")

    final = final_chart or {}
    if final.get("scope") == "final_test" and final.get("evidence_status") == "available":
        try:
            observed = _finite(final.get("observed"), maximum=2000)
            predicted = _finite(final.get("predicted"), maximum=2000)
            residual = _finite(final.get("residual"), maximum=2000)
            if not len(observed) == len(predicted) == len(residual) or not np.allclose(residual, predicted - observed):
                raise ValueError("点对或残差不一致")
        except ValueError as exc:
            raise ValueError(f"最终测试图证据不合法：{exc}") from exc
        unit = label(target_unit) or "单位未记录"
        explanation = f"来自冻结后唯一最终测试的持久化点对，展示 {len(observed)}/{final.get('n_total', len(observed))} 个样本；未读取或重评 Test。误差单位：{unit}。"
        fig, (ax,), font = _figure("最终测试预测对照 · 测试集（Test）", f"实测值（{unit}）", f"预测值（{unit}）")
        lo, hi = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
        ax.scatter(observed, predicted, color="#0f766e", alpha=0.75, s=24)
        ax.plot([lo, hi], [lo, hi], "--", color="#64748b", label="理想预测（1∶1）")
        ax.legend(prop=font)
        save(fig, "final_prediction.png", "最终测试预测与实测对照（Test）", explanation + "点越接近虚线，预测越接近实测值。")
        fig, (ax,), font = _figure("最终测试残差 · 测试集（Test）", f"实测值（{unit}）", f"残差：预测－实测（{unit}）")
        ax.scatter(observed, residual, color="#2563eb", alpha=0.75, s=24)
        ax.axhline(0, color="#64748b", linestyle="--")
        save(fig, "final_residuals.png", "最终测试残差诊断（Test）", explanation + "残差定义为预测－实测，检查零线两侧分布与局部误差；不据此重新调参。")
    return figures
