import { expect, test } from "@rstest/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import {
  ProcessChartView,
  plotLabel,
} from "@/components/workspace/nir/process-chart";
import { ProcessFacts } from "@/components/workspace/nir/process-facts";
import { decisionPipelineLabel } from "@/core/nir/autonomous";
import { factLabel, factValue } from "@/core/nir/process-display";

test("Chinese process labels preserve technical values and English locale", () => {
  expect(factLabel("actual_configuration", true)).toBe("实际执行配置");
  expect(factValue("strategy", "grid", true)).toBe("授权范围内的网格搜索");
  expect(factValue("action", "SWITCH_MODEL", true)).toBe("切换模型家族");
  expect(factValue("scope", "calibration_cv", true)).toBe(
    "校准交叉验证（Cal CV）",
  );
  expect(factValue("strategy", "grid", false)).toBe("grid");
  expect(factValue("action", "unknown", true)).toBe("unknown");
  const html = renderToStaticMarkup(
    createElement(ProcessFacts, {
      zh: true,
      value: {
        fits: 10,
        expected_fits: 12,
        strategy: "grid",
        actual_configuration: { method: "ridge", steps: [] },
      },
    }),
  );
  expect(html).toContain("实际拟合次数");
  expect(html).toContain("预计拟合次数");
  expect(html).toContain("授权范围内的网格搜索");
  const old = renderToStaticMarkup(
    createElement(ProcessFacts, {
      zh: true,
      value: { observations: ["Validation error increased"] },
    }),
  );
  expect(old).toContain("英文历史证据（展开查看原文）");
  expect(old).toContain("Validation error increased");
});

test("localizes actual pipeline in order and distinguishes absent evidence from raw", () => {
  const pipeline = {
    preprocessing: [{ method: "snv" }, { method: "derivative1" }],
    model_method: "pls",
  };
  expect(decisionPipelineLabel({ pipeline }, true)).toBe(
    "标准正态变量变换（SNV） → 一阶导数 → 偏最小二乘回归（PLS）",
  );
  expect(
    decisionPipelineLabel(
      { pipeline: { preprocessing: [], model_method: "svr" } },
      true,
    ),
  ).toContain("不做预处理");
  expect(decisionPipelineLabel({}, true)).toBe("方案未记录");
  expect(pipeline.preprocessing).toEqual([
    { method: "snv" },
    { method: "derivative1" },
  ]);
});

test("chart legend uses readable sample labels while retaining sample identity", () => {
  expect(plotLabel("RMSE", true)).toBe("均方根误差（RMSE）");
  expect(plotLabel("calibration_cv", true)).toBe("校准交叉验证（Cal CV）");
  const html = renderToStaticMarkup(
    createElement(ProcessChartView, {
      chart: {
        chart_schema_version: 1,
        chart_id: "chart",
        run_id: "run",
        attempt_id: "exp",
        step_execution_id: "step",
        type: "spectra",
        scope: "calibration_after",
        evidence_status: "available",
        x: [1, 2],
        series: [{ id: "84d98f23ce19", values: [1, 3] }],
        n_total: 1,
        n_returned: 1,
        sampling: "full_population",
      },
      zh: true,
      zoom: [0, 1],
      onZoom: () => undefined,
      selected: [],
      onSelect: () => undefined,
    }),
  );
  expect(html).toContain("样本 1");
  expect(html).toContain('title="84d98f23ce19"');
});
