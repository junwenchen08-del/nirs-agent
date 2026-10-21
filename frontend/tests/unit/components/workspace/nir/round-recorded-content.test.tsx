import { expect, test } from "@rstest/core";
import { renderToStaticMarkup } from "react-dom/server";

import {
  RecordedStatements,
  RoundParameters,
} from "@/components/workspace/nir/round-recorded-content";
import { recordedStatements } from "@/core/nir/round-content";

test("recorded claim objects render readable statements without empty numbered cards or evidence JSON", () => {
  const value = [
    {
      claim: "先尝试散射校正，再评价预测误差。",
      evidence_refs: ["profile:calibration"],
    },
    { evidence_refs: ["internal_only"] },
  ];
  const html = renderToStaticMarkup(<RecordedStatements value={value} zh />);
  expect(html).toContain("先尝试散射校正，再评价预测误差。");
  expect(html).not.toContain("profile:calibration");
  expect(html).not.toContain("internal_only");
  expect(html).not.toContain("<ol");
  expect(recordedStatements([{ evidence_refs: ["missing"] }])).toEqual([]);
});

test("historical strings remain readable and arbitrary nested technical objects do not become prose", () => {
  expect(
    recordedStatements([
      "已有观察",
      { claim: "已有观察" },
      { arbitrary: "不推断内容" },
      { description: "原始说明" },
    ]),
  ).toEqual(["已有观察", "原始说明"]);
});

test("model parameter names remain usable and untrusted strings are escaped", () => {
  const html = renderToStaticMarkup(
    <RoundParameters
      value={{ C: 10, epsilon: 0.1, custom_parameter: "<script>" }}
      zh
    />,
  );
  expect(html).toContain("C");
  expect(html).toContain("10");
  expect(html).toContain("custom_parameter");
  expect(html).toContain("&lt;script&gt;");
  expect(html).not.toContain("查看技术记录");
});
