import { expect, rs, test } from "@rstest/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

rs.mock("@/core/i18n/hooks", () => ({ useI18n: () => ({ locale: "zh-CN" }) }));
rs.mock("@/core/nir/process-hooks", () => ({ useProcessResource: rs.fn() }));

import { AutonomousProcess } from "@/components/workspace/nir/autonomous-process";
import { useProcessResource } from "@/core/nir/process-hooks";

test("compact progress preserves a delivery failure after Test and never loads final charts", () => {
  const read = rs.mocked(useProcessResource);
  read.mockClear();
  read.mockReturnValue({
    data: {
      schema_version: 1,
      session_id: "s_summary",
      stage: "finalized",
      automatic_delivery: { status: "failed" },
    },
    isError: false,
    isPending: false,
  } as unknown as ReturnType<typeof useProcessResource>);
  const html = renderToStaticMarkup(
    createElement(AutonomousProcess, {
      threadId: "thread_summary",
      sessionId: "s_summary",
      compact: true,
    }),
  );
  expect(html).toContain("自动交付失败，保留已完成证据");
  expect(html).not.toContain("独立 Test 已完成");
  expect(read.mock.calls.map((args) => args[1])).toEqual([
    null,
    "autonomous/s_summary",
  ]);
});
