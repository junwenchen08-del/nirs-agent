import type { Message } from "@langchain/langgraph-sdk";
import { expect, rs, test } from "@rstest/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

rs.mock("@/core/nir/process-hooks", () => ({ useProcessResource: rs.fn() }));
rs.mock("@/core/auth/AuthProvider", () => ({
  useAuth: () => ({ user: { id: "owner" } }),
}));
rs.mock("@/core/i18n/hooks", () => ({
  useI18n: () => ({ locale: "zh-CN", t: { common: { download: "下载" } } }),
}));
rs.mock("@/components/workspace/artifacts/context", () => ({
  useArtifacts: () => ({ select: rs.fn(), setOpen: rs.fn() }),
}));

import { NIRDeliveryFiles } from "@/components/workspace/messages/nir-delivery-files";
import { useProcessResource } from "@/core/nir/process-hooks";

const root = "mnt/user-data/outputs/search_abc/delivery_def/";
const message = {
  type: "ai",
  content: `[下载包](/${root}delivery.zip)`,
  additional_kwargs: { nir_deterministic_delivery: true },
} as Message;
const publication = {
  schema_version: 1,
  session_id: "search_abc",
  stage: "finalized",
  test_status: "consumed",
  export_status: "succeeded",
  downloads: ["model.joblib", "report.html", "delivery.zip"].map((name) => ({
    name,
    path: root + name,
  })),
};

test("old completed chat renders the real package card through a single read-only resource", () => {
  const mocked = rs.mocked(useProcessResource);
  mocked.mockReset();
  mocked.mockReturnValue({ data: publication } as ReturnType<
    typeof useProcessResource
  >);
  const html = renderToStaticMarkup(
    createElement(NIRDeliveryFiles, { message, threadId: "thread_a" }),
  );
  expect(html).toContain("完整建模交付包");
  expect(html).toContain(
    `/api/threads/thread_a/artifacts/${root}delivery.zip?download=true`,
  );
  expect(html).toContain("report.html");
  expect(mocked.mock.calls.map((call) => call.slice(0, 2))).toEqual([
    ["thread_a", "autonomous/search_abc"],
  ]);
});

test("refresh renders identical download links without any execution resource", () => {
  const mocked = rs.mocked(useProcessResource);
  mocked.mockReset();
  mocked.mockReturnValue({ data: publication } as ReturnType<
    typeof useProcessResource
  >);
  const render = () =>
    renderToStaticMarkup(
      createElement(NIRDeliveryFiles, { message, threadId: "thread_a" }),
    );
  expect(render()).toBe(render());
  expect(
    mocked.mock.calls.every((call) => call[1] === "autonomous/search_abc"),
  ).toBe(true);
});

test("failed publication never exposes links inferred from message text", () => {
  rs.mocked(useProcessResource).mockReturnValue({
    data: { ...publication, export_status: "failed" },
  } as ReturnType<typeof useProcessResource>);
  const html = renderToStaticMarkup(
    createElement(NIRDeliveryFiles, { message, threadId: "thread_a" }),
  );
  expect(html).not.toContain("delivery.zip");
  expect(html).not.toContain("下载");
});
