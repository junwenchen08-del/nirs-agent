import type { Message } from "@langchain/langgraph-sdk";
import { expect, test } from "@rstest/core";

import {
  nirDeliveryFiles,
  nirDeliverySession,
} from "@/core/artifacts/delivery";

const root = "mnt/user-data/outputs/search_abc/delivery_def/";
const message = {
  type: "ai",
  content: `[下载包](/${root}delivery.zip)`,
  additional_kwargs: { nir_deterministic_delivery: true },
} as Message;
const summary = {
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

test("historical deterministic delivery restores its reference without a new run", () => {
  expect(nirDeliverySession(message)).toBe("search_abc");
  expect(nirDeliveryFiles(summary, "search_abc")).toEqual(
    summary.downloads.map((file) => "/" + file.path),
  );
});

test("new server delivery metadata does not depend on Markdown link formatting", () => {
  expect(
    nirDeliverySession({
      ...message,
      content: "交付完成",
      additional_kwargs: {
        nir_deterministic_delivery: true,
        nir_delivery_session_id: "search_abc",
      },
    }),
  ).toBe("search_abc");
});

test("ordinary assistant and human claims do not trigger artifact discovery", () => {
  expect(nirDeliverySession({ ...message, additional_kwargs: {} })).toBeNull();
  expect(nirDeliverySession({ ...message, type: "human" })).toBeNull();
  expect(
    nirDeliverySession({
      ...message,
      content: "[包](/mnt/user-data/outputs/search_abc/../delivery.zip)",
    }),
  ).toBeNull();
});

test("only finalized published outputs from the exact session appear", () => {
  expect(
    nirDeliveryFiles({ ...summary, stage: "searching" }, "search_abc"),
  ).toEqual([]);
  expect(
    nirDeliveryFiles({ ...summary, export_status: "failed" }, "search_abc"),
  ).toEqual([]);
  expect(nirDeliveryFiles(summary, "search_other")).toEqual([]);
  expect(
    nirDeliveryFiles(
      {
        ...summary,
        downloads: [
          ...summary.downloads,
          {
            name: "delivery.zip",
            path: "mnt/user-data/outputs/search_other/delivery_def/delivery.zip",
          },
          { name: "delivery.zip", path: root + "../private/delivery.zip" },
          { name: "delivery.zip", path: root + "model.joblib" },
        ],
      },
      "search_abc",
    ),
  ).toEqual(summary.downloads.map((file) => "/" + file.path));
});
