import { expect, test, type Page } from "@playwright/test";

import { libraryDatasets, previewOf } from "../fixtures/dataset-library";

import { mockLangGraphAPI } from "./utils/mock-api";

const dataset = libraryDatasets[0]!;
const partition = {
  profile_id: "split_training",
  version: 3,
  method: "random",
  counts: { calibration: 36, validation: 12, test: 12 },
  parent_profile_id: "parse_1",
  source_thread_id: "source",
  session_id: "search_1",
};

async function setup(page: Page) {
  await mockLangGraphAPI(page).ready;
  const mutations: string[] = [];
  const reads: URL[] = [];
  await page.route("**/api/langgraph/threads", (route) =>
    route.fulfill({
      json: { ...route.request().postDataJSON(), status: "idle", values: {} },
    }),
  );
  await page.route("**/api/nir/datasets/**", (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() !== "GET")
      mutations.push(route.request().postData() ?? "");
    if (url.pathname.endsWith("/attach"))
      return route.fulfill({
        json: {
          dataset_id: dataset.id,
          profile_id: partition.profile_id,
          status: "attached",
          virtual_path: "/mnt/user-data/uploads/fixed-partition.csv",
          attachment_size_bytes: 1024,
        },
      });
    if (url.pathname.endsWith("/preview"))
      return route.fulfill({
        json: {
          ...previewOf(dataset),
          profile_id: url.searchParams.get("profile_id"),
        },
      });
    if (url.pathname.endsWith("/uses"))
      return route.fulfill({ json: { uses: [], count: 0 } });
    if (url.pathname.includes("/workspace")) {
      reads.push(url);
      const pinned =
        url.searchParams.get("profile_id") === partition.profile_id;
      const identity = {
        dataset_id: dataset.id,
        source_sha256: dataset.sha256,
        profile_id: pinned ? partition.profile_id : null,
      };
      if (url.pathname.endsWith("/rows"))
        return route.fulfill({
          json: {
            ...identity,
            total: 60,
            offset: 0,
            rows: [
              { index: 0, name: "样本-1", values: [10] },
              { index: 1, name: "样本-2", values: [11] },
            ],
          },
        });
      if (url.pathname.endsWith("/spectra"))
        return route.fulfill({
          json: {
            ...identity,
            axis: [1000, 1100],
            total: 60,
            point_count: 2,
            curves: [{ index: 0, name: "样本-1", values: [1, 2] }],
          },
        });
      return route.fulfill({
        json: {
          ...identity,
          version: pinned ? 3 : 0,
          saved: pinned,
          read_only: pinned,
          partition: pinned ? partition : null,
          sample_count: 60,
          feature_count: 2,
          axis_kind: "axis",
          axis: [1000, 1100],
          fields: ["水分"],
          usages: Array.from({ length: 60 }, (_, i) =>
            pinned ? (i < 36 ? 1 : i < 48 ? 2 : 3) : 0,
          ),
          range: [1000, 1100],
          selected_feature_count: 2,
        },
      });
    }
    return route.fulfill({
      json: {
        ...dataset,
        profiles: [
          {
            id: partition.profile_id,
            dataset_id: dataset.id,
            profile_version: 3,
            profile_status: "split",
            mapping: {},
            partition,
          },
        ],
      },
    });
  });
  return { reads, mutations };
}

test.beforeEach(async ({ page, baseURL }) => {
  await page
    .context()
    .addCookies([{ name: "locale", value: "zh-CN", url: baseURL! }]);
});

test("library partition opens read-only membership and reuses the exact version in a new chat", async ({
  page,
}) => {
  const state = await setup(page);
  await page.goto(`/workspace/nir/datasets/${dataset.id}`);
  await page.getByRole("tab", { name: "数据概览", exact: true }).click();
  const versions = page.getByRole("region", { name: "已保存的划分版本" });
  await expect(versions.getByRole("link", { name: "查看划分" })).toBeVisible();
  await versions.getByRole("link", { name: "查看划分" }).click();
  await expect(page).toHaveURL(/profile=split_training/);
  await expect(
    page.getByText("固定划分版本 v3", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "样本用途 1", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "保存修改", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("spinbutton", { name: "波段起始值" }),
  ).toBeDisabled();
  const pinned = state.reads.filter(
    (url) => url.searchParams.get("profile_id") === partition.profile_id,
  );
  expect(pinned.some((url) => url.pathname.endsWith("/rows"))).toBe(true);
  expect(pinned.some((url) => url.pathname.endsWith("/spectra"))).toBe(true);
  expect(state.mutations).toHaveLength(0);
  if (process.env.DATASET_PARTITIONS_QA_DIR)
    await page.screenshot({
      path: `${process.env.DATASET_PARTITIONS_QA_DIR}/fixed-partition.png`,
      fullPage: true,
      animations: "disabled",
    });
  await page
    .getByRole("button", { name: "在新对话中使用", exact: true })
    .click();
  await expect.poll(() => state.mutations.length).toBe(1);
  expect(JSON.parse(state.mutations[0]!).profile_id).toBe(partition.profile_id);
  await expect(
    page.getByText(dataset.name, { exact: true }).first(),
  ).toBeVisible();
});

test("model report links directly to its saved partition instead of current dataset settings", async ({
  page,
}) => {
  await setup(page);
  await page.route(
    "**/api/nir/models/partition-model/versions/v1/report",
    (route) =>
      route.fulfill({
        json: {
          model: {
            model_id: "partition-model",
            version: "v1",
            status: "ready",
            source_dataset_id: dataset.id,
            source_profile_id: partition.profile_id,
          },
          dataset: {
            id: dataset.id,
            name: dataset.name,
            source_sha256: dataset.sha256,
            profile_id: partition.profile_id,
            partition,
          },
          report: {
            schema_version: 1,
            target: "水分",
            method: "pls",
            sample_count: 60,
            feature_count: 2,
            model_feature_count: 2,
            parameters: {},
            preprocessing: [],
            metrics: [],
            charts: [],
          },
        },
      }),
  );
  await page.goto("/workspace/nir/models/partition-model/v1");
  await page.getByRole("tab", { name: "训练数据", exact: true }).click();
  const link = page.getByRole("link", { name: "查看训练时的划分" });
  await expect(link).toHaveAttribute(
    "href",
    `/workspace/nir/datasets/${dataset.id}?profile=split_training`,
  );
  await link.click();
  await expect(
    page.getByText("固定划分版本 v3", { exact: true }),
  ).toBeVisible();
});
