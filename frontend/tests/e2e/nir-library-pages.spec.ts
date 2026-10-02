import { expect, test, type Page } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

const READY_DATASET = {
  id: "ds_ready",
  name: "Tablet calibration",
  original_filename: "tablets.csv",
  sha256: "a".repeat(64),
  size_bytes: 2048,
  media_type: "text/csv",
  status: "ready",
  created_at: "2026-09-01T00:00:00Z",
};

const ARCHIVED_DATASET = {
  ...READY_DATASET,
  id: "ds_archived",
  name: "Archived batch",
  status: "archived",
};

const MODEL = {
  id: "nmv_1",
  model_id: "tablet_assay",
  version: "tablet-v1",
  status: "ready",
  method: "pls",
  preprocessing: { steps: [{ method: "snv" }] },
  validation_scope: "independent_holdout_not_external",
  artifact_size_bytes: 4096,
  artifact_sha256: "b".repeat(64),
  metrics_sha256: "c".repeat(64),
  training_data_sha256: "d".repeat(64),
  metrics_summary: { R2_val: 0.91, RMSEP: 0.32, RPD: 3.1 },
  source_dataset_id: "ds_ready",
  source_profile_id: "dsp_1",
  source_thread_id: "source-thread",
  source_run_id: "run-1",
  source_attempt: 1,
  created_at: "2026-09-02T00:00:00Z",
  updated_at: "2026-09-02T00:00:00Z",
  last_used_at: null,
  reused_existing: false,
};

const USAGE = {
  dataset_bytes: 2048,
  model_bytes: 4096,
  thread_bytes: 1024,
  accounted_total_bytes: 7168,
  max_user_dataset_bytes: 1024 ** 3,
  max_user_model_bytes: 1024 ** 3,
  library_write_admission_bytes: 3 * 1024 ** 3,
  strict_total_quota: false,
  disk_free_bytes: 10 * 1024 ** 3,
  min_free_disk_bytes: 1024 ** 3,
};

async function mockLibrary(page: Page) {
  await page.route("**/api/nir/storage/usage", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(USAGE),
    }),
  );
  await page.route("**/api/nir/datasets", (route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          datasets: [READY_DATASET, ARCHIVED_DATASET],
          count: 2,
        }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(READY_DATASET),
    });
  });
  await page.route("**/api/nir/datasets/ds_ready", (route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          ...READY_DATASET,
          profiles: [
            {
              id: "dsp_1",
              dataset_id: "ds_ready",
              profile_version: 1,
              profile_status: "confirmed",
              task_type: "calibration",
              schema_status: "confirmed_mapping",
              mapping: { target_column: "assay" },
              mapping_sha256: "e".repeat(64),
              confirmed_at: "2026-09-01T01:00:00Z",
              created_at: "2026-09-01T00:30:00Z",
            },
          ],
        }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "deleted" }),
    });
  });
  await page.route("**/api/nir/datasets/ds_ready/uses", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        uses: [
          {
            attachment_id: "use_1",
            dataset_id: "ds_ready",
            profile_id: "dsp_1",
            thread_id: "deleted-source-thread",
            run_id: "run-1",
            workflow_project_id: "project-1",
            source_sha256: "a".repeat(64),
            virtual_path: "/mnt/user-data/uploads/tablets.csv",
            status: "attached",
            reused_existing: true,
          },
        ],
        count: 1,
      }),
    }),
  );
  await page.route("**/api/nir/datasets/ds_ready/attach", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "attached",
        virtual_path: "/mnt/user-data/uploads/tablets.csv",
      }),
    }),
  );
  await page.route("**/api/nir/datasets/ds_archived", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "deleted",
        already_deleted: false,
        reclaimed_bytes: 2048,
        attached_thread_copies_retained: true,
      }),
    }),
  );
  await page.route("**/api/nir/models?include_archived=true", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ models: [MODEL], count: 1 }),
    }),
  );
  await page.route(
    "**/api/nir/models/tablet_assay/versions/tablet-v1/attach",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          status: "attached",
          model_path:
            "/mnt/user-data/outputs/models/tablet_assay/tablet-v1/model.pkl",
        }),
      }),
  );
}

test("dataset library shows owner-scoped assets, history, attach, and controlled deletion", async ({
  page,
}) => {
  const routes = mockLangGraphAPI(page);
  await routes.ready;
  await mockLibrary(page);
  await page.goto("/workspace/nir/datasets");

  await expect(
    page.getByRole("heading", { name: "Spectral dataset library" }),
  ).toBeVisible();
  await expect(
    page.getByText("Tablet calibration", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Persistent datasets", { exact: true }),
  ).toBeVisible();

  await page
    .getByRole("button", { name: "Details and history" })
    .first()
    .click();
  await expect(
    page.getByText("deleted-source-thread", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("deleted-source-thread", { exact: true }),
  ).not.toHaveAttribute("href");
  await page.getByRole("button", { name: "Close" }).click();

  await page.getByRole("button", { name: "Attach to chat" }).first().click();
  await page.getByLabel("Target chat ID").fill("target-thread");
  await page.getByRole("button", { name: "Attach to chat" }).last().click();
  await expect(page.getByText("Operation completed").last()).toBeVisible();

  const archivedCard = page
    .getByTestId("nir-dataset-card")
    .filter({ hasText: "Archived batch" });
  await archivedCard.getByRole("button", { name: "Delete bytes" }).click();
  const confirm = page.getByLabel(/Type the exact identifier/);
  await confirm.fill("ds_archived");
  await page.getByRole("button", { name: "Delete bytes" }).last().click();
  await expect(page.getByText("Operation completed").last()).toBeVisible();
});

test("model library labels validation conservatively and attaches an approved version", async ({
  page,
}) => {
  const routes = mockLangGraphAPI(page);
  await routes.ready;
  await mockLibrary(page);
  await page.goto("/workspace/nir/models");

  await expect(
    page.getByRole("heading", { name: "Persistent model library" }),
  ).toBeVisible();
  await expect(page.getByText("tablet_assay", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Independent internal holdout; not external validation", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByText("Validation scope is not production approval", {
      exact: true,
    }),
  ).toBeVisible();

  await page.getByRole("button", { name: "Attach to chat" }).click();
  await page.getByLabel("Target chat ID").fill("prediction-thread");
  await page.getByRole("button", { name: "Attach to chat" }).last().click();
  await expect(page.getByText("Operation completed")).toBeVisible();
});
