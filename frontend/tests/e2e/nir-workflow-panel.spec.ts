import { expect, test } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

const THREAD_ID = "00000000-0000-0000-0000-0000000000aa";

test.beforeEach(async ({ context }) => {
  await context.route("**/api/v1/auth/me", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "default",
        email: "default@test.local",
        system_role: "admin",
        needs_setup: false,
        oauth_provider: null,
      }),
    }),
  );
});

test("shows bounded current NIR workflow evidence in the chat drawer", async ({
  page,
}) => {
  const routes = mockLangGraphAPI(page, {
    threads: [
      {
        thread_id: THREAD_ID,
        title: "NIR calibration",
        historyValues: {},
        values: {
          nir_workflow: {
            project_id: "nir-project-one",
            task_type: "calibration",
            stage: "review",
            revision: 8,
            attempt: 2,
            max_attempts: 3,
            validation_goal: "external_validation",
            approval_status: "pending",
            next_action: "request_approval",
            dataset_id: "ds_one",
            dataset_profile_id: "dsp_one",
            dataset_sha256: "a".repeat(64),
            data_path: "/mnt/user-data/uploads/private.csv",
            model_path: "/mnt/user-data/outputs/private.pkl",
            attempt_evidence: {
              method: "pls",
              pipeline_steps: ["snv", "derivative1"],
              validation_scope: "independent_external_validation",
              metrics_summary: { RMSEP: 0.1234, R2_val: 0.9345, RPD: 3.45 },
            },
          },
        },
      },
    ],
  });
  await routes.ready;

  await page.goto(`/workspace/chats/${THREAD_ID}`);
  const trigger = page.getByRole("button", { name: "Open NIR workflow" });
  await expect(trigger).toBeVisible({ timeout: 15_000 });
  await trigger.click();

  await expect(
    page.getByRole("heading", { name: "NIR workflow" }),
  ).toBeVisible();
  await expect(page.getByText("Review", { exact: true })).toBeVisible();
  await expect(page.getByText("Attempt 2 of 3")).toBeVisible();
  await expect(page.getByText("Independent external validation")).toBeVisible();
  await expect(page.getByText("snv → derivative1")).toBeVisible();
  await expect(page.getByText("ds_one", { exact: true })).toBeVisible();
  await expect(
    page.getByText("/mnt/user-data/uploads/private.csv"),
  ).toBeHidden();
  await expect(
    page.getByText("/mnt/user-data/outputs/private.pkl"),
  ).toBeHidden();
});

test("does not show the NIR drawer in an ordinary chat", async ({ page }) => {
  const routes = mockLangGraphAPI(page, {
    threads: [{ thread_id: THREAD_ID, title: "Ordinary chat" }],
  });
  await routes.ready;

  await page.goto(`/workspace/chats/${THREAD_ID}`);
  await expect(
    page.getByRole("button", { name: "Open NIR workflow" }),
  ).toBeHidden();
});
