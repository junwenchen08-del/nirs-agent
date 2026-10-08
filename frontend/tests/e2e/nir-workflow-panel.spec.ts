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

test("shows bounded modeling story, selection and result at desktop widths", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
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
            next_action: "request_user_approval",
            dataset_id: "ds_one",
            dataset_profile_id: "dsp_one",
            dataset_sha256: "a".repeat(64),
            data_path: "/mnt/user-data/uploads/private.csv",
            model_path: "/mnt/user-data/outputs/private.pkl",
            audit_evidence: {
              n_samples: 80,
              n_wavelengths: 150,
              has_nan: false,
              data_path: "/mnt/private.csv",
            },
            attempts: [
              {
                attempt: 1,
                method: "pls",
                pipeline_steps: ["snv"],
                protocol: "p",
                validation_scope: "independent_external_validation",
                metrics_summary: { RMSEP: 0.22 },
                decision_facts: {
                  preprocessing_selection: {
                    selected_candidate_id: "snv",
                    reason_code: "lowest_cv_rmse",
                    candidates: [
                      { candidate_id: "raw", cv_rmse: 0.4 },
                      { candidate_id: "snv", cv_rmse: 0.3, selected: true },
                    ],
                  },
                  wavelength_selection_decision: {
                    selected_method: "none",
                    reason_code: "cars_improvement_below_threshold",
                  },
                  wavelength_selection_candidates: [
                    { method: "none", RMSE_tuning: 0.32 },
                    { method: "cars", RMSE_tuning: 0.31 },
                  ],
                  model_selection_decision: {
                    selected_method: "pls",
                    reason_code: "pls_only_candidate",
                  },
                  model_candidates: [{ method: "pls", RMSE_tuning: 0.3 }],
                },
              },
              {
                attempt: 2,
                method: "pls",
                pipeline_steps: ["snv", "derivative1"],
                protocol: "p",
                validation_scope: "independent_external_validation",
                metrics_summary: { RMSEP: 0.1234, R2_val: 0.9345, RPD: 3.45 },
                passed: true,
              },
            ],
            reflections: [
              {
                source_attempt: 1,
                should_retry: true,
                reason: "Residual structure remained",
              },
            ],
            retry_plans: [
              {
                source_attempt: 1,
                target_attempt: 2,
                method: "pls",
                pipeline_steps: ["snv", "derivative1"],
                rationale: "Improve baseline shape",
              },
            ],
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
  const drawer = page.locator('[data-slot="sheet-content"]');
  const compactBox = await drawer.boundingBox();
  expect(compactBox?.width).toBeGreaterThanOrEqual(900);
  expect(compactBox?.width).toBeLessThanOrEqual(1100);

  await expect(
    page.getByRole("heading", { name: "ChemAgent modeling process" }),
  ).toBeVisible();
  await expect(page.getByText("Attempt 2 of 3")).toBeVisible();
  await expect(page.getByText("Samples")).toBeVisible();
  await expect(page.getByText("Residual structure remained")).toBeVisible();
  await expect(page.getByText("Improve baseline shape")).toBeVisible();
  await expect(page.getByText("Review the model results")).toBeVisible();
  await expect(page.getByText("ds_one", { exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Selection evidence" }).click();
  await page.getByLabel("View attempt").selectOption("1");
  await expect(
    page.getByText(
      "This option had the lowest error in training data validation.",
    ),
  ).toBeVisible();
  await expect(page.getByText("Reason code: lowest_cv_rmse")).toBeHidden();
  await page.getByText("Show original technical record").first().click();
  await expect(page.getByText("Reason code: lowest_cv_rmse")).toBeVisible();
  await expect(
    page.getByRole("img", { name: "cv_rmse candidate comparison" }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Final result" }).click();
  await expect(page.getByText("Independent external validation")).toBeVisible();
  await expect(
    page.getByText("/mnt/user-data/uploads/private.csv"),
  ).toBeHidden();
  await expect(
    page.getByText("/mnt/user-data/outputs/private.pkl"),
  ).toBeHidden();
  await page.setViewportSize({ width: 1920, height: 1080 });
  await expect(page.getByRole("tab", { name: "Final result" })).toBeVisible();
  const wideBox = await drawer.boundingBox();
  expect(wideBox?.width).toBeGreaterThanOrEqual(900);
  expect(wideBox?.width).toBeLessThanOrEqual(1100);
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

test("explains hashed selections in Chinese without showing raw codes by default", async ({ page, context }) => {
  await context.addCookies([{ name: "locale", value: "zh-CN", url: process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000" }]);
  const candidateId = "82cb8f406e10b404";
  const routes = mockLangGraphAPI(page, {
    threads: [{
      thread_id: THREAD_ID,
      title: "Corn calibration",
      values: { nir_workflow: {
        project_id: "corn",
        task_type: "calibration",
        stage: "review",
        revision: 1,
        attempt: 1,
        max_attempts: 1,
        attempts: [{
          attempt: 1,
          method: "svr",
          pipeline_steps: ["snv"],
          protocol: "automated_analysis_three_way_holdout",
          validation_scope: "independent_holdout_not_external",
          decision_facts: {
            preprocessing_selection: {
              selected_candidate_id: candidateId,
              reason_code: "simplest_within_one_percent_rmsecv",
              candidates: [{ candidate_id: candidateId, steps: ["snv"], cv_rmse: 0.2, selected: true }],
            },
            wavelength_selection_decision: {
              selected_method: "none",
              adoption: { reason_code: "full_spectrum_for_non_pls_method" },
            },
            model_selection_decision: {
              selected_method: "svr",
              reason_code: "explicit_or_no_pls_baseline",
              reason: "The caller explicitly requested method='svr'.",
            },
          },
        }],
      } },
    }],
  });
  await routes.ready;
  await page.goto(`/workspace/chats/${THREAD_ID}`);
  await page.getByRole("button", { name: "打开近红外建模过程" }).click();
  await page.getByRole("tab", { name: "选择依据" }).click();
  await expect(page.getByText("本轮选定的方案")).toBeVisible();
  await expect(page.getByText("与误差最低方案的差距不超过 1%，因此采用步骤更少的方案。")).toBeVisible();
  await expect(page.getByText("当前使用的不是 PLS 模型，自动 CARS 比较只适用于 PLS，因此保留全部波长。")).toBeVisible();
  await expect(page.getByText(`原始选择: ${candidateId}`)).toBeHidden();
  await page.getByText("查看原始技术记录").first().click();
  await expect(page.getByText(`原始选择: ${candidateId}`)).toBeVisible();
});

test("shows blocked best-effort result without inventing selection evidence", async ({
  page,
}) => {
  const routes = mockLangGraphAPI(page, {
    threads: [
      {
        thread_id: THREAD_ID,
        title: "Blocked calibration",
        values: {
          nir_workflow: {
            project_id: "blocked-one",
            task_type: "calibration",
            stage: "blocked",
            revision: 7,
            attempt: 1,
            max_attempts: 1,
            approval_status: "not_required",
            attempts: [
              {
                attempt: 1,
                method: "pls",
                pipeline_steps: ["snv"],
                passed: false,
                metrics_summary: { RMSEP: 2.4 },
                validation_scope: "independent_holdout_not_external",
              },
            ],
            reflection: {
              source_attempt: 1,
              should_retry: false,
              stop_reason: "retry_budget_exhausted",
              reason: "Quality threshold not met",
            },
          },
        },
      },
    ],
  });
  await routes.ready;
  await page.goto(`/workspace/chats/${THREAD_ID}`);
  await page.getByRole("button", { name: "Open NIR workflow" }).click();
  await expect(page.getByText("Quality threshold not met")).toBeVisible();
  await page.getByRole("tab", { name: "Selection evidence" }).click();
  await expect(page.getByText("Evidence not recorded").first()).toBeVisible();
  await page.getByRole("tab", { name: "Final result" }).click();
  await expect(page.getByText("Retry budget exhausted")).toBeVisible();
  await expect(page.getByText("Best available evaluation")).toBeVisible();
  await expect(page.getByText("No external validation evidence")).toBeVisible();
});

test("does not carry workflow evidence across chats", async ({ page }) => {
  const otherThread = "00000000-0000-0000-0000-0000000000bb";
  const routes = mockLangGraphAPI(page, {
    threads: [
      {
        thread_id: THREAD_ID,
        title: "First workflow",
        values: {
          nir_workflow: {
            project_id: "first",
            task_type: "calibration",
            stage: "planning",
            revision: 2,
            dataset_id: "ds_first",
          },
        },
      },
      {
        thread_id: otherThread,
        title: "Second workflow",
        values: {
          nir_workflow: {
            project_id: "second",
            task_type: "calibration",
            stage: "planning",
            revision: 2,
            dataset_id: "ds_second",
          },
        },
      },
    ],
  });
  await routes.ready;
  await page.goto(`/workspace/chats/${THREAD_ID}`);
  await page.getByRole("button", { name: "Open NIR workflow" }).click();
  await expect(page.getByText("ds_first")).toBeVisible();
  await page.goto(`/workspace/chats/${otherThread}`);
  await page.getByRole("button", { name: "Open NIR workflow" }).click();
  await expect(page.getByText("ds_second")).toBeVisible();
  await expect(page.getByText("ds_first")).toBeHidden();
});
