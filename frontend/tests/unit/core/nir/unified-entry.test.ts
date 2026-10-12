import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { expect, test } from "@rstest/core";

test("default composer contains no modeling mode or writable budget control", () => {
  const composer = readFileSync(
    resolve("src/components/workspace/input-box.tsx"),
    "utf8",
  );
  const page = readFileSync(
    resolve("src/app/workspace/chats/[thread_id]/page.tsx"),
    "utf8",
  );
  expect(composer).not.toMatch(
    /AutonomousModeControl|enableAutonomousModeling|autonomousSelection/,
  );
  expect(page).not.toContain("enableAutonomousModeling");
  expect(page).toContain("modelingChatContext");
  expect(
    existsSync(resolve("src/components/workspace/nir/autonomous-mode.tsx")),
  ).toBe(false);
});
