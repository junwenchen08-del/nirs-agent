import { expect, test } from "@rstest/core";

import { authErrorMessage } from "@/core/auth/error-message";

test("Chinese authentication feedback explains common errors", () => {
  expect(
    authErrorMessage(
      { code: "token_expired", message: "Token expired" },
      "zh-CN",
    ),
  ).toContain("登录已过期");
  expect(
    authErrorMessage({ detail: "Current password is incorrect" }, "zh-CN"),
  ).toContain("当前密码不正确");
  expect(
    authErrorMessage({ detail: "Email already registered" }, "zh-CN"),
  ).toContain("邮箱已被使用");
  expect(authErrorMessage(null, "zh-CN")).toContain("身份验证失败");
});

test("existing Chinese feedback and explicitly selected English are preserved", () => {
  expect(authErrorMessage({ detail: "该账号暂不可用" }, "zh-CN")).toBe(
    "该账号暂不可用",
  );
  expect(
    authErrorMessage({ detail: "Current password is incorrect" }, "en-US"),
  ).toBe("Current password is incorrect");
});
