import type { Locale } from "@/core/i18n/locale";

import { parseAuthError } from "./types";

/** Keep authentication feedback actionable for the selected interface language. */
export function authErrorMessage(data: unknown, locale: Locale): string {
  const error = parseAuthError(data);
  if (locale !== "zh-CN" || /\p{Script=Han}/u.test(error.message))
    return error.message;
  if (error.message === "Authentication failed")
    return "身份验证失败，请稍后重试。";
  if (/current password|old password/i.test(error.message))
    return "当前密码不正确，请重新输入。";
  if (
    /password.*(?:at least|minimum|min\.|too short|8 characters)/i.test(
      error.message,
    )
  )
    return "密码至少需要 8 个字符。";
  if (/password.*(?:match|different)/i.test(error.message))
    return "两次输入的密码不一致。";
  if (/email.*(?:exists|registered|in use)/i.test(error.message))
    return "该邮箱已被使用，请直接登录或更换邮箱。";
  const messages: Record<string, string> = {
    invalid_credentials: "邮箱或密码不正确，请检查后重试。",
    token_expired: "登录已过期，请重新登录。",
    token_invalid: "登录状态无效，请重新登录。",
    user_not_found: "未找到该账号，请检查邮箱。",
    email_already_exists: "该邮箱已被使用，请直接登录或更换邮箱。",
    provider_not_found: "此登录方式暂不可用，请联系管理员。",
    not_authenticated: "请先登录后再操作。",
    system_already_initialized: "系统已完成初始化，请前往登录页。",
  };
  return messages[error.code] ?? "操作未完成，请检查输入信息后重试。";
}
