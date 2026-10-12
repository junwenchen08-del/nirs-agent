import type { Message } from "@langchain/langgraph-sdk";

import { parseSummary } from "@/core/nir/autonomous";

/** Discover only a server-generated delivery's session, never infer success. */
export function nirDeliverySession(message: Message): string | null {
  if (
    message.type !== "ai" ||
    message.additional_kwargs?.nir_deterministic_delivery !== true
  )
    return null;
  const session = message.additional_kwargs.nir_delivery_session_id;
  if (typeof session === "string" && /^search_[A-Za-z0-9]+$/.test(session))
    return session;
  // Older immutable messages have links but no session metadata. Actual file
  // availability is still checked by the owner-scoped read-only publication API.
  if (typeof message.content !== "string") return null;
  return (
    /\]\(\/mnt\/user-data\/outputs\/(search_[A-Za-z0-9]+)\/delivery_[A-Za-z0-9]+\/delivery\.zip\)/.exec(
      message.content,
    )?.[1] ?? null
  );
}

export function nirDeliveryFiles(raw: unknown, session: string): string[] {
  const summary = parseSummary(raw);
  if (
    summary?.session_id !== session ||
    summary.stage !== "finalized" ||
    summary.export_status !== "succeeded"
  )
    return [];
  const prefix = `mnt/user-data/outputs/${session}/`;
  return [
    ...new Set(
      summary.downloads
        .filter(
          (file) =>
            file.path.startsWith(prefix) &&
            file.path.slice(file.path.lastIndexOf("/") + 1) === file.name,
        )
        .map((file) => `/${file.path}`),
    ),
  ];
}

/** Fold only recognizable NIR output sets; unrelated artifacts stay visible. */
export function groupDeliveryFiles(files: string[]) {
  const unique = [...new Set(files)];
  const directories = new Set<string>();
  const collections = new Set<string>();
  for (const path of unique) {
    const directory = path.slice(0, path.lastIndexOf("/") + 1);
    const name = path.slice(directory.length);
    if (/^collection_summary\.(md|html)$/.test(name))
      collections.add(directory);
    if (
      name === "delivery.zip" ||
      (/^(report|auto_split_report|partitioned_report)\.(md|html)$/.test(
        name,
      ) &&
        unique.some(
          (file) =>
            file.startsWith(directory) &&
            !file.slice(directory.length).includes("/") &&
            file.endsWith(".pkl"),
        ) &&
        unique.some(
          (file) => file.startsWith(directory) && file.endsWith(".json"),
        ))
    )
      directories.add(directory);
  }
  const supporting: string[] = [];
  const primary = unique.filter((path) => {
    const directory = path.slice(0, path.lastIndexOf("/") + 1);
    const name = path.slice(directory.length);
    const inCollection = [...collections].some(
      (root) => directory.startsWith(root) && directory !== root,
    );
    const known =
      /^(?:report\.md|(?:auto_split|partitioned)_report\.md|metrics\.json|model\.pkl(?:\.manifest\.json)?|raw_spectra\.png|predicted_vs_reference\.png|residuals\.png|cv_curve\.png|vip_scores\.png|regression_coefficients\.png)$/.test(
        name,
      );
    const redundantMarkdown =
      /^(?:report|auto_split_report|partitioned_report|collection_summary)\.md$/.test(
        name,
      ) && unique.includes(path.replace(/\.md$/, ".html"));
    const fold =
      redundantMarkdown ||
      (inCollection && known) ||
      (directories.has(directory) && known && !name.endsWith("report.md"));
    if (fold) supporting.push(path);
    return !fold;
  });
  return { primary, supporting };
}

export function resolveReportImage(
  src: string,
  reportURL?: string,
): string | undefined {
  if (/^(https?:|data:|\/)/.test(src)) return src;
  if (
    !reportURL ||
    src.split("/").some((part) => part === "..") ||
    src.includes("\\")
  )
    return undefined;
  try {
    return new URL(src, new URL(reportURL, "http://artifact.local"))
      .toString()
      .replace(/^http:\/\/artifact\.local/, "");
  } catch {
    return undefined;
  }
}
