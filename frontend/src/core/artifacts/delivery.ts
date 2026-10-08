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
