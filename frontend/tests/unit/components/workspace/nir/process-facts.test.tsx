import { expect, test } from "@rstest/core";
import { renderToStaticMarkup } from "react-dom/server";

import { ProcessFacts } from "@/components/workspace/nir/process-facts";

const range = (length: number) => Array.from({ length }, (_, index) => index);

/**
 * A search grid is a short scalar list. Rendered as numbered bordered rows it
 * reads like a stack of disabled inputs and pushes the round detail off screen,
 * so short scalar lists must stay inline chips.
 */
test("short scalar search grids render as inline chips, not boxed rows", () => {
  const html = renderToStaticMarkup(
    <ProcessFacts zh value={{ search_space: { n_components: range(10) } }} />,
  );
  expect(html.match(/<li/g)).toHaveLength(10);
  expect(html).toContain("bg-muted");
  expect(html).not.toContain("rounded-md border p-2");
});

test("record options and long lists keep the numbered bordered rows", () => {
  const options = renderToStaticMarkup(
    <ProcessFacts zh value={{ comparisons: [{ previous: "第 1 轮" }] }} />,
  );
  expect(options).toContain("rounded-md border p-2");
  expect(options).not.toContain("bg-muted rounded px-2");

  const long = renderToStaticMarkup(
    <ProcessFacts zh value={{ thresholds_used: range(13) }} />,
  );
  expect(long).toContain("rounded-md border p-2");
});
