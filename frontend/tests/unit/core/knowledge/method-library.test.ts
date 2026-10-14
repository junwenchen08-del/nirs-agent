import { describe, expect, test } from "@rstest/core";

import {
  filterMethodCards,
  methodCategory,
  methodGroup,
} from "@/core/knowledge/method-library";

import { methodCards, methodCatalog } from "../../../fixtures/method-catalog";

describe("method library taxonomy", () => {
  test("groups method cards by purpose rather than the execution interface", () => {
    expect(methodGroup(methodCards[0]!)).toBe("preprocessing");
    expect(methodGroup(methodCards[2]!)).toBe("modeling");
    expect(methodGroup(methodCards[3]!)).toBe("other");
    expect(
      methodGroup({ ...methodCards[0]!, method_kind: "mcp_reference" }),
    ).toBe("preprocessing");
    expect(
      methodGroup({ ...methodCards[2]!, method_kind: "mcp_reference" }),
    ).toBe("modeling");
  });

  test("keeps wavelength alignment separate from ordinary preprocessing", () => {
    expect(methodGroup({ ...methodCards[3]!, category: "adaptation" })).toBe(
      "other",
    );
    expect(methodGroup({ ...methodCards[3]!, category: "unfamiliar" })).toBe(
      "other",
    );
  });

  test("assigns legacy native preprocessors to their scientific category", () => {
    expect(
      methodCategory({
        ...methodCards[0]!,
        method_id: "robust_snv",
        category: "native",
      }),
    ).toBe("scatter");
    expect(
      methodCategory({
        ...methodCards[0]!,
        method_id: "despike",
        category: undefined,
      }),
    ).toBe("smooth");
  });

  test("combines group, subcategory and applicability search without changing cards", () => {
    const before = JSON.stringify(methodCards);
    const selected = filterMethodCards(
      methodCards,
      { group: "preprocessing", category: "scatter", query: "  颗粒大小  " },
      methodCatalog.category_labels,
    );
    expect(selected.map((card) => card.method_id)).toEqual(["snv"]);
    expect(
      filterMethodCards(methodCards, {
        group: "modeling",
        category: "scatter",
        query: "",
      }),
    ).toEqual([]);
    expect(JSON.stringify(methodCards)).toBe(before);
  });

  test("searches identifiers, categories and limitations as well as titles", () => {
    expect(
      filterMethodCards(methodCards, {
        group: "all",
        category: "all",
        query: "窄峰",
      }).map((c) => c.method_id),
    ).toContain("sg_smooth");
    expect(
      filterMethodCards(methodCards, {
        group: "all",
        category: "all",
        query: "SNV",
      }).map((c) => c.method_id),
    ).toEqual(["snv"]);
    expect(
      filterMethodCards(
        methodCards,
        { group: "all", category: "all", query: "散射校正" },
        methodCatalog.category_labels,
      ).map((c) => c.method_id),
    ).toEqual(["snv"]);
  });
});
