import { beforeEach, describe, expect, rs, test } from "@rstest/core";

rs.mock("@/core/api/fetcher", () => ({ fetch: rs.fn() }));
rs.mock("@/core/config", () => ({ getBackendBaseURL: () => "/backend" }));

import { fetch as fetcher } from "@/core/api/fetcher";
import {
  saveMethodCard,
  searchMethodCards,
  setCandidateValue,
} from "@/core/knowledge/method-api";

const mockedFetch = rs.mocked(fetcher);
beforeEach(() => {
  mockedFetch.mockReset();
});

describe("method knowledge management", () => {
  test("edits carry the displayed catalog revision to prevent stale writes", async () => {
    mockedFetch.mockResolvedValue(
      new Response(JSON.stringify({ revision: 2 }), { status: 200 }),
    );
    await saveMethodCard("sg_smooth", { title: "Reviewed title" }, 1);
    const [url, init] = mockedFetch.mock.calls[0]!;
    expect(url).toBe("/backend/api/method-knowledge/cards/sg_smooth");
    expect(JSON.parse(init!.body as string)).toEqual({
      expected_revision: 1,
      changes: { title: "Reviewed title" },
    });
  });
  test("a conflict remains an error instead of silently overwriting", async () => {
    mockedFetch.mockResolvedValue(
      new Response(JSON.stringify({ detail: "Catalog changed; refresh" }), {
        status: 409,
      }),
    );
    await expect(saveMethodCard("sg_smooth", {}, 0)).rejects.toThrow(
      "Catalog changed; refresh",
    );
  });
  test("search preview uses method endpoint independently from paper RAG", async () => {
    mockedFetch.mockResolvedValue(
      new Response(JSON.stringify({ results: [] }), { status: 200 }),
    );
    await searchMethodCards("高频噪声");
    expect(mockedFetch.mock.calls[0]![0]).toBe(
      "/backend/api/method-knowledge/search",
    );
  });
  test("parameter fields preserve zero, remove defaults and reject nonfinite values", () => {
    const schema = { type: "number" as const, default: 1 };
    expect(setCandidateValue({ p: 1 }, "p", "0", schema)).toEqual({ p: 0 });
    expect(setCandidateValue({ p: 1 }, "p", "", schema)).toEqual({});
    expect(() => setCandidateValue({}, "p", "Infinity", schema)).toThrow();
    expect(setCandidateValue({}, "mode", "mean", { type: "string" })).toEqual({
      mode: "mean",
    });
  });
});
