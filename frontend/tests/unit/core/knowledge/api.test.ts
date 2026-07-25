import { beforeEach, describe, expect, test, rs } from "@rstest/core";

rs.mock("@/core/api/fetcher", () => ({
  fetch: rs.fn(),
}));

rs.mock("@/core/config", () => ({
  getBackendBaseURL: () => "/backend",
}));

import { fetch as fetcher } from "@/core/api/fetcher";
import {
  setKnowledgeDocumentStatus,
  updateKnowledgeDocumentMetadata,
  uploadKnowledgeDocumentsSequentially,
} from "@/core/knowledge/api";

const mockedFetch = rs.mocked(fetcher);

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText: status >= 400 ? "Bad Request" : "OK",
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  mockedFetch.mockReset();
});

describe("knowledge upload api", () => {
  test("sends each selected file in its own sequential request", async () => {
    mockedFetch
      .mockResolvedValueOnce(
        jsonResponse(200, {
          results: [
            {
              filename: "first.pdf",
              success: true,
              chunks_added: 10,
            },
          ],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          results: [
            {
              filename: "second.pdf",
              success: true,
              chunks_added: 20,
            },
          ],
        }),
      );

    const files = [
      new File(["first"], "first.pdf", { type: "application/pdf" }),
      new File(["second"], "second.pdf", { type: "application/pdf" }),
    ];
    const results = await uploadKnowledgeDocumentsSequentially(
      files,
      "Shared title",
      "2024",
    );

    expect(results).toHaveLength(2);
    expect(mockedFetch).toHaveBeenCalledTimes(2);
    for (const [index, call] of mockedFetch.mock.calls.entries()) {
      expect(call[0]).toBe("/backend/api/knowledge/documents/batch");
      const request = call[1]!;
      if (!(request.body instanceof FormData)) {
        throw new Error("Expected a multipart FormData request");
      }
      const form = request.body;
      expect(form.getAll("files")).toHaveLength(1);
      const uploadedFile = form.get("files");
      if (!(uploadedFile instanceof File)) {
        throw new Error("Expected one uploaded File");
      }
      expect(uploadedFile.name).toBe(files[index]?.name);
      expect(form.get("title")).toBe("Shared title");
      expect(form.get("year")).toBe("2024");
    }
  });

  test("records one failed file and continues with the next", async () => {
    mockedFetch
      .mockResolvedValueOnce(jsonResponse(504, { detail: "Gateway timeout" }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          results: [{ filename: "second.pdf", success: true, chunks_added: 5 }],
        }),
      );

    const results = await uploadKnowledgeDocumentsSequentially(
      [new File(["first"], "first.pdf"), new File(["second"], "second.pdf")],
      "",
      "",
    );

    expect(results).toEqual([
      {
        filename: "first.pdf",
        success: false,
        error: "Gateway timeout",
      },
      {
        filename: "second.pdf",
        success: true,
        chunks_added: 5,
      },
    ]);
    expect(mockedFetch).toHaveBeenCalledTimes(2);
  });
});

describe("knowledge document status api", () => {
  test("publishes a URL-encoded document id", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(200, {
        doc_id: "doi:10.1000/paper",
        review_status: "published",
      }),
    );

    const result = await setKnowledgeDocumentStatus(
      "doi:10.1000/paper",
      "published",
    );

    expect(mockedFetch).toHaveBeenCalledWith(
      "/backend/api/knowledge/documents/doi%3A10.1000%2Fpaper/status",
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review_status: "published" }),
      },
    );
    expect(result).toEqual({
      doc_id: "doi:10.1000/paper",
      review_status: "published",
    });
  });

  test("surfaces the backend error when publication fails", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(400, { detail: "Document not found" }),
    );

    await expect(
      setKnowledgeDocumentStatus("missing", "published"),
    ).rejects.toThrow("Document not found");
  });
});

describe("knowledge document metadata api", () => {
  test("updates metadata for a URL-encoded document id", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(200, {
        doc_id: "doi:10.1000/paper",
        title: "Updated paper",
        authors: ["Alice", "Bob"],
        year: 2025,
        doi: "10.1000/paper",
        language: "en",
        domains: ["meat"],
        quality_tier: "A",
        review_status: "published",
      }),
    );

    const metadata = {
      title: "Updated paper",
      authors: ["Alice", "Bob"],
      year: 2025,
      doi: "10.1000/paper",
      language: "en",
      domains: ["meat"],
      quality_tier: "A" as const,
    };
    const result = await updateKnowledgeDocumentMetadata(
      "doi:10.1000/paper",
      metadata,
    );

    expect(mockedFetch).toHaveBeenCalledWith(
      "/backend/api/knowledge/documents/doi%3A10.1000%2Fpaper/metadata",
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(metadata),
      },
    );
    expect(result.title).toBe("Updated paper");
    expect(result.quality_tier).toBe("A");
  });

  test("surfaces backend metadata validation errors", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(422, { detail: "Invalid publication year" }),
    );

    await expect(
      updateKnowledgeDocumentMetadata("doc-1", {
        title: "Paper",
        authors: [],
        year: 3025,
        doi: null,
        language: "en",
        domains: [],
        quality_tier: "C",
      }),
    ).rejects.toThrow("Invalid publication year");
  });
});
