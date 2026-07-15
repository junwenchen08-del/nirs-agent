"use client";

import {
  FileIcon,
  Loader2Icon,
  RefreshCwIcon,
  SearchIcon,
  Trash2Icon,
  UploadIcon,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { getBackendBaseURL } from "@/core/config";
import { fetch } from "@/core/api/fetcher";
import { useI18n } from "@/core/i18n/hooks";
import { toast } from "sonner";

import { SettingsSection } from "./settings-section";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface KnowledgeDocument {
  doc_id: string;
  title: string;
  source: string;
  year: number | null;
  chunk_count: number;
}

interface KnowledgeStats {
  documents: number;
  total_chunks: number;
}

interface KnowledgeSearchResult {
  content: string;
  source: string;
  score: number;
  entities: {
    methods: string[];
    models: string[];
    datasets: string[];
    metrics: string[];
  };
  related_entities: string[];
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

// Supported file extensions (must match backend _SUPPORTED_EXTS)
const SUPPORTED_EXTS = [".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".csv"];
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB

async function readErrorDetail(response: Response): Promise<string> {
  const data = (await response.json().catch(() => ({}))) as { detail?: string };
  return data.detail ?? `HTTP ${response.status}: ${response.statusText}`;
}

async function apiListDocuments(): Promise<KnowledgeDocument[]> {
  const resp = await fetch(`${getBackendBaseURL()}/api/knowledge/documents`);
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  const json = await resp.json();
  return json.documents as KnowledgeDocument[];
}

async function apiGetStats(): Promise<KnowledgeStats> {
  const resp = await fetch(`${getBackendBaseURL()}/api/knowledge/stats`);
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  const json = await resp.json();
  return {
    documents: json.documents,
    total_chunks: json.total_chunks,
  };
}

async function apiUploadDocument(
  file: File,
  title: string,
  year: string,
): Promise<{ chunks_added: number; doc_id: string }> {
  const form = new FormData();
  form.append("file", file);
  if (title) form.append("title", title);
  if (year) form.append("year", year);

  const resp = await fetch(`${getBackendBaseURL()}/api/knowledge/documents`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  return resp.json();
}

async function apiDeleteDocument(docId: string): Promise<void> {
  const encoded = encodeURIComponent(docId);
  const resp = await fetch(
    `${getBackendBaseURL()}/api/knowledge/documents/${encoded}`,
    { method: "DELETE" },
  );
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
}

async function apiSearch(
  query: string,
  topK: number = 5,
): Promise<KnowledgeSearchResult[]> {
  const resp = await fetch(`${getBackendBaseURL()}/api/knowledge/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  const json = await resp.json();
  return json.results as KnowledgeSearchResult[];
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function KnowledgeSettingsPage() {
  const { t } = useI18n();
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [uploadOpen, setUploadOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<KnowledgeDocument | null>(null);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [docs, s] = await Promise.all([apiListDocuments(), apiGetStats()]);
      setDocuments(docs);
      setStats(s);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  const handleUploadSuccess = useCallback(
    (chunksAdded: number) => {
      toast.success(t.settings.knowledge.uploadSuccess.replace("{count}", String(chunksAdded)));
      setUploadOpen(false);
      void loadAll();
    },
    [loadAll, t.settings.knowledge.uploadSuccess],
  );

  const handleDelete = useCallback(
    async (doc: KnowledgeDocument) => {
      try {
        await apiDeleteDocument(doc.doc_id);
        toast.success(
          t.settings.knowledge.deleteSuccess.replace("{docId}", doc.doc_id),
        );
        setDeleteTarget(null);
        void loadAll();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        toast.error(t.settings.knowledge.deleteFailed.replace("{message}", msg));
      }
    },
    [loadAll, t.settings.knowledge.deleteSuccess, t.settings.knowledge.deleteFailed],
  );

  return (
    <SettingsSection
      title={t.settings.knowledge.title}
      description={t.settings.knowledge.description}
    >
      <div className="flex flex-col gap-6">
        {/* Stats + toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex gap-6">
            <StatCard
              label={t.settings.knowledge.statsDocuments}
              value={stats?.documents}
              loading={loading}
            />
            <StatCard
              label={t.settings.knowledge.statsChunks}
              value={stats?.total_chunks}
              loading={loading}
            />
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => void loadAll()}
              disabled={loading}
            >
              <RefreshCwIcon className={`size-4 ${loading ? "animate-spin" : ""}`} />
              {t.settings.knowledge.refreshButton}
            </Button>
            <Button size="sm" onClick={() => setUploadOpen(true)}>
              <UploadIcon className="size-4" />
              {t.settings.knowledge.uploadButton}
            </Button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <Alert variant="destructive">
            <AlertDescription>
              {t.settings.knowledge.error}: {error}
              {error.includes("unreachable") || error.includes("503") ? (
                <>
                  {" — "}
                  {t.settings.knowledge.serverUnreachable}
                </>
              ) : null}
            </AlertDescription>
          </Alert>
        )}

        {/* Document list */}
        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        ) : documents.length === 0 ? (
          <div className="text-muted-foreground rounded-lg border border-dashed p-8 text-center text-sm">
            {t.settings.knowledge.empty}
          </div>
        ) : (
          <DocumentTable
            documents={documents}
            onDelete={(doc) => setDeleteTarget(doc)}
          />
        )}

        {/* Search test */}
        <SearchTest onSearch={apiSearch} />

        {/* Upload dialog */}
        <UploadDialog
          open={uploadOpen}
          onOpenChange={setUploadOpen}
          onSuccess={handleUploadSuccess}
        />

        {/* Delete confirm dialog */}
        <Dialog
          open={deleteTarget !== null}
          onOpenChange={(open) => !open && setDeleteTarget(null)}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{t.settings.knowledge.deleteConfirmTitle}</DialogTitle>
              <DialogDescription>
                {t.settings.knowledge.deleteConfirmDescription}
              </DialogDescription>
            </DialogHeader>
            {deleteTarget && (
              <div className="bg-muted rounded-md p-3 text-sm">
                <div className="font-medium">{deleteTarget.title || deleteTarget.doc_id}</div>
                {deleteTarget.source && (
                  <div className="text-muted-foreground mt-1 break-all text-xs">
                    {deleteTarget.source}
                  </div>
                )}
                <div className="text-muted-foreground mt-1">
                  {deleteTarget.chunk_count} {t.settings.knowledge.statsChunks}
                </div>
              </div>
            )}
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline">{t.common.cancel}</Button>
              </DialogClose>
              <Button
                variant="destructive"
                onClick={() => deleteTarget && void handleDelete(deleteTarget)}
              >
                <Trash2Icon className="size-4" />
                {t.settings.knowledge.deleteButton}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </SettingsSection>
  );
}

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

function StatCard({
  label,
  value,
  loading,
}: {
  label: string;
  value: number | undefined;
  loading: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="text-2xl font-semibold">
        {loading ? "…" : value ?? 0}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Document table
// ---------------------------------------------------------------------------

function DocumentTable({
  documents,
  onDelete,
}: {
  documents: KnowledgeDocument[];
  onDelete: (doc: KnowledgeDocument) => void;
}) {
  const { t } = useI18n();
  return (
    <div className="overflow-hidden rounded-lg border">
      <table className="w-full text-sm">
        <thead className="bg-muted/50">
          <tr>
            <th className="px-3 py-2 text-left font-medium">
              {t.settings.knowledge.columnTitle}
            </th>
            <th className="px-3 py-2 text-left font-medium">
              {t.settings.knowledge.columnYear}
            </th>
            <th className="px-3 py-2 text-left font-medium">
              {t.settings.knowledge.columnChunks}
            </th>
            <th className="px-3 py-2 text-right font-medium">
              {t.settings.knowledge.columnActions}
            </th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr key={doc.doc_id} className="border-t">
              <td className="px-3 py-2">
                <div className="flex items-start gap-2">
                  <FileIcon className="text-muted-foreground mt-0.5 size-4 shrink-0" />
                  <div className="min-w-0">
                    <div className="truncate font-medium">
                      {doc.title || doc.doc_id}
                    </div>
                    <div className="text-muted-foreground truncate text-xs">
                      {doc.source || doc.doc_id}
                    </div>
                  </div>
                </div>
              </td>
              <td className="px-3 py-2">
                {doc.year ?? t.settings.knowledge.noYear}
              </td>
              <td className="px-3 py-2">{doc.chunk_count}</td>
              <td className="px-3 py-2 text-right">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onDelete(doc)}
                >
                  <Trash2Icon className="size-4" />
                  <span className="sr-only">{t.settings.knowledge.deleteButton}</span>
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload dialog
// ---------------------------------------------------------------------------

function UploadDialog({
  open,
  onOpenChange,
  onSuccess,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: (chunksAdded: number) => void;
}) {
  const { t } = useI18n();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [year, setYear] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSelectFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError(t.settings.knowledge.uploadHint);
      return;
    }
    // Client-side file type validation
    const fileName = file.name.toLowerCase();
    const isValidExt = SUPPORTED_EXTS.some((ext) => fileName.endsWith(ext));
    if (!isValidExt) {
      setError(t.settings.knowledge.unsupportedType.replace("{ext}", SUPPORTED_EXTS.join(", ")));
      return;
    }
    // Client-side file size validation
    if (file.size > MAX_FILE_SIZE) {
      setError(
        t.settings.knowledge.fileTooLarge.replace(
          "{size}",
          String(Math.round(file.size / 1024 / 1024)),
        ),
      );
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const result = await apiUploadDocument(file, title, year);
      onSuccess(result.chunks_added);
      // Reset
      setFile(null);
      setTitle("");
      setYear("");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      toast.error(t.settings.knowledge.uploadFailed.replace("{message}", msg));
    } finally {
      setUploading(false);
    }
  };

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      // Reset on close
      setFile(null);
      setTitle("");
      setYear("");
      setError(null);
    }
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t.settings.knowledge.uploadButton}</DialogTitle>
          <DialogDescription>{t.settings.knowledge.uploadHint}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* File picker */}
          <div className="space-y-2">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.docx,.txt,.md,.markdown,.html,.htm,.csv"
              onChange={handleFileChange}
            />
            <Button
              variant="outline"
              onClick={handleSelectFile}
              disabled={uploading}
              className="w-full justify-start"
            >
              <FileIcon className="size-4" />
              {file ? file.name : t.settings.knowledge.uploadButton}
            </Button>
          </div>

          {/* Title */}
          <div className="space-y-1">
            <label className="text-sm font-medium">
              {t.settings.knowledge.titleLabel}
            </label>
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder={t.settings.knowledge.titlePlaceholder}
              disabled={uploading}
            />
          </div>

          {/* Year */}
          <div className="space-y-1">
            <label className="text-sm font-medium">
              {t.settings.knowledge.yearLabel}
            </label>
            <Input
              type="number"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              placeholder={t.settings.knowledge.yearPlaceholder}
              disabled={uploading}
            />
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline" disabled={uploading}>
              {t.common.cancel}
            </Button>
          </DialogClose>
          <Button onClick={() => void handleUpload()} disabled={uploading || !file}>
            {uploading ? (
              <Loader2Icon className="size-4 animate-spin" />
            ) : (
              <UploadIcon className="size-4" />
            )}
            {t.settings.knowledge.uploadButton}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Search test
// ---------------------------------------------------------------------------

function SearchTest({
  onSearch,
}: {
  onSearch: (query: string, topK?: number) => Promise<KnowledgeSearchResult[]>;
}) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<KnowledgeSearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const r = await onSearch(query.trim(), 5);
      setResults(r);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setResults(null);
    } finally {
      setSearching(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSearch();
    }
  };

  return (
    <div className="space-y-3 border-t pt-6">
      <div className="text-base font-semibold">{t.settings.knowledge.searchTitle}</div>
      <div className="flex gap-2">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t.settings.knowledge.searchPlaceholder}
          disabled={searching}
        />
        <Button onClick={() => void handleSearch()} disabled={searching || !query.trim()}>
          {searching ? (
            <Loader2Icon className="size-4 animate-spin" />
          ) : (
            <SearchIcon className="size-4" />
          )}
          {t.settings.knowledge.searchButton}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>
            {t.settings.knowledge.error}: {error}
          </AlertDescription>
        </Alert>
      )}

      {results && results.length === 0 && (
        <div className="text-muted-foreground text-sm">
          {t.settings.knowledge.searchEmpty}
        </div>
      )}

      {results && results.length > 0 && (
        <div className="space-y-3">
          {results.map((r, i) => (
            <div key={`${r.source}-${i}`} className="rounded-lg border p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-muted-foreground truncate text-xs">
                  {r.source}
                </div>
                <Badge variant="secondary" className="shrink-0">
                  score: {r.score.toFixed(4)}
                </Badge>
              </div>
              <p className="line-clamp-4 text-sm leading-relaxed">{r.content}</p>
              {r.entities.methods.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {r.entities.methods.map((m) => (
                    <Badge key={m} variant="outline" className="text-xs">
                      {m}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
