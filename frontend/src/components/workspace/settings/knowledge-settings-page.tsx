"use client";

import {
  CheckIcon,
  FileIcon,
  Loader2Icon,
  PencilIcon,
  RefreshCwIcon,
  SearchIcon,
  Trash2Icon,
  UploadIcon,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";
import { useI18n } from "@/core/i18n/hooks";
import {
  type KnowledgeDocumentMetadata,
  type KnowledgeMetadataResult,
  type KnowledgeQualityTier,
  type KnowledgeReviewStatus,
  setKnowledgeDocumentStatus,
  updateKnowledgeDocumentMetadata,
  uploadKnowledgeDocumentsSequentially,
} from "@/core/knowledge/api";

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
  review_status: KnowledgeReviewStatus;
  authors: string[];
  doi: string | null;
  language: string;
  domains: string[];
  quality_tier: KnowledgeQualityTier;
  publication_readiness?: {
    ready: boolean;
    missing_fields: string[];
    warnings: Array<{ code: string; message: string }>;
    doi_status: "valid" | "missing" | "not_applicable";
    message: string;
  } | null;
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

interface KnowledgeRetrievalDiagnostics {
  abstained: boolean;
  reason: string;
  candidate_count: number;
  result_count: number;
  top_score: number | null;
  runner_up_document_score: number | null;
  document_margin: number | null;
}

interface KnowledgeSearchResponse {
  results: KnowledgeSearchResult[];
  retrieval: KnowledgeRetrievalDiagnostics | null;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

// Supported file extensions (must match backend _SUPPORTED_EXTS)
const SUPPORTED_EXTS = [
  ".pdf",
  ".docx",
  ".txt",
  ".md",
  ".markdown",
  ".html",
  ".htm",
  ".csv",
];
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
  topK = 5,
): Promise<KnowledgeSearchResponse> {
  const resp = await fetch(`${getBackendBaseURL()}/api/knowledge/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  const json = await resp.json();
  return {
    results: json.results as KnowledgeSearchResult[],
    retrieval:
      (json.retrieval as KnowledgeRetrievalDiagnostics | undefined) ?? null,
  };
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
  const [deleteTarget, setDeleteTarget] = useState<KnowledgeDocument | null>(
    null,
  );
  const [publishingDocId, setPublishingDocId] = useState<string | null>(null);
  const [editTarget, setEditTarget] = useState<KnowledgeDocument | null>(null);

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
      toast.success(
        t.settings.knowledge.uploadSuccess.replace(
          "{count}",
          String(chunksAdded),
        ),
      );
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
        toast.error(
          t.settings.knowledge.deleteFailed.replace("{message}", msg),
        );
      }
    },
    [
      loadAll,
      t.settings.knowledge.deleteSuccess,
      t.settings.knowledge.deleteFailed,
    ],
  );

  const handlePublish = useCallback(
    async (doc: KnowledgeDocument) => {
      setPublishingDocId(doc.doc_id);
      try {
        const updated = await setKnowledgeDocumentStatus(
          doc.doc_id,
          "published",
        );
        setDocuments((current) =>
          current.map((item) =>
            item.doc_id === updated.doc_id
              ? { ...item, review_status: updated.review_status }
              : item,
          ),
        );
        toast.success(
          t.settings.knowledge.publishSuccess.replace(
            "{title}",
            doc.title || doc.doc_id,
          ),
        );
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        toast.error(
          t.settings.knowledge.publishFailed.replace("{message}", msg),
        );
      } finally {
        setPublishingDocId(null);
      }
    },
    [t.settings.knowledge.publishFailed, t.settings.knowledge.publishSuccess],
  );

  const handleMetadataSaved = useCallback(
    (updated: KnowledgeMetadataResult) => {
      setDocuments((current) =>
        current.map((item) =>
          item.doc_id === updated.doc_id ? { ...item, ...updated } : item,
        ),
      );
      setEditTarget(null);
      toast.success(
        t.settings.knowledge.metadataUpdateSuccess.replace(
          "{title}",
          updated.title || updated.doc_id,
        ),
      );
    },
    [t.settings.knowledge.metadataUpdateSuccess],
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
              <RefreshCwIcon
                className={`size-4 ${loading ? "animate-spin" : ""}`}
              />
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
            onEdit={setEditTarget}
            onPublish={(doc) => void handlePublish(doc)}
            publishingDocId={publishingDocId}
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

        {editTarget && (
          <EditMetadataDialog
            key={editTarget.doc_id}
            document={editTarget}
            onClose={() => setEditTarget(null)}
            onSaved={handleMetadataSaved}
          />
        )}

        {/* Delete confirm dialog */}
        <Dialog
          open={deleteTarget !== null}
          onOpenChange={(open) => !open && setDeleteTarget(null)}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {t.settings.knowledge.deleteConfirmTitle}
              </DialogTitle>
              <DialogDescription>
                {t.settings.knowledge.deleteConfirmDescription}
              </DialogDescription>
            </DialogHeader>
            {deleteTarget && (
              <div className="bg-muted rounded-md p-3 text-sm">
                <div className="font-medium">
                  {deleteTarget.title || deleteTarget.doc_id}
                </div>
                {deleteTarget.source && (
                  <div className="text-muted-foreground mt-1 text-xs break-all">
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
        {loading ? "…" : (value ?? 0)}
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
  onEdit,
  onPublish,
  publishingDocId,
}: {
  documents: KnowledgeDocument[];
  onDelete: (doc: KnowledgeDocument) => void;
  onEdit: (doc: KnowledgeDocument) => void;
  onPublish: (doc: KnowledgeDocument) => void;
  publishingDocId: string | null;
}) {
  const { t } = useI18n();
  const statusLabels: Record<KnowledgeReviewStatus, string> = {
    draft: t.settings.knowledge.statusDraft,
    needs_review: t.settings.knowledge.statusNeedsReview,
    published: t.settings.knowledge.statusPublished,
    retired: t.settings.knowledge.statusRetired,
  };
  const statusVariants: Record<
    KnowledgeReviewStatus,
    "default" | "secondary" | "destructive" | "outline"
  > = {
    draft: "secondary",
    needs_review: "outline",
    published: "default",
    retired: "destructive",
  };

  return (
    <div className="w-full overflow-x-auto rounded-lg border">
      <table className="w-full min-w-[640px] table-fixed text-sm">
        <colgroup>
          <col />
          <col className="w-16" />
          <col className="w-16" />
          <col className="w-20" />
          <col className="w-32" />
        </colgroup>
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
            <th className="px-3 py-2 text-left font-medium">
              {t.settings.knowledge.columnStatus}
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
              <td className="px-3 py-2">
                <Badge variant={statusVariants[doc.review_status]}>
                  {statusLabels[doc.review_status]}
                </Badge>
              </td>
              <td className="px-3 py-2 text-right">
                <div className="flex justify-end gap-1">
                  {doc.review_status !== "published" && (
                    <Button
                      variant="outline"
                      size="icon-sm"
                      onClick={() => onPublish(doc)}
                      disabled={
                        publishingDocId !== null ||
                        doc.publication_readiness?.ready === false
                      }
                      title={
                        doc.publication_readiness?.ready === false ||
                        doc.publication_readiness?.warnings?.length
                          ? doc.publication_readiness.message
                          : undefined
                      }
                      aria-label={
                        publishingDocId === doc.doc_id
                          ? t.settings.knowledge.publishingButton
                          : t.settings.knowledge.publishButton
                      }
                    >
                      {publishingDocId === doc.doc_id ? (
                        <Loader2Icon className="size-4 animate-spin" />
                      ) : (
                        <CheckIcon className="size-4" />
                      )}
                    </Button>
                  )}
                  <Button
                    variant="outline"
                    size="icon-sm"
                    onClick={() => onEdit(doc)}
                    disabled={publishingDocId !== null}
                    title={t.settings.knowledge.editButton}
                    aria-label={t.settings.knowledge.editButton}
                  >
                    <PencilIcon className="size-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => onDelete(doc)}
                    disabled={publishingDocId === doc.doc_id}
                    title={t.settings.knowledge.deleteButton}
                    aria-label={t.settings.knowledge.deleteButton}
                  >
                    <Trash2Icon className="size-4" />
                  </Button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metadata dialog
// ---------------------------------------------------------------------------

function parseMetadataList(value: string): string[] {
  return [
    ...new Set(
      value
        .split(/[,;，；\n]/)
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  ];
}

function EditMetadataDialog({
  document,
  onClose,
  onSaved,
}: {
  document: KnowledgeDocument;
  onClose: () => void;
  onSaved: (updated: KnowledgeMetadataResult) => void;
}) {
  const { t } = useI18n();
  const [title, setTitle] = useState(document.title);
  const [authors, setAuthors] = useState(document.authors.join(", "));
  const [year, setYear] = useState(
    document.year === null ? "" : String(document.year),
  );
  const [doi, setDoi] = useState(document.doi ?? "");
  const [language, setLanguage] = useState(document.language);
  const [domains, setDomains] = useState(document.domains.join(", "));
  const [qualityTier, setQualityTier] = useState<KnowledgeQualityTier>(
    document.quality_tier,
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    const parsedYear = year.trim() ? Number(year) : null;
    if (
      parsedYear !== null &&
      (!Number.isInteger(parsedYear) || parsedYear < 1000 || parsedYear > 2100)
    ) {
      setError(t.settings.knowledge.invalidYear);
      return;
    }

    const metadata: KnowledgeDocumentMetadata = {
      title: title.trim(),
      authors: parseMetadataList(authors),
      year: parsedYear,
      doi: doi.trim() || null,
      language: language.trim() || "und",
      domains: parseMetadataList(domains),
      quality_tier: qualityTier,
    };

    setSaving(true);
    setError(null);
    try {
      const updated = await updateKnowledgeDocumentMetadata(
        document.doc_id,
        metadata,
      );
      onSaved(updated);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
      toast.error(
        t.settings.knowledge.metadataUpdateFailed.replace("{message}", message),
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t.settings.knowledge.editMetadataTitle}</DialogTitle>
          <DialogDescription>
            {t.settings.knowledge.editMetadataDescription}
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-2 sm:grid-cols-2">
          <div className="space-y-1 sm:col-span-2">
            <label className="text-sm font-medium">
              {t.settings.knowledge.titleLabel}
            </label>
            <Input
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              disabled={saving}
            />
          </div>

          <div className="space-y-1 sm:col-span-2">
            <label className="text-sm font-medium">
              {t.settings.knowledge.authorsLabel}
            </label>
            <Input
              value={authors}
              onChange={(event) => setAuthors(event.target.value)}
              placeholder={t.settings.knowledge.authorsPlaceholder}
              disabled={saving}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">
              {t.settings.knowledge.yearLabel}
            </label>
            <Input
              type="number"
              min={1000}
              max={2100}
              value={year}
              onChange={(event) => setYear(event.target.value)}
              disabled={saving}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">
              {t.settings.knowledge.qualityTierLabel}
            </label>
            <Select
              value={qualityTier}
              onValueChange={(value) =>
                setQualityTier(value as KnowledgeQualityTier)
              }
              disabled={saving}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(["A", "B", "C", "D", "E"] as const).map((tier) => (
                  <SelectItem key={tier} value={tier}>
                    {tier}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1 sm:col-span-2">
            <label className="text-sm font-medium">
              {t.settings.knowledge.doiLabel}
            </label>
            <Input
              value={doi}
              onChange={(event) => setDoi(event.target.value)}
              placeholder={t.settings.knowledge.doiPlaceholder}
              disabled={saving}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">
              {t.settings.knowledge.languageLabel}
            </label>
            <Input
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
              placeholder={t.settings.knowledge.languagePlaceholder}
              disabled={saving}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">
              {t.settings.knowledge.domainsLabel}
            </label>
            <Input
              value={domains}
              onChange={(event) => setDomains(event.target.value)}
              placeholder={t.settings.knowledge.domainsPlaceholder}
              disabled={saving}
            />
          </div>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={saving}>
            {t.common.cancel}
          </Button>
          <Button onClick={() => void handleSave()} disabled={saving}>
            {saving && <Loader2Icon className="size-4 animate-spin" />}
            {saving
              ? t.settings.knowledge.savingMetadataButton
              : t.settings.knowledge.saveMetadataButton}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
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
  const [files, setFiles] = useState<File[]>([]);
  const [title, setTitle] = useState("");
  const [year, setYear] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSelectFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files;
    if (selected && selected.length > 0) {
      const newFiles = Array.from(selected);
      // Validate and collect errors for all files
      const rejected: string[] = [];
      const accepted: File[] = [];
      for (const f of newFiles) {
        const fn = f.name.toLowerCase();
        const isValidExt = SUPPORTED_EXTS.some((ext) => fn.endsWith(ext));
        if (!isValidExt) {
          rejected.push(
            t.settings.knowledge.unsupportedType.replace(
              "{ext}",
              SUPPORTED_EXTS.join(", "),
            ) + ` (${f.name})`,
          );
          continue;
        }
        if (f.size > MAX_FILE_SIZE) {
          rejected.push(
            t.settings.knowledge.fileTooLarge.replace(
              "{size}",
              String(Math.round(f.size / 1024 / 1024)),
            ) + ` (${f.name})`,
          );
          continue;
        }
        accepted.push(f);
      }
      // Avoid duplicate filenames
      setFiles((prev) => {
        const existing = new Set(prev.map((f) => f.name));
        const merged = [...prev];
        for (const f of accepted) {
          if (!existing.has(f.name)) {
            merged.push(f);
            existing.add(f.name);
          }
        }
        return merged;
      });
      if (rejected.length > 0) {
        setError(rejected.join("\n"));
      } else {
        setError(null);
      }
    }
    // Reset input so selecting the same file again still fires onChange
    e.target.value = "";
  };

  const handleRemoveFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError(t.settings.knowledge.uploadHint);
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const results = await uploadKnowledgeDocumentsSequentially(
        files,
        title,
        year,
      );
      const totalChunks = results
        .filter((r) => r.success)
        .reduce((sum, r) => sum + (r.chunks_added ?? 0), 0);
      const failed = results.filter((r) => !r.success);
      if (failed.length > 0) {
        const failedMsgs = failed
          .map((r) => `${r.filename}: ${r.error ?? "unknown error"}`)
          .join("; ");
        if (results.every((r) => !r.success)) {
          throw new Error(failedMsgs);
        } else {
          // Partial success
          toast.error(
            t.settings.knowledge.batchPartialFailed.replace(
              "{message}",
              failedMsgs,
            ),
          );
        }
      }
      onSuccess(totalChunks);
      // Reset
      setFiles([]);
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
      setFiles([]);
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
          <DialogDescription>
            {t.settings.knowledge.uploadHint}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* File picker (supports multiple files) */}
          <div className="space-y-2">
            <input
              ref={fileInputRef}
              type="file"
              multiple
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
              <UploadIcon className="size-4" />
              {files.length > 0
                ? t.settings.knowledge.batchFilesSelected.replace(
                    "{count}",
                    String(files.length),
                  )
                : t.settings.knowledge.batchSelectFiles}
            </Button>
            {/* Selected files list */}
            {files.length > 0 && (
              <div className="max-h-40 space-y-1 overflow-y-auto rounded-md border p-2">
                {files.map((f, idx) => (
                  <div
                    key={`${f.name}-${idx}`}
                    className="flex items-center justify-between gap-2 text-sm"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      <FileIcon className="text-muted-foreground size-4 shrink-0" />
                      <span className="truncate">{f.name}</span>
                      <span className="text-muted-foreground shrink-0 text-xs">
                        {(f.size / 1024).toFixed(0)} KB
                      </span>
                    </div>
                    {!uploading && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="size-6 p-0"
                        onClick={() => handleRemoveFile(idx)}
                      >
                        <Trash2Icon className="size-3" />
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            )}
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
              <AlertDescription className="whitespace-pre-line">
                {error}
              </AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline" disabled={uploading}>
              {t.common.cancel}
            </Button>
          </DialogClose>
          <Button
            onClick={() => void handleUpload()}
            disabled={uploading || files.length === 0}
          >
            {uploading ? (
              <Loader2Icon className="size-4 animate-spin" />
            ) : (
              <UploadIcon className="size-4" />
            )}
            {uploading
              ? t.settings.knowledge.batchUploading
              : t.settings.knowledge.uploadButton}
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
  onSearch: (query: string, topK?: number) => Promise<KnowledgeSearchResponse>;
}) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<KnowledgeSearchResult[] | null>(null);
  const [retrieval, setRetrieval] =
    useState<KnowledgeRetrievalDiagnostics | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const response = await onSearch(query.trim(), 5);
      setResults(response.results);
      setRetrieval(response.retrieval);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setResults(null);
      setRetrieval(null);
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

  const abstentionReason =
    retrieval?.reason === "below_weak_score"
      ? t.settings.knowledge.searchReasonLowScore
      : retrieval?.reason === "ambiguous_across_documents"
        ? t.settings.knowledge.searchReasonAmbiguous
        : t.settings.knowledge.searchReasonNoCandidates;

  return (
    <div className="space-y-3 border-t pt-6">
      <div className="text-base font-semibold">
        {t.settings.knowledge.searchTitle}
      </div>
      <div className="flex gap-2">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t.settings.knowledge.searchPlaceholder}
          disabled={searching}
        />
        <Button
          onClick={() => void handleSearch()}
          disabled={searching || !query.trim()}
        >
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

      {results?.length === 0 && retrieval?.abstained && (
        <Alert>
          <AlertDescription>
            {t.settings.knowledge.searchAbstained} {abstentionReason}
            {retrieval.top_score !== null && (
              <>
                {" "}
                {t.settings.knowledge.searchTopScore}:{" "}
                {retrieval.top_score.toFixed(4)}
              </>
            )}
          </AlertDescription>
        </Alert>
      )}

      {results?.length === 0 && !retrieval?.abstained && (
        <div className="text-muted-foreground text-sm">
          {t.settings.knowledge.searchEmpty}
        </div>
      )}

      {(results?.length ?? 0) > 0 && (
        <div className="space-y-3">
          {results?.map((r, i) => (
            <div key={`${r.source}-${i}`} className="rounded-lg border p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-muted-foreground truncate text-xs">
                  {r.source}
                </div>
                <Badge variant="secondary" className="shrink-0">
                  score: {r.score.toFixed(4)}
                </Badge>
              </div>
              <p className="line-clamp-4 text-sm leading-relaxed">
                {r.content}
              </p>
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
