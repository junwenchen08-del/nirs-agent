"use client";

import {
  Loader2Icon,
  PencilIcon,
  PlusIcon,
  RefreshCwIcon,
  SearchIcon,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useI18n } from "@/core/i18n/hooks";
import {
  type CandidateParams,
  type MethodCard,
  type MethodCardChanges,
  type MethodCatalog,
  type MethodParameter,
  type MethodSearchResult,
  listMethodCards,
  resetMethodCard,
  saveMethodCard,
  searchMethodCards,
  setCandidateValue,
} from "@/core/knowledge/method-api";

function McpParameters({ card, zh }: { card: MethodCard; zh: boolean }) {
  const text = (cn: string, en: string) => (zh ? cn : en);
  const schema = card.mcp_parameters;
  return (
    <div
      className="min-w-0 space-y-3 rounded-md border p-3"
      data-testid="mcp-parameter-reference"
    >
      <strong className="text-sm">
        {text(
          "当前 MCP 参数定义（只读）",
          "Installed MCP parameters (read-only)",
        )}
      </strong>
      <p className="text-muted-foreground text-xs break-all">
        {card.mcp_capability_id} ·{" "}
        {card.mcp_provider_version ?? card.runtime_provider_version}
      </p>
      <p className="text-xs">
        {card.mcp_available
          ? text(
              "目录已暴露此能力。执行前仍需 MCP 校验必需输入、操作协议及训练边界；普通自动预处理候选保持受控。",
              "The catalog exposes this capability. Execution still requires MCP validation of required inputs, operations and training boundaries; automatic preprocessing remains governed.",
            )
          : text(
              "当前 MCP 未暴露此文档项，仅用于知识检索。",
              "This documented item is not exposed by the installed MCP; reference only.",
            )}
      </p>
      {!!card.mcp_operations?.length && (
        <p className="text-xs break-words">
          {text("目录操作：", "Catalog operations: ")}
          {card.mcp_operations.join(", ")}
        </p>
      )}
      {schema && Object.keys(schema.properties ?? {}).length > 0 && (
        <div className="max-w-full overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b">
                <th className="p-2">
                  {text("参数／类型", "Parameter / type")}
                </th>
                <th className="p-2">{text("默认值", "Default")}</th>
                <th className="p-2">
                  {text("定义与约束", "Definition and constraints")}
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(schema.properties ?? {}).map(([name, field]) => (
                <tr className="border-b align-top" key={name}>
                  <td className="p-2">
                    <code>{name}</code>
                    {schema.required?.includes(name) && (
                      <span className="ml-1">{text("必需", "required")}</span>
                    )}
                    <p className="text-muted-foreground">
                      {field.type ??
                        field["x-python-type"] ??
                        text("复合／引用", "compound / reference")}
                    </p>
                  </td>
                  <td className="max-w-36 p-2 break-all">
                    {field.default_repr ??
                      (Object.hasOwn(field, "default")
                        ? JSON.stringify(field.default)
                        : text("无默认值", "no default"))}
                  </td>
                  <td className="p-2 break-words">
                    {field.description ??
                      text(
                        "见完整 Schema 和 MCP 操作协议",
                        "See full schema and MCP operation contract",
                      )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {schema && (
        <details className="text-xs">
          <summary className="cursor-pointer">
            {text("完整 MCP Schema", "Full MCP schema")}
          </summary>
          <pre className="mt-2 max-h-80 overflow-auto break-all whitespace-pre-wrap">
            {JSON.stringify(schema, null, 2)}
          </pre>
        </details>
      )}
      {card.source_documented_signature && (
        <details className="text-xs">
          <summary className="cursor-pointer">
            {text(
              "官网参数（与安装版本分别展示）",
              "Website parameters (separate from installed version)",
            )}
          </summary>
          <p className="my-2 break-all">{card.source_documented_signature}</p>
          {Object.entries(card.source_documented_parameters ?? {}).map(
            ([name, description]) => (
              <p className="my-1 break-words" key={name}>
                <strong>{name}: </strong>
                {description}
              </p>
            ),
          )}
          <p className="text-muted-foreground mt-2">
            {card.documentation_runtime_difference?.trim()
              ? card.documentation_runtime_difference
              : text(
                  "官网为动态文档，执行以安装版本 MCP 描述为准；默认值与示例不是普适最优范围。",
                  "Website documentation changes. Execute against installed MCP definitions; defaults and examples are not universal optima.",
                )}
          </p>
        </details>
      )}
      {card.official_documentation === false && (
        <p className="text-muted-foreground text-xs">
          {text(
            "来源为当前安装包的 MCP 目录，不属于本次官网导航条目。",
            "Source: installed package MCP catalog, outside the reviewed website navigation.",
          )}
        </p>
      )}
    </div>
  );
}

export function MethodKnowledgeSettingsPage() {
  const { locale } = useI18n();
  const zh = locale.startsWith("zh");
  const text = (cn: string, en: string) => (zh ? cn : en);
  const [catalog, setCatalog] = useState<MethodCatalog | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [edit, setEdit] = useState<MethodCard | "new" | null>(null);
  const [reset, setReset] = useState<MethodCard | null>(null);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState<MethodSearchResult | null>(null);
  const [kind, setKind] = useState("all");
  const [category, setCategory] = useState("all");
  const [filter, setFilter] = useState("");
  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setCatalog(await listMethodCards());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);
  async function changeStatus(
    card: MethodCard,
    status: "published" | "retired",
  ) {
    if (!catalog) return;
    setBusy(true);
    try {
      setCatalog(
        await saveMethodCard(
          card.method_id,
          { review_status: status },
          catalog.revision,
        ),
      );
      setSearch(null);
      toast.success(
        text(
          "状态已更新，新建模将使用当前已启用方法",
          "Status updated for new modeling runs",
        ),
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  async function runSearch() {
    setBusy(true);
    setError("");
    try {
      setSearch(await searchMethodCards(query));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="flex flex-col gap-5" data-testid="method-knowledge-manager">
      <p className="text-muted-foreground text-sm">
        {text(
          "维护预处理、建模和 MCP 方法参考。按类别查看用途、适用条件和参数；保存后生效，无需启动论文 RAG 服务。",
          "Maintain preprocessing, model and MCP references. Browse categories, applicability and parameters; edits persist without a paper RAG server.",
        )}
      </p>
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm">
          {catalog
            ? text(
                `${catalog.count} 个方法 · 已启用 ${catalog.cards.filter((card) => card.review_status === "published").length} 个`,
                `${catalog.count} methods · ${catalog.cards.filter((card) => card.review_status === "published").length} published`,
              )
            : text("正在加载…", "Loading…")}
        </span>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => void load()}
            disabled={loading || busy}
          >
            <RefreshCwIcon className="size-4" />
            {text("刷新方法", "Refresh methods")}
          </Button>
          <Button
            size="sm"
            onClick={() => setEdit("new")}
            disabled={
              !catalog?.can_edit ||
              busy ||
              !catalog.available_methods.some(
                (item) =>
                  !catalog.cards.some(
                    (card) => card.method_id === item.method_id,
                  ),
              )
            }
            title={text(
              "只能为尚未收录的已接入算法新增卡片",
              "Add references for installed algorithms not yet in the catalog",
            )}
          >
            <PlusIcon className="size-4" />
            {text("新增方法卡", "Add method card")}
          </Button>
        </div>
      </div>
      {error && (
        <Alert variant="destructive">
          <AlertDescription>
            {error}
            <Button variant="link" onClick={() => void load()}>
              {text("重新加载", "Reload")}
            </Button>
          </AlertDescription>
        </Alert>
      )}
      {loading && <Loader2Icon className="size-5 animate-spin" />}
      {catalog && (
        <>
          <label className="flex items-center gap-3 text-sm">
            {text("方法类型", "Method type")}
            <select
              aria-label={text("方法类型", "Method type")}
              className="bg-background rounded-md border p-2"
              value={kind}
              onChange={(e) => setKind(e.target.value)}
            >
              <option value="all">
                {text(`全部（${catalog.count}）`, `All (${catalog.count})`)}
              </option>
              <option value="preprocessing">
                {text("预处理", "Preprocessing")} (
                {
                  catalog.cards.filter((c) => c.method_kind === "preprocessing")
                    .length
                }
                )
              </option>
              <option value="modeling">
                {text("回归建模", "Regression models")} (
                {
                  catalog.cards.filter((c) => c.method_kind === "modeling")
                    .length
                }
                )
              </option>
              <option value="mcp_reference">
                {text("MCP 显式调用参考", "MCP explicit references")} (
                {
                  catalog.cards.filter((c) => c.method_kind === "mcp_reference")
                    .length
                }
                )
              </option>
            </select>
          </label>
          {catalog.coverage && (
            <p
              className="text-muted-foreground text-sm"
              data-testid="method-coverage"
            >
              {text(
                `官网文档 ${catalog.coverage.official_covered_count}/${catalog.coverage.official_document_count} · MCP 能力 ${catalog.coverage.mcp_covered_count}/${catalog.coverage.mcp_capability_count} · 安装版本 ${catalog.coverage.runtime_provider_version}`,
                `Official documents ${catalog.coverage.official_covered_count}/${catalog.coverage.official_document_count} · MCP capabilities ${catalog.coverage.mcp_covered_count}/${catalog.coverage.mcp_capability_count} · Installed ${catalog.coverage.runtime_provider_version}`,
              )}
            </p>
          )}
          <div className="flex items-center gap-3">
            <select
              aria-label={text("方法分类", "Method category")}
              className="bg-background rounded-md border p-2 text-sm"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              <option value="all">{text("所有分类", "All categories")}</option>
              {Object.entries(catalog.category_labels ?? {}).map(
                ([key, label]) => (
                  <option key={key} value={key}>
                    {zh ? label : key.replaceAll("_", " ")}
                  </option>
                ),
              )}
            </select>
            <Input
              aria-label={text("筛选方法列表", "Filter method list")}
              placeholder={text(
                "按名称、用途或 MCP ID 筛选",
                "Filter by name, purpose or MCP ID",
              )}
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              maxLength={200}
            />
          </div>
          <div className="space-y-3">
            {catalog.cards
              .filter(
                (card) =>
                  (kind === "all" ||
                    (card.method_kind ?? "preprocessing") === kind) &&
                  (category === "all" || card.category === category) &&
                  [
                    card.title,
                    card.method_id,
                    card.mcp_capability_id,
                    card.summary_zh,
                    card.keywords,
                  ]
                    .join(" ")
                    .toLowerCase()
                    .includes(filter.trim().toLowerCase()),
              )
              .map((card) => (
                <article
                  key={card.method_id}
                  className="space-y-3 rounded-lg border p-4"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <strong className="break-words">{card.title}</strong>
                        <Badge variant="outline">
                          {card.method_kind === "modeling"
                            ? text("回归建模", "Regression model")
                            : card.method_kind === "mcp_reference"
                              ? text("MCP 方法参考", "MCP reference")
                              : text("预处理", "Preprocessing")}
                        </Badge>
                        {card.category && (
                          <Badge variant="outline">
                            {zh
                              ? (catalog.category_labels?.[card.category] ??
                                card.category)
                              : card.category.replaceAll("_", " ")}
                          </Badge>
                        )}
                        <Badge
                          variant={
                            card.review_status === "published"
                              ? "default"
                              : "secondary"
                          }
                        >
                          {card.review_status === "published"
                            ? text("已启用", "Published")
                            : card.review_status === "retired"
                              ? text("已停用", "Retired")
                              : text("草稿", "Draft")}
                        </Badge>
                        {!card.auto_eligible && (
                          <Badge variant="outline">
                            {card.method_kind === "modeling"
                              ? text("规划参考", "Planning reference")
                              : card.method_kind === "mcp_reference"
                                ? card.mcp_available
                                  ? card.mcp_execution_supported
                                    ? text(
                                        "MCP 已暴露／显式调用",
                                        "MCP exposed / explicit call",
                                      )
                                    : text(
                                        "MCP 抽象接口／不可执行",
                                        "MCP abstract / not executable",
                                      )
                                  : text(
                                      "仅文档／MCP 未暴露",
                                      "Documentation only / MCP unavailable",
                                    )
                                : text(
                                    "仅供参考／显式选择",
                                    "Reference / explicit only",
                                  )}
                          </Badge>
                        )}
                      </div>
                      <p className="text-muted-foreground mt-1 text-xs">
                        {card.method_id} · {card.provider ?? "chemotools"}{" "}
                        {card.runtime_provider_version}
                      </p>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setEdit(card)}
                      disabled={busy || !catalog.can_edit}
                    >
                      <PencilIcon className="size-4" />
                      {text("编辑", "Edit")}
                    </Button>
                  </div>
                  <p className="text-sm">
                    {card.summary_zh ||
                      text("尚未填写方法说明", "No description yet")}
                  </p>
                  <details className="space-y-2 text-sm">
                    <summary className="text-primary cursor-pointer">
                      {text(
                        "适用条件、局限与参数说明",
                        "Applicability, limitations and parameters",
                      )}
                    </summary>
                    <p className="mt-2">
                      <strong>{text("适用条件：", "Applicability: ")}</strong>
                      {card.applicability_zh || card.planning_notes_zh}
                    </p>
                    <p>
                      <strong>{text("局限：", "Limitations: ")}</strong>
                      {card.limitations_zh}
                    </p>
                    <p>
                      <strong>{text("参数说明：", "Parameters: ")}</strong>
                      {card.parameter_guidance_zh}
                    </p>
                    <p className="text-muted-foreground text-xs">
                      {text(
                        "适用建议为项目解释，需通过训练内验证；不代表官网限定的光谱类型或最优参数。",
                        "Applicability is project guidance requiring training validation, not an official spectrum restriction or optimum.",
                      )}
                    </p>
                    {card.method_kind === "modeling" && (
                      <p className="text-muted-foreground text-xs">
                        {text(
                          "模型卡片用于规划检索；参考参数编辑不会修改当前自动模型搜索网格。",
                          "Model cards guide planning; reference parameter edits do not change the current automatic search grid.",
                        )}
                      </p>
                    )}
                    {card.mcp_capability_id && (
                      <McpParameters card={card} zh={zh} />
                    )}
                  </details>
                  <div className="flex flex-wrap gap-2">
                    {card.problem_tags.map((tag) => (
                      <Badge variant="outline" key={tag}>
                        {zh
                          ? (catalog.problem_tags[tag] ?? tag)
                          : tag.replaceAll("_", " ")}
                      </Badge>
                    ))}
                  </div>
                  <div className="flex flex-wrap items-center gap-3 text-xs">
                    <a
                      href={card.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline"
                    >
                      {text("查看来源", "View source")}
                    </a>
                    <span>
                      {card.method_kind === "mcp_reference"
                        ? text(
                            "参数由 MCP describe / validate 校验",
                            "Parameters validated by MCP describe / validate",
                          )
                        : text(
                            `${card.candidate_params.length} 组${card.method_kind === "modeling" ? "参考" : "候选"}参数`,
                            `${card.candidate_params.length} parameter candidates`,
                          )}
                    </span>
                    <span className="text-muted-foreground">
                      {text("整理日期", "Reviewed")}: {card.retrieved_on}
                    </span>
                    <div className="ml-auto flex gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={busy || !catalog.can_edit}
                        onClick={() =>
                          void changeStatus(
                            card,
                            card.review_status === "published"
                              ? "retired"
                              : "published",
                          )
                        }
                      >
                        {card.review_status === "published"
                          ? text("停用", "Retire")
                          : text("启用", "Publish")}
                      </Button>
                      {card.has_default && (
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={busy || !catalog.can_edit}
                          onClick={() => setReset(card)}
                        >
                          {text("恢复内置版本", "Restore default")}
                        </Button>
                      )}
                    </div>
                  </div>
                </article>
              ))}
          </div>
          <section className="space-y-3 rounded-lg border p-4">
            <h3 className="font-medium">
              {text("方法检索测试", "Test method search")}
            </h3>
            <p className="text-muted-foreground text-xs">
              {text(
                "只检索已启用的方法。这里预览检索结果，不训练模型；正式建模仍根据 Cal 诊断和交叉验证决定采用。",
                "Search published references without training. Actual selection still uses Cal diagnostics and cross-validation.",
              )}
            </p>
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                void runSearch();
              }}
            >
              <Input
                aria-label={text("方法检索关键词", "Method search query")}
                placeholder={text(
                  "例如：高频噪声、基线漂移、散射",
                  "e.g. noise, baseline drift, scatter",
                )}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                maxLength={2000}
              />
              <Button type="submit" disabled={busy || !query.trim()}>
                <SearchIcon className="size-4" />
                {text("检索", "Search")}
              </Button>
            </form>
            {search && (
              <div className="space-y-2" data-testid="method-search-results">
                {search.abstained ? (
                  <p className="text-sm">
                    {text(
                      "没有匹配的已启用方法",
                      "No matching published reference",
                    )}
                  </p>
                ) : (
                  search.results.map((card) => (
                    <div
                      key={card.method_id}
                      className="bg-muted rounded-md p-3 text-sm"
                    >
                      <strong>{card.title}</strong>
                      <p className="mt-1">{card.summary_zh}</p>
                      <p className="text-muted-foreground mt-1 text-xs">
                        {card.method_id} · {card.evidence_id}
                      </p>
                    </div>
                  ))
                )}
              </div>
            )}
          </section>
          <details className="rounded-lg border p-4">
            <summary className="cursor-pointer text-sm font-medium">
              {text("修改记录", "Change history")} ·{" "}
              {text(`版本 ${catalog.revision}`, `revision ${catalog.revision}`)}
            </summary>
            <p className="text-muted-foreground my-2 text-xs break-all">
              {catalog.index_version}
            </p>
            {catalog.history.length ? (
              <ul className="space-y-2 text-xs">
                {catalog.history.map((item) => (
                  <li key={item.revision}>
                    #{item.revision} · {item.method_id} · {item.action} ·{" "}
                    {new Date(item.changed_at).toLocaleString()} · {item.actor}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground text-sm">
                {text(
                  "当前使用内置版本，尚无网页修改",
                  "Using packaged references; no edits yet",
                )}
              </p>
            )}
          </details>
          <p className="text-muted-foreground text-xs">
            {text(
              "管理员可维护共享方法卡。新增方法卡只能关联已接入的算法；启用参考资料不代表该算法会被自动采用。历史模型和报告不变。",
              "Admins maintain shared method references. New cards can only bind installed algorithms. Publication does not guarantee automatic adoption. Historical models and reports remain unchanged.",
            )}
          </p>
        </>
      )}
      {catalog && edit && (
        <MethodEditor
          key={edit === "new" ? "new" : edit.method_id}
          catalog={catalog}
          card={edit === "new" ? null : edit}
          onClose={() => setEdit(null)}
          onSave={(updated) => {
            setCatalog(updated);
            setEdit(null);
            setSearch(null);
          }}
        />
      )}
      <Dialog
        open={reset !== null}
        onOpenChange={(open) => {
          if (!open && !busy) setReset(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {text("恢复内置版本", "Restore packaged reference")}
            </DialogTitle>
            <DialogDescription>
              {text(
                `将 ${reset?.title ?? ""} 恢复为内置内容和启用状态。修改记录保留，历史模型不变。`,
                `Restore ${reset?.title ?? ""} to its packaged content and status. History and existing models are preserved.`,
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => setReset(null)}
            >
              {text("取消", "Cancel")}
            </Button>
            <Button
              disabled={busy}
              onClick={async () => {
                if (!reset || !catalog) return;
                setBusy(true);
                try {
                  setCatalog(
                    await resetMethodCard(reset.method_id, catalog.revision),
                  );
                  setReset(null);
                  setSearch(null);
                  toast.success(text("已恢复内置版本", "Default restored"));
                } catch (e) {
                  toast.error(e instanceof Error ? e.message : String(e));
                } finally {
                  setBusy(false);
                }
              }}
            >
              {text("确认恢复", "Restore")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function MethodEditor({
  catalog,
  card,
  onClose,
  onSave,
}: {
  catalog: MethodCatalog;
  card: MethodCard | null;
  onClose: () => void;
  onSave: (catalog: MethodCatalog) => void;
}) {
  const { locale } = useI18n();
  const zh = locale.startsWith("zh");
  const text = (cn: string, en: string) => (zh ? cn : en);
  const available = catalog.available_methods.filter(
    (item) =>
      !catalog.cards.some((existing) => existing.method_id === item.method_id),
  );
  const [methodId, setMethodId] = useState(
    card?.method_id ?? available[0]?.method_id ?? "",
  );
  const spec =
    card ??
    catalog.available_methods.find((item) => item.method_id === methodId);
  const [title, setTitle] = useState(card?.title ?? available[0]?.title ?? "");
  const [summary, setSummary] = useState(card?.summary_zh ?? "");
  const [source, setSource] = useState(
    card?.source_url ?? "https://chemotools.org/methods/",
  );
  const [section, setSection] = useState(card?.source_section ?? "");
  const [keywords, setKeywords] = useState(card?.keywords ?? "");
  const [tags, setTags] = useState<string[]>(card?.problem_tags ?? []);
  const [avoid, setAvoid] = useState<string[]>(card?.avoid_tags ?? []);
  const [notes, setNotes] = useState(card?.planning_notes_zh ?? "");
  const [applicability, setApplicability] = useState(
    card?.applicability_zh ?? "",
  );
  const [limitations, setLimitations] = useState(card?.limitations_zh ?? "");
  const [guidance, setGuidance] = useState(card?.parameter_guidance_zh ?? "");
  const [priority, setPriority] = useState(card?.priority ?? 99);
  const [params, setParams] = useState<CandidateParams[]>(
    card?.candidate_params ?? [{}],
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  function toggle(
    value: string,
    list: string[],
    set: (value: string[]) => void,
  ) {
    set(
      list.includes(value)
        ? list.filter((item) => item !== value)
        : [...list, value],
    );
  }
  function updateParam(
    index: number,
    name: string,
    raw: string,
    schema: MethodParameter,
  ) {
    try {
      const updated = setCandidateValue(params[index] ?? {}, name, raw, schema);
      setParams((current) =>
        current.map((item, i) => (i === index ? updated : item)),
      );
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }
  async function save(status: "draft" | "published") {
    setSaving(true);
    setError("");
    const changes: MethodCardChanges = {
      title,
      summary_zh: summary,
      source_url: source,
      source_section: section,
      keywords,
      problem_tags: tags,
      avoid_tags: avoid,
      planning_notes_zh: notes,
      applicability_zh: applicability,
      limitations_zh: limitations,
      parameter_guidance_zh: guidance,
      candidate_params: params,
      priority,
      review_status: status,
    };
    try {
      onSave(await saveMethodCard(methodId, changes, catalog.revision));
      toast.success(
        text(
          "已保存，新的建模将读取当前启用内容",
          "Saved for new modeling runs",
        ),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !saving) onClose();
      }}
    >
      <DialogContent className="flex max-h-[90vh] flex-col overflow-hidden sm:max-w-3xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>
            {card
              ? text("编辑方法卡", "Edit method card")
              : text("新增方法卡", "Add method card")}
          </DialogTitle>
          <DialogDescription>
            {text(
              "维护已有算法的参考与搜索参数，运行时约束和算法实现不可在这里修改。",
              "Edit references and candidate settings for installed algorithms; runtime constraints remain enforced.",
            )}
          </DialogDescription>
        </DialogHeader>
        <div className="min-h-0 space-y-4 overflow-y-auto pr-2">
          {!card && (
            <label className="block text-sm">
              {text("关联算法", "Installed algorithm")}
              <select
                aria-label={text("关联算法", "Installed algorithm")}
                className="bg-background mt-1 w-full rounded-md border p-2"
                value={methodId}
                onChange={(e) => {
                  setMethodId(e.target.value);
                  setTitle(
                    available.find((item) => item.method_id === e.target.value)
                      ?.title ?? "",
                  );
                  setParams([{}]);
                }}
              >
                {available.map((item) => (
                  <option key={item.method_id} value={item.method_id}>
                    {item.method_id} · {item.title}
                  </option>
                ))}
              </select>
              {!available.length && (
                <p className="text-destructive">
                  {text(
                    "所有可关联算法已有方法卡",
                    "All available algorithms already have cards",
                  )}
                </p>
              )}
            </label>
          )}
          <p className="text-muted-foreground text-xs break-all">
            {methodId} · {spec?.provider ?? "chemotools"}{" "}
            {spec?.runtime_provider_version}
          </p>
          {spec?.method_kind === "modeling" && (
            <Alert>
              <AlertDescription>
                {text(
                  "模型卡片用于规划检索。这里的参数组合仅供参考，不会修改现有自动模型搜索网格。",
                  "Model cards guide planning. These reference combinations do not change the current automatic search grid.",
                )}
              </AlertDescription>
            </Alert>
          )}
          {card?.method_kind === "mcp_reference" && (
            <Alert>
              <AlertDescription>
                {text(
                  "该卡片用于规划检索与 MCP 显式调用。MCP 参数定义只读；数组、模型等输入由调用时提供，不作为自动搜索组合保存。",
                  "This card guides planning and explicit MCP calls. MCP parameters are read-only; array and model inputs are supplied at execution, not saved as automatic candidates.",
                )}
              </AlertDescription>
            </Alert>
          )}
          <label className="block text-sm">
            {text("适用条件", "Applicability")}
            <Textarea
              aria-label={text("适用条件", "Applicability")}
              value={applicability}
              onChange={(e) => setApplicability(e.target.value)}
              maxLength={2000}
            />
          </label>
          <label className="block text-sm">
            {text("方法局限", "Limitations")}
            <Textarea
              aria-label={text("方法局限", "Limitations")}
              value={limitations}
              onChange={(e) => setLimitations(e.target.value)}
              maxLength={2000}
            />
          </label>
          <label className="block text-sm">
            {text("参数说明", "Parameter guidance")}
            <Textarea
              aria-label={text("参数说明", "Parameter guidance")}
              value={guidance}
              onChange={(e) => setGuidance(e.target.value)}
              maxLength={2000}
            />
          </label>
          <label className="block text-sm">
            {text("方法名称", "Method title")}
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              maxLength={150}
            />
          </label>
          <label className="block text-sm">
            {text("用途与说明", "Purpose and description")}
            <Textarea
              value={summary}
              onChange={(e) => setSummary(e.target.value)}
              maxLength={2000}
            />
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-sm">
              {text("来源链接", "Source URL")}
              <Input
                value={source}
                onChange={(e) => setSource(e.target.value)}
                maxLength={500}
              />
            </label>
            <label className="text-sm">
              {text("来源章节", "Source section")}
              <Input
                value={section}
                onChange={(e) => setSection(e.target.value)}
                maxLength={250}
              />
            </label>
          </div>
          <label className="block text-sm">
            {text("检索关键词（中英文）", "Search keywords")}
            <Input
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              maxLength={1000}
            />
          </label>
          {(
            [
              [text("适用问题", "Relevant problems"), tags, setTags],
              [text("避免自动采用的情况", "Avoid when"), avoid, setAvoid],
            ] as const
          ).map(([label, values, setter]) => (
            <fieldset key={label} className="space-y-2">
              <legend className="text-sm">{label}</legend>
              <div className="flex flex-wrap gap-3">
                {Object.entries(catalog.problem_tags).map(([key, name]) => (
                  <label className="flex items-center gap-1 text-xs" key={key}>
                    <input
                      type="checkbox"
                      checked={values.includes(key)}
                      onChange={() => toggle(key, values, setter)}
                    />
                    {zh ? name : key.replaceAll("_", " ")}
                  </label>
                ))}
              </div>
            </fieldset>
          ))}
          {card?.method_kind === "mcp_reference" ? (
            <McpParameters card={card} zh={zh} />
          ) : (
            <fieldset className="space-y-3">
              <legend className="text-sm">
                {spec?.method_kind === "modeling"
                  ? text("参考参数组合", "Reference parameter combinations")
                  : text("候选参数组合", "Candidate parameter combinations")}
              </legend>
              <p className="text-muted-foreground text-xs">
                {spec?.method_kind === "modeling"
                  ? text(
                      "最多保存 8 组参考参数，留空表示默认值；字段上下界是项目参考约束。这些组合用于知识说明，不修改自动模型搜索网格。",
                      "Store up to 8 reference combinations; blank fields mean defaults. Bounds are project reference constraints. These combinations describe the method and do not change automatic model search grids.",
                    )
                  : text(
                      "留空使用运行时默认值；最多 8 组，正式建模仍受整体候选预算限制。这里是项目搜索值，不是官网推荐范围。",
                      "Empty fields use runtime defaults. Up to 8 combinations within the training budget; these are project settings, not official recommended ranges.",
                    )}
              </p>
              {params.map((candidate, index) => (
                <div key={index} className="rounded-md border p-3">
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span>
                      {text(`组合 ${index + 1}`, `Combination ${index + 1}`)}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={params.length <= 1 || saving}
                      onClick={() =>
                        setParams((current) =>
                          current.filter((_, i) => i !== index),
                        )
                      }
                    >
                      {text("移除", "Remove")}
                    </Button>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(spec?.runtime_parameters ?? {}).map(
                      ([name, schema]) => (
                        <label key={name} className="text-xs">
                          {name}
                          <span className="text-muted-foreground ml-1">
                            {schema.minimum !== undefined
                              ? `≥${schema.minimum} `
                              : ""}
                            {schema.maximum !== undefined
                              ? `≤${schema.maximum} `
                              : ""}
                            {schema.odd ? text("奇数", "odd") : ""}
                          </span>
                          {schema.choices || schema.type === "boolean" ? (
                            <select
                              aria-label={`candidate-${index + 1}-${name}`}
                              className="bg-background mt-1 w-full rounded-md border p-2"
                              value={
                                candidate[name] === undefined
                                  ? ""
                                  : String(candidate[name])
                              }
                              onChange={(e) =>
                                updateParam(index, name, e.target.value, schema)
                              }
                            >
                              <option value="">
                                {text("默认", "Default")}:{" "}
                                {String(schema.default ?? "")}
                              </option>
                              {(schema.choices ?? [true, false]).map(
                                (value) => (
                                  <option
                                    key={String(value)}
                                    value={String(value)}
                                  >
                                    {String(value)}
                                  </option>
                                ),
                              )}
                            </select>
                          ) : (
                            <Input
                              aria-label={`candidate-${index + 1}-${name}`}
                              className="mt-1"
                              type={
                                schema.type === "string" ? "text" : "number"
                              }
                              min={schema.minimum}
                              max={schema.maximum}
                              step={schema.type === "integer" ? 1 : "any"}
                              value={
                                candidate[name] === undefined
                                  ? ""
                                  : String(candidate[name])
                              }
                              placeholder={
                                text("默认", "Default") +
                                ": " +
                                String(schema.default ?? "")
                              }
                              onChange={(e) =>
                                updateParam(index, name, e.target.value, schema)
                              }
                            />
                          )}
                        </label>
                      ),
                    )}
                  </div>
                  {!Object.keys(spec?.runtime_parameters ?? {}).length && (
                    <p className="text-muted-foreground text-xs">
                      {text(
                        "此方法没有可调整的运行时参数",
                        "This method has no editable runtime parameters",
                      )}
                    </p>
                  )}
                </div>
              ))}
              <Button
                variant="outline"
                size="sm"
                disabled={params.length >= 8 || saving}
                onClick={() => setParams((current) => [...current, {}])}
              >
                {text("添加参数组合", "Add combination")}
              </Button>
            </fieldset>
          )}
          <label className="block text-sm">
            {text("项目使用边界与注意事项", "Project boundaries and notes")}
            <Textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              maxLength={2000}
            />
          </label>
          <label className="block text-sm">
            {text(
              "候选优先级（0–99，较小者优先）",
              "Candidate priority (0–99, lower first)",
            )}
            <Input
              type="number"
              min={0}
              max={99}
              step={1}
              value={priority}
              onChange={(e) => setPriority(Number(e.target.value))}
            />
          </label>
          <p className="text-muted-foreground text-xs">
            {text(
              "保存草稿会暂停此卡片的检索；保存并启用表示已核对说明、来源和参数。历史建模证据不变。",
              "Saving a draft excludes this card from retrieval. Save and publish after checking the reference and settings. Existing evidence is unchanged.",
            )}
          </p>
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>
        <DialogFooter className="shrink-0 border-t pt-3">
          <Button variant="outline" onClick={onClose} disabled={saving}>
            {text("取消", "Cancel")}
          </Button>
          <Button
            variant="secondary"
            onClick={() => void save("draft")}
            disabled={saving || !methodId || !title.trim()}
          >
            {text("保存草稿", "Save draft")}
          </Button>
          <Button
            onClick={() => void save("published")}
            disabled={saving || !methodId || !title.trim()}
          >
            {saving && <Loader2Icon className="size-4 animate-spin" />}
            {text("保存并启用", "Save and publish")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
