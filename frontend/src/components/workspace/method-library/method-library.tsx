"use client";

import { useQuery } from "@tanstack/react-query";
import {
  ArrowUpRightIcon,
  BookOpenIcon,
  RefreshCwIcon,
  SearchIcon,
} from "lucide-react";
import { useMemo, useState } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useI18n } from "@/core/i18n/hooks";
import { listMethodCards, type MethodCard } from "@/core/knowledge/method-api";
import {
  categoryLabel,
  filterMethodCards,
  methodCategory,
  methodGroup,
  type MethodGroup,
} from "@/core/knowledge/method-library";

const groups: Array<MethodGroup | "all"> = [
  "all",
  "preprocessing",
  "modeling",
  "other",
];

export function MethodLibrary() {
  const { locale } = useI18n();
  const zh = locale.startsWith("zh");
  const text = (cn: string, en: string) => (zh ? cn : en);
  const [group, setGroup] = useState<MethodGroup | "all">("all");
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const {
    data: catalog,
    error,
    isPending,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ["method-library"],
    queryFn: ({ signal }) => listMethodCards(signal),
    retry: false,
    staleTime: 0,
  });
  const cards = useMemo(() => catalog?.cards ?? [], [catalog]);
  const groupNames = {
    all: text("全部方法", "All methods"),
    preprocessing: text("预处理", "Preprocessing"),
    modeling: text("建模", "Modeling"),
    other: text("其他", "Other"),
  };
  const groupCards = useMemo(
    () =>
      cards.filter((card) => group === "all" || methodGroup(card) === group),
    [cards, group],
  );
  const categories = useMemo(
    () => [...new Set(groupCards.map(methodCategory))].sort(),
    [groupCards],
  );
  const visibleCards = useMemo(
    () =>
      filterMethodCards(
        cards,
        { group, category, query },
        catalog?.category_labels,
      ),
    [cards, group, category, query, catalog?.category_labels],
  );
  const selected = cards.find((card) => card.method_id === selectedId);
  return (
    <div
      className="mx-auto w-full max-w-7xl space-y-6 p-4 sm:p-8"
      data-testid="method-library"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-2">
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <BookOpenIcon className="size-6" />
            {text("方法库", "Method library")}
          </h1>
          <p className="text-muted-foreground text-sm">
            {text(
              "按类型和用途浏览方法，点击卡片查看适用条件、局限与参数。",
              "Browse methods by type and purpose. Open a card for applicability, limitations and parameters.",
            )}
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => void refetch()}
          disabled={isFetching}
        >
          <RefreshCwIcon
            className={isFetching ? "size-4 animate-spin" : "size-4"}
          />
          {text("刷新", "Refresh")}
        </Button>
      </div>
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error.message}</AlertDescription>
        </Alert>
      )}
      <Tabs
        value={group}
        onValueChange={(value) => {
          setGroup(value as MethodGroup | "all");
          setCategory("all");
        }}
      >
        <TabsList className="h-auto flex-wrap">
          {groups.map((value) => (
            <TabsTrigger key={value} value={value}>
              {groupNames[value]}
              <span className="ml-1 text-xs opacity-60">
                {value === "all"
                  ? cards.length
                  : cards.filter((card) => methodGroup(card) === value).length}
              </span>
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      <div className="grid min-w-0 gap-6 md:grid-cols-[12rem_minmax(0,1fr)]">
        <aside
          className="space-y-3"
          aria-label={text("方法分类", "Method categories")}
        >
          <h2 className="text-muted-foreground text-xs font-medium">
            {text("用途分类", "Categories")}
          </h2>
          <div className="flex flex-wrap gap-1 md:flex-col" role="group">
            {["all", ...categories].map((value) => (
              <Button
                key={value}
                variant={category === value ? "secondary" : "ghost"}
                className="h-auto justify-between text-left whitespace-normal"
                aria-pressed={category === value}
                onClick={() => setCategory(value)}
              >
                {value === "all"
                  ? text("全部分类", "All categories")
                  : categoryLabel(value, zh, catalog?.category_labels)}
                <span className="text-muted-foreground ml-3 text-xs">
                  {value === "all"
                    ? groupCards.length
                    : groupCards.filter(
                        (card) => methodCategory(card) === value,
                      ).length}
                </span>
              </Button>
            ))}
          </div>
        </aside>
        <section
          className="min-w-0 space-y-4"
          aria-label={text("方法卡片列表", "Method cards")}
        >
          <div className="relative">
            <SearchIcon className="text-muted-foreground pointer-events-none absolute top-3 left-3 size-4" />
            <Input
              className="pl-9"
              aria-label={text("搜索方法", "Search methods")}
              placeholder={text(
                "搜索名称、用途、适用条件…",
                "Search names, purpose or applicability…",
              )}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              maxLength={200}
            />
          </div>
          <p className="text-muted-foreground text-xs" aria-live="polite">
            {isPending
              ? text("正在加载方法…", "Loading methods…")
              : text(
                  `显示 ${visibleCards.length} 个方法`,
                  `${visibleCards.length} methods shown`,
                )}
          </p>
          {!isPending && !error && visibleCards.length === 0 && (
            <div className="text-muted-foreground rounded-xl border border-dashed p-8 text-center text-sm">
              {cards.length
                ? text(
                    "没有匹配的方法，请调整分类或搜索词。",
                    "No matching methods. Adjust the category or search.",
                  )
                : text("方法库暂无卡片。", "No method cards available.")}
            </div>
          )}
          <div className="grid min-w-0 gap-3 lg:grid-cols-2 xl:grid-cols-3">
            {visibleCards.map((card) => (
              <button
                type="button"
                key={card.method_id}
                className="bg-card hover:bg-accent/40 focus-visible:ring-ring flex min-w-0 flex-col gap-3 rounded-xl border p-5 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none"
                onClick={() => setSelectedId(card.method_id)}
                data-testid="method-card"
                aria-label={card.title}
              >
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">
                    {categoryLabel(
                      methodCategory(card),
                      zh,
                      catalog?.category_labels,
                    )}
                  </Badge>
                  {card.review_status !== "published" && (
                    <Badge variant="outline">
                      {card.review_status === "draft"
                        ? text("草稿", "Draft")
                        : text("已停用", "Retired")}
                    </Badge>
                  )}
                </div>
                <h3 className="font-medium break-words">{card.title}</h3>
                <p className="text-muted-foreground line-clamp-3 text-sm">
                  {card.summary_zh ||
                    text("暂无说明", "No description recorded")}
                </p>
                <p className="text-primary mt-auto flex items-center gap-1 pt-2 text-xs">
                  {text(
                    "查看适用条件与参数",
                    "View applicability and parameters",
                  )}
                  <ArrowUpRightIcon className="size-3" />
                </p>
              </button>
            ))}
          </div>
        </section>
      </div>
      <p className="text-muted-foreground text-xs">
        {text(
          "方法卡片仍可在“设置 → 知识库 → 方法知识库”中管理，两个入口使用同一份内容。",
          "Manage these same cards in Settings → Knowledge → Method knowledge.",
        )}
      </p>
      <Dialog
        open={Boolean(selected)}
        onOpenChange={(open) => {
          if (!open) setSelectedId(null);
        }}
      >
        <DialogContent className="max-h-[85dvh] overflow-y-auto sm:max-w-3xl">
          {selected && (
            <>
              <DialogHeader>
                <DialogTitle className="break-words">
                  {selected.title}
                </DialogTitle>
                <DialogDescription className="break-words">
                  {selected.summary_zh.length > 0
                    ? selected.summary_zh
                    : text("方法详情", "Method details")}
                </DialogDescription>
              </DialogHeader>
              <div className="flex flex-wrap gap-2">
                <Badge>{groupNames[methodGroup(selected)]}</Badge>
                <Badge variant="secondary">
                  {categoryLabel(
                    methodCategory(selected),
                    zh,
                    catalog?.category_labels,
                  )}
                </Badge>
                <Badge variant="outline">
                  {selected.review_status === "published"
                    ? text("已发布", "Published")
                    : selected.review_status === "draft"
                      ? text("草稿", "Draft")
                      : text("已停用", "Retired")}
                </Badge>
              </div>
              <DetailSection
                title={text("适用条件", "Applicability")}
                content={
                  selected.applicability_zh.length > 0
                    ? selected.applicability_zh
                    : selected.planning_notes_zh
                }
                empty={text("暂无适用条件说明", "No applicability recorded")}
              />
              <DetailSection
                title={text("局限与注意事项", "Limitations")}
                content={selected.limitations_zh}
                empty={text("暂无局限说明", "No limitations recorded")}
              />
              <DetailSection
                title={text("参数说明", "Parameter guidance")}
                content={selected.parameter_guidance_zh}
                empty={text("暂无参数说明", "No parameter guidance recorded")}
              />
              <ParameterTable card={selected} zh={zh} />
              <div className="text-muted-foreground space-y-2 border-t pt-4 text-xs">
                <p className="break-all">
                  {selected.method_id} · {selected.provider}{" "}
                  {selected.runtime_provider_version}
                </p>
                {selected.source_url && (
                  <a
                    className="text-primary inline-flex items-center gap-1 underline"
                    href={selected.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {text("查看方法来源", "View method source")}
                    <ArrowUpRightIcon className="size-3" />
                  </a>
                )}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

function DetailSection({
  title,
  content,
  empty,
}: {
  title: string;
  content: string;
  empty: string;
}) {
  return (
    <section className="space-y-2">
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="text-sm leading-6 whitespace-pre-wrap">
        {content || <span className="text-muted-foreground">{empty}</span>}
      </p>
    </section>
  );
}

function ParameterTable({ card, zh }: { card: MethodCard; zh: boolean }) {
  const text = (cn: string, en: string) => (zh ? cn : en);
  const parameters =
    card.method_kind === "mcp_reference"
      ? Object.entries(card.mcp_parameters?.properties ?? {})
      : Object.entries(card.runtime_parameters ?? {});
  return (
    <section className="space-y-2">
      <h3 className="text-sm font-semibold">
        {text("参数定义", "Parameter definitions")}
      </h3>
      {parameters.length ? (
        <div className="max-w-full overflow-x-auto rounded-md border">
          <table className="w-full text-left text-xs">
            <thead className="bg-muted">
              <tr>
                <th className="p-3">{text("名称／类型", "Name / type")}</th>
                <th className="p-3">{text("默认值", "Default")}</th>
                <th className="p-3">
                  {text("说明与约束", "Description and constraints")}
                </th>
              </tr>
            </thead>
            <tbody>
              {parameters.map(([name, spec]) => {
                const field = spec as Record<string, unknown>;
                const constraints = [
                  typeof field.minimum === "number" ? `≥ ${field.minimum}` : "",
                  typeof field.maximum === "number" ? `≤ ${field.maximum}` : "",
                  field.odd ? text("奇数", "odd") : "",
                  Array.isArray(field.choices) ? field.choices.join(", ") : "",
                  Array.isArray(field.enum) ? field.enum.join(", ") : "",
                ]
                  .filter(Boolean)
                  .join(" · ");
                const defaultValue =
                  typeof field.default_repr === "string"
                    ? field.default_repr
                    : Object.hasOwn(field, "default")
                      ? JSON.stringify(field.default)
                      : "—";
                return (
                  <tr key={name} className="border-t align-top">
                    <td className="p-3">
                      <code>{name}</code>
                      <p className="text-muted-foreground mt-1">
                        {typeof field.type === "string" ? field.type : "—"}
                      </p>
                    </td>
                    <td className="max-w-32 p-3 break-words">{defaultValue}</td>
                    <td className="p-3">
                      <p>
                        {typeof field.description_zh === "string"
                          ? field.description_zh
                          : typeof field.description === "string"
                            ? field.description
                            : "—"}
                      </p>
                      {constraints && (
                        <p className="text-muted-foreground mt-1">
                          {constraints}
                        </p>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-muted-foreground text-sm">
          {text("暂无参数定义", "No parameters recorded")}
        </p>
      )}
    </section>
  );
}
