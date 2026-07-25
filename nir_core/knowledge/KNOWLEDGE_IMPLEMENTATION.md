# NIR 知识库系统 — 完整实现方案 v2.0

> 本文档覆盖知识库的完整架构设计、后端 API、前端界面设计、部署架构和未来扩展规划。

---

## 1. 系统概述

### 1.1 目标

为 NIR-Agent 构建本地知识库 RAG 系统，实现：

- 论文 PDF/DOCX/Markdown 自动解析 → 智能分块 → 实体提取 → 向量化 → 持久化存储
- 智能体通过 `nir_search_knowledge` 工具语义检索相关论文段落
- 用户通过前端设置页面对知识库进行增删查操作
- 支持 100+ 篇论文，检索延迟 <100ms
- 未来可扩展图谱检索（Neo4j），不修改现有代码

### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **接口抽象** | Agent 工具层只依赖 `KnowledgeRetriever` Protocol，不感知具体实现 |
| **Docker-Host 桥接** | 搜索服务器运行在宿主机（Anaconda Python），Docker 容器通过 HTTP 调用 |
| **中文友好** | PDF 中文文本规范化（NFKC + CJK 空格消除）、中英文实体别名映射 |
| **零外部服务** | ChromaDB 本地持久化 + SentenceTransformer 本地嵌入，无需 API Key |
| **前端一致性** | 复用 DeerFlow 现有 shadcn/ui 组件 + i18n + SettingsSection 框架 |

---

## 2. 整体架构

### 2.1 四层架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      前端层 (Next.js + shadcn/ui)                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Settings → Knowledge Settings Page                         │    │
│  │  · 文档列表表格 (title, year, chunks, actions)              │    │
│  │  · 上传对话框 (文件选择 + 元数据)                            │    │
│  │  · 删除确认对话框                                            │    │
│  │  · 搜索测试区 (query → 结果 + score + 实体标签)             │    │
│  └───────────────────────┬─────────────────────────────────────┘    │
│                          │ /api/knowledge/*                        │
└──────────────────────────┼─────────────────────────────────────────┘
                           │
┌──────────────────────────┼─────────────────────────────────────────┐
│                   后端层 (FastAPI Gateway)                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  knowledge.py router                                        │    │
│  │  · GET    /documents      → 代理到搜索服务器                 │    │
│  │  · POST   /documents      → base64 编码 + 转发              │    │
│  │  · DELETE /documents/{id} → 代理删除                        │    │
│  │  · GET    /stats          → 代理统计                        │    │
│  │  · POST   /search         → 代理搜索                        │    │
│  └───────────────────────┬─────────────────────────────────────┘    │
│                          │ HTTP → host.docker.internal:8089        │
└──────────────────────────┼─────────────────────────────────────────┘
                           │
┌──────────────────────────┼─────────────────────────────────────────┐
│                搜索服务器层 (宿主机 Anaconda Python)                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  _search_server.py (http.server, port 8089)                 │    │
│  │  · GET  /health       → 健康检查                            │    │
│  │  · GET  /documents    → 列出所有文档                        │    │
│  │  · POST /documents    → 解析→分块→实体提取→入库             │    │
│  │  · DELETE /documents/{id} → 删除文档                        │    │
│  │  · GET  /stats        → 知识库统计                          │    │
│  │  · POST /search       → 语义检索                            │    │
│  └───────────────────────┬─────────────────────────────────────┘    │
│                          │                                        │
└──────────────────────────┼─────────────────────────────────────────┘
                           │
┌──────────────────────────┼─────────────────────────────────────────┐
│                   核心库层 (nir_core/knowledge/)                    │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐  ┌────────────┐   │
│  │ parser   │  │ chunker  │  │entity_extractor│  │ vectorstore│   │
│  │PDF→MD    │  │section/  │  │ NIR 领域实体   │  │ ChromaDB   │   │
│  │DOCX→MD   │  │naive/qa  │  │ 中英文别名     │  │ HNSW 索引  │   │
│  └──────────┘  └──────────┘  └────────────────┘  └────────────┘   │
│                          │                                        │
│              ChromaDB (.chromadb-bge-m3/) + BGE-M3                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流：文档入库

```
用户上传 PDF
  ↓ 前端 FormData → /api/knowledge/documents
  ↓ 后端 base64 编码 → POST http://host.docker.internal:8089/documents
  ↓ _search_server.py 接收 {filename, content_b64, title, year}
  ↓ base64 解码 → 写入临时文件
  ↓ parser.parse_document(tmp_path) → Markdown 文本
  ↓   · _normalize_pdf_text(): NFKC 规范化 + CJK 空格消除
  ↓ chunker.chunk_document(markdown, strategy="section") → list[Chunk]
  ↓ entity_extractor.extract_entities(markdown) → {methods, models, datasets, metrics}
  ↓   · 中文别名: 偏最小二乘→pls, 标准正态变量→snv, ...
  ↓   · CJK 术语: 子串匹配（非 \b 词边界）
  ↓ 向量存储: ChromaDBRetriever.add_documents(chunks)
  ↓   · _sanitise_metadata(): None→跳过, list→JSON序列化
  ↓   · SentenceTransformer.encode() → 384维嵌入
  ↓   · ChromaDB collection.add(ids, embeddings, documents, metadatas)
  ↓ 返回 {doc_id, chunks_added, title, methods, models, datasets, metrics}
```

### 2.3 数据流：语义检索

```
智能体调用 nir_search_knowledge(query="SNV vs MSC soil")
  ↓ tools.py: 优先直接 import nir_core.knowledge (快速路径)
  ↓          失败 → HTTP 回退 (Docker 环境)
  ↓          _search_knowledge_via_http(query, top_k)
  ↓          POST http://host.docker.internal:8089/search
  ↓ _search_server.py: ChromaDBRetriever.search(query, top_k)
  ↓          SentenceTransformer.encode(query) → 查询向量
  ↓          ChromaDB HNSW 检索 → top-k 结果
  ↓          cosine distance → similarity score
  ↓ 返回 [{content, source, score, entities, related_entities}]
  ↓ 智能体根据论文段落调整预处理/建模策略
```

---

## 3. 核心库层详细设计

### 3.1 模块职责

| 文件 | 职责 | 关键接口 |
|------|------|---------|
| `base.py` | 数据模型 + 检索抽象 | `Chunk`, `SearchResult`, `KnowledgeRetriever` Protocol |
| `config.py` | 配置 + 工厂 | `KnowledgeConfig`, `get_config()`, `get_retriever()` |
| `parser.py` | 文档解析 | `parse_document()`, `_normalize_pdf_text()` |
| `chunker.py` | 智能分块 | `chunk_document()` (naive/section/qa) |
| `entity_extractor.py` | NIR 实体提取 | `extract_entities()`, 中英文别名映射 |
| `vectorstore.py` | 向量存储 | `ChromaDBRetriever`, `_sanitise_metadata()` |
| `cli.py` | CLI 管理 | add/import-dir/list/search/delete/rebuild/stats |
| `_search_server.py` | HTTP 服务 | /health, /documents, /stats, /search, DELETE |
| `_import_papers.py` | 导入脚本 | 设置环境变量 + 批量导入 |

### 3.2 抽象接口 `KnowledgeRetriever`

```python
class KnowledgeRetriever(Protocol):
    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[SearchResult]: ...
    def get_related_entities(self, entity: str, relation: str | None = None) -> list[str]: ...
    def add_documents(self, chunks: list[Chunk]) -> int: ...
    def delete_document(self, doc_id: str) -> bool: ...
    def list_documents(self) -> list[dict]: ...
```

当前唯一实现：`ChromaDBRetriever`。未来扩展 `GraphEnhancedRetriever`（ChromaDB + Neo4j），零现有代码修改。

### 3.3 文档解析器

**支持的格式与依赖：**

| 格式 | 依赖 | 解析方式 | 特殊处理 |
|------|------|---------|---------|
| `.pdf` | `pypdf` | PyPDFLoader | `_normalize_pdf_text()`: NFKC + CJK 空格消除 |
| `.docx` | `python-docx` | 段落提取 → Markdown 标题层级 | — |
| `.txt` / `.md` | 无 | 直接读取 (UTF-8) | — |
| `.html` / `.htm` | `markdownify` | HTML → Markdown | — |
| `.csv` | `pandas` | DataFrame → Markdown 表格 | — |

**中文 PDF 文本规范化** (`_normalize_pdf_text`):

pypdf 提取中文 PDF 时会产生两类问题：
1. **CJK 字符间多余空格**：`近 红 外 光 谱` → `近红外光谱`
2. **全角字符**：`ｍ→m`, `２→2`, `＿→_`

处理方法：
```python
_CJK_RANGE = r"\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
_CJK_SPACE_CJK = re.compile(rf"(?<=[{_CJK_RANGE}])\s+(?=[{_CJK_RANGE}])")

def _normalize_pdf_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)      # 全角→半角
    text = _CJK_SPACE_CJK.sub("", text)              # CJK间空格→消除
    return text
```

### 3.4 智能分块器

| 策略 | 适用场景 | 说明 |
|------|---------|------|
| `section` | 论文 (默认) | 按 Markdown `#`/`##`/`###` 标题切分，超长段落递归 naive |
| `naive` | 通用 | 按段落合并到 max_tokens (512)，支持重叠 |
| `qa` | 问答型文档 | 提取 Q→A 对 (`Q:`/`问：` → `A:`/`答：`)，退化回 naive |

### 3.5 实体提取器

**NIR 领域实体类别：**

| 类别 | 示例 | 中文别名 |
|------|------|---------|
| 预处理方法 | snv, msc, sg_smooth, derivative1, airpls, asls | 标准正态变量→snv, 多元散射校正→msc, 一阶导数→derivative1 |
| 建模方法 | pls, pcr, svr, random_forest, cnn, lstm | 偏最小二乘→pls, 支持向量回归→svr |
| 变量选择 | spa, cars, uve, vip, mcuves | 连续投影算法→spa, 竞争性自适应重加权采样→cars |
| 评价指标 | r2, rmse, rmsecv, rmsep, rpd | 决定系数→r2, 均方根误差→rmse |
| 数据集 | soil, corn, wheat, rice, meat | — |

**匹配策略：**
- 英文术语：`\b` 词边界正则匹配
- 中文术语/CJK 字符：子串匹配（Python `\b` 不支持 CJK）

### 3.6 向量存储

```python
class ChromaDBRetriever:
    """ChromaDB 向量检索实现"""

    def __init__(self, db_path, embedding_model, collection_name):
        self._model = SentenceTransformer(embedding_model)     # BGE-M3, 1024维
        self._client = chromadb.PersistentClient(path=db_path)  # 本地持久化
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def search(self, query, top_k=5, where=None):
        query_emb = self._model.encode([query])
        results = self._collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k, where=where)
        # cosine distance → similarity: score = 1 - distance

    def add_documents(self, chunks):
        # 批量编码 + 写入, metadata 经 _sanitise_metadata 清洗

    def delete_document(self, doc_id):
        self._collection.delete(where={"doc_id": doc_id})

    def list_documents(self):
        # 汇总所有文档 + chunk_count
```

**Metadata 清洗规则** (`_sanitise_metadata`):

| 值类型 | 处理 | 原因 |
|--------|------|------|
| `None` | **跳过该 key** | ChromaDB 不接受 null metadata 值 |
| `str/int/float/bool` | 原样保留 | ChromaDB 原生支持 |
| `list/tuple` | `json.dumps()` 序列化 | ChromaDB 不支持数组类型 |
| `dict` | `json.dumps()` 序列化 | ChromaDB 不支持嵌套对象 |
| 其他 | `str()` 转换 | 兜底 |

---

## 4. 搜索服务器层

### 4.1 为什么需要搜索服务器

| 方案 | 问题 |
|------|------|
| 直接在 Docker 容器中 import nir_core.knowledge | 需要 chromadb + sentence-transformers + torch (~2GB)，且 numpy 2.x 与 h5py/bottleneck 不兼容 |
| 子进程调用 Anaconda Python | Docker 容器内无法访问宿主机的 `D:\CodeTool\Anaconda\python.exe` |
| **HTTP 搜索服务器 (当前方案)** | 宿主机运行轻量 HTTP 服务，Docker 容器通过 `host.docker.internal:8089` 调用 |

### 4.2 环境变量链

搜索服务器启动前必须设置（在 `_search_server.py` 顶部）：

```python
os.environ["USE_TF"] = "0"                # 禁用 TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"     # transformers 不加载 TF
os.environ["HF_HUB_OFFLINE"] = "1"         # 禁止 HuggingFace 网络请求
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # 离线模式
```

### 4.3 API 端点一览

| 方法 | 路径 | 请求体 | 响应 | 说明 |
|------|------|--------|------|------|
| GET | `/health` | — | `{"status": "ok"}` | 健康检查 |
| GET | `/documents` | — | `{"documents": [...], "count": N}` | 列出所有文档 |
| GET | `/stats` | — | `{"documents": N, "total_chunks": M, ...}` | 知识库统计 |
| POST | `/search` | `{"query": "...", "top_k": 5}` | `{"results": [...], "count": N}` | 语义检索 |
| POST | `/documents` | `{"filename": "...", "content_b64": "...", "title": "...", "year": 2023}` | `{"doc_id": "...", "chunks_added": N}` | 添加文档 |
| DELETE | `/documents/{doc_id}` | — | `{"deleted": true}` | 删除文档 |

### 4.4 文档入库管线 (POST /documents)

```
1. base64 解码 → 原始文件字节
2. 大小检查 (≤ 50MB)
3. 写入临时文件 (保留原始扩展名供 parser 判断格式)
4. parse_document(tmp_path) → Markdown
5. chunk_document(markdown, strategy="section", max_tokens=512) → list[Chunk]
6. extract_entities(markdown) → {methods, models, datasets, metrics}
7. chunk.metadata.update({title, year, **entities})
8. ChromaDBRetriever.add_documents(chunks) → 写入 ChromaDB
9. 删除临时文件
10. 返回 {doc_id, chunks_added, title, methods, models, datasets, metrics}
```

---

## 5. 后端层详细设计

### 5.1 FastAPI Router (`knowledge.py`)

所有端点都是对搜索服务器的 HTTP 代理：

| 端点 | 方法 | 搜索服务器映射 | 特殊处理 |
|------|------|---------------|---------|
| `/api/knowledge/documents` | GET | → GET /documents | — |
| `/api/knowledge/documents` | POST | → POST /documents | multipart → base64 转换 |
| `/api/knowledge/documents/{doc_id}` | DELETE | → DELETE /documents/{doc_id} | URL 编码 doc_id |
| `/api/knowledge/stats` | GET | → GET /stats | — |
| `/api/knowledge/search` | POST | → POST /search | — |

### 5.2 URL 解析策略

```python
def _kb_base_url() -> str:
    # 优先级: NIR_KNOWLEDGE_URL 环境变量 → host.docker.internal:8089 → localhost:8089
    url = os.environ.get("NIR_KNOWLEDGE_URL", "").rstrip("/")
    if url:
        return url
    return "http://host.docker.internal:8089"
```

### 5.3 上传流程 (POST /documents)

```
1. 接收 multipart: file + title (可选) + year (可选)
2. 文件类型白名单检查: .pdf, .docx, .txt, .md, .html, .csv
3. 大小检查: ≤ 50MB
4. base64 编码: content_b64 = base64.b64encode(raw).decode("ascii")
5. JSON 转发: POST http://host.docker.internal:8089/documents
              Body: {filename, content_b64, title, year}
6. 返回 {doc_id, chunks_added, title, methods, models, datasets, metrics}
```

### 5.4 Agent 工具 (`nir_search_knowledge`)

```python
@tool("nir_search_knowledge", parse_docstring=True)
def nir_search_knowledge_tool(runtime, query, top_k=5):
    """Search the NIR knowledge base for relevant paper sections."""

    # 快速路径: 直接 import (本地开发环境)
    if _knowledge_search_mode in (None, "direct"):
        try:
            from nir_core.knowledge.config import get_retriever
            retriever = get_retriever()
            results = retriever.search(query, top_k=top_k)
            _knowledge_search_mode = "direct"
            return _ok(process(results))
        except Exception:
            _knowledge_search_mode = "http"

    # 回退路径: HTTP 搜索服务器 (Docker 环境)
    return _search_knowledge_via_http(query, top_k)
```

缓存机制：`_knowledge_search_mode` 缓存检测结果，避免每次调用都尝试 import。

---

## 6. 前端界面设计

### 6.1 入口位置

```
设置对话框 (SettingsDialog)
  ├── 通用 (General)
  ├── 模型 (Models)
  ├── 技能 (Skills)
  ├── 知识库 (Knowledge)    ← 新增
  └── 关于 (About)
```

左侧菜单图标：`LibraryIcon` (lucide-react)

### 6.2 页面布局

```
┌─────────────────────────────────────────────────────────────────────┐
│  知识库                                                              │
│  管理 NIR 近红外知识库中的文档。上传的论文会自动解析、分块、提取     │
│  实体并建立向量索引，供智能体检索引用。                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐                                │
│  │ 文档数        │  │ 分块数        │     [刷新] [上传文档]          │
│  │     2         │  │    67        │                                │
│  └──────────────┘  └──────────────┘                                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ 标题              │ 年份  │ 分块数 │ 操作                       ││
│  ├───────────────────┼───────┼────────┼─────────────────────────────││
│  │ 📄 基于机器学习的  │  —    │   53   │                    [删除]  ││
│  │    近红外光谱...   │       │        │                            ││
│  ├───────────────────┼───────┼────────┼─────────────────────────────││
│  │ 📄 基于深度学习的  │  —    │   14   │                    [删除]  ││
│  │    近红外光谱...   │       │        │                            ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  搜索测试                                                           │
│  ┌────────────────────────────────────────────┐ [搜索]             │
│  │ 输入查询内容，例如：SNV 与 MSC 的区别       │                    │
│  └────────────────────────────────────────────┘                    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ 基于深度学习的近红外光谱定量分析模型研究_卢丰.pdf   score: 0.72 ││
│  │                                                                 ││
│  │ 深度学习在近红外光谱分析中的应用主要包括一维卷积神经网络        ││
│  │ (1D-CNN)、长短期记忆网络(LSTM)以及混合架构CNN-LSTM...          ││
│  │                                                                 ││
│  │ [cnn] [lstm] [pls] [snv] [rmsep]                               ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 组件分解

#### `KnowledgeSettingsPage` (主组件)

| 状态 | 类型 | 说明 |
|------|------|------|
| `documents` | `KnowledgeDocument[]` | 文档列表 |
| `stats` | `KnowledgeStats \| null` | 统计信息 |
| `loading` | `boolean` | 加载中 |
| `error` | `string \| null` | 错误信息 |
| `uploadOpen` | `boolean` | 上传对话框开关 |
| `deleteTarget` | `KnowledgeDocument \| null` | 删除目标 (null=关闭) |

**生命周期：**
- `onMount` → `loadAll()` 并行请求文档列表 + 统计
- 上传成功 → toast + 关闭对话框 + `loadAll()`
- 删除成功 → toast + 关闭对话框 + `loadAll()`

#### `StatCard` (统计卡片)

```
┌──────────────┐
│ 文档数        │   ← text-xs text-muted-foreground
│     2         │   ← text-2xl font-semibold
└──────────────┘
```

#### `DocumentTable` (文档表格)

| 列 | 字段 | 宽度 | 说明 |
|----|------|------|------|
| 标题 | `title \n source` | flex | FileIcon + 标题(粗体) + 来源(小字) |
| 年份 | `year ?? "—"` | 固定 | — |
| 分块数 | `chunk_count` | 固定 | — |
| 操作 | 删除按钮 | 固定 | ghost 按钮 + Trash2Icon |

表格样式：`overflow-hidden rounded-lg border`，表头 `bg-muted/50`。

#### `UploadDialog` (上传对话框)

```
┌─────────────────────────────────────────────────┐
│  上传文档                                         │
│  支持 PDF、DOCX、TXT、MD、HTML、CSV 格式          │
├─────────────────────────────────────────────────┤
│                                                   │
│  [📄 选择文件...]                                 │
│                                                   │
│  标题（可选）                                      │
│  ┌─────────────────────────────────────────────┐ │
│  │ 留空则使用文件名                             │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  年份（可选）                                      │
│  ┌─────────────────────────────────────────────┐ │
│  │ 如 2023                                     │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│                [取消]  [📤 上传文档]               │
└─────────────────────────────────────────────────┘
```

- 文件选择：隐藏 `<input type="file">` + 按钮触发
- accept: `.pdf,.docx,.txt,.md,.markdown,.html,.htm,.csv`
- 上传中：按钮禁用 + Loader2Icon 旋转动画
- 关闭时重置所有状态

#### `SearchTest` (搜索测试区)

```
──────────────────────────────────────────────
搜索测试
┌──────────────────────────────────────────┐ [🔍 搜索]
│ 输入查询内容，例如：SNV 与 MSC 的区别    │
└──────────────────────────────────────────┘

结果卡片:
┌──────────────────────────────────────────────────┐
│ 来源文件名.pdf                        score: 0.72│
│                                                    │
│ 匹配的文本内容 (line-clamp-4, 最多4行)...         │
│                                                    │
│ [snv] [msc] [pls] [rmsep]   ← 方法实体 Badge     │
└──────────────────────────────────────────────────┘
```

- 输入框支持 Enter 键触发搜索
- score 显示为 `Badge variant="secondary"`
- 实体标签显示为 `Badge variant="outline"` + `text-xs`
- 结果内容 `line-clamp-4` 限制 4 行

#### `DeleteConfirmDialog` (删除确认)

```
┌─────────────────────────────────────────────────┐
│  删除文档？                                       │
│  此操作将永久删除该文档及其所有分块，无法撤销。    │
├─────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐ │
│  │ 基于机器学习的近红外光谱定量分析模型研究       │ │
│  │ 基于机器学习的近红外光谱定量分析模型研究_赵峰媛 │ │
│  │ 53 分块数                                    │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│                        [取消]  [🗑 删除]           │
└─────────────────────────────────────────────────┘
```

### 6.4 UI 组件库依赖

全部使用 DeerFlow 现有 shadcn/ui 组件：

| 组件 | 用途 |
|------|------|
| `Button` | 上传、刷新、搜索、删除 |
| `Input` | 搜索框、标题、年份 |
| `Dialog` / `DialogContent` / `DialogHeader` / ... | 上传对话框、删除确认 |
| `Alert` / `AlertDescription` | 错误提示 |
| `Badge` | score 标签、实体标签 |
| `Skeleton` | 加载骨架屏 |
| `SettingsSection` | 页面标题 + 描述容器 |

图标 (lucide-react)：`FileIcon`, `UploadIcon`, `Trash2Icon`, `RefreshCwIcon`, `SearchIcon`, `Loader2Icon`, `LibraryIcon`

### 6.5 国际化 (i18n)

所有文案通过 `useI18n()` hook 读取，支持中文 (zh-CN) 和英文 (en-US)：

**中文键值：**

| 键 | 值 |
|----|-----|
| `settings.knowledge.title` | 知识库 |
| `settings.knowledge.description` | 管理 NIR 近红外知识库中的文档... |
| `settings.knowledge.empty` | 知识库为空。点击上方按钮上传第一篇文档。 |
| `settings.knowledge.statsDocuments` | 文档数 |
| `settings.knowledge.statsChunks` | 分块数 |
| `settings.knowledge.uploadButton` | 上传文档 |
| `settings.knowledge.uploadHint` | 支持 PDF、DOCX、TXT、MD、HTML、CSV 格式，单文件最大 50MB。 |
| `settings.knowledge.uploadSuccess` | 上传成功，共生成 {count} 个分块。 |
| `settings.knowledge.deleteConfirmTitle` | 删除文档？ |
| `settings.knowledge.deleteConfirmDescription` | 此操作将永久删除该文档及其所有分块，无法撤销。 |
| `settings.knowledge.searchTitle` | 搜索测试 |
| `settings.knowledge.searchPlaceholder` | 输入查询内容，例如：SNV 与 MSC 的区别 |
| `settings.knowledge.serverUnreachable` | 知识库服务器不可达。请在主机上启动... |

### 6.6 错误处理

| 场景 | 处理 |
|------|------|
| 搜索服务器不可达 (503) | Alert + "服务器不可达" 提示 + 启动命令 |
| 上传文件类型不支持 | 后端 400 → 前端 toast.error |
| 上传文件过大 (>50MB) | 后端 413 → 前端 toast.error |
| ChromaDB metadata None 值 | `_sanitise_metadata` 跳过 None key |
| 上传解析失败 | 搜索服务器 500 → 后端 500 → 前端 toast.error |
| Nginx body size 限制 | `client_max_body_size 100M` (server 块级别) |

---

## 7. 部署架构

### 7.1 Docker Compose 关系

```
┌─────────────────────────────────────────────────────────────┐
│  Docker Compose (docker-compose-dev.yaml)                    │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │  nginx       │   │  frontend    │   │  gateway      │    │
│  │  :2026       │──→│  :3000       │   │  :8001        │    │
│  │  proxy       │   │  Next.js     │   │  FastAPI      │    │
│  └──────────────┘   └──────────────┘   └──────┬───────┘    │
│                                                │             │
│                         host.docker.internal:8089             │
└────────────────────────────────────────────────┼─────────────┘
                                                 │
┌────────────────────────────────────────────────┼─────────────┐
│  宿主机 (Windows)                              │             │
│  ┌─────────────────────────────────────────────┴───────┐    │
│  │  _search_server.py (Anaconda Python, port 8089)     │    │
│  │  · ChromaDB (.chromadb/)                            │    │
│  │  · BGE-M3 1024维嵌入模型                           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Docker Volume 挂载

```yaml
# docker-compose-dev.yaml 关键挂载
services:
  gateway:
    volumes:
      - ../backend/:/app/backend/        # 代码热更新
      - ../nir_core:/app/nir_core        # nir_core 代码
      - gateway-venv:/app/backend/.venv  # Python 虚拟环境 (不共享)
    extra_hosts:
      - "host.docker.internal:host-gateway"  # 宿主机访问
```

### 7.3 Nginx 配置

```nginx
server {
    server_name _;
    client_max_body_size 100M;          # 所有 API 允许大文件上传

    location /api/ {
        proxy_pass http://gateway:8001;
        # ... proxy headers
    }

    location /api/threads/{thread_id}/uploads {
        client_max_body_size 100M;       # (冗余但保留)
        proxy_pass http://gateway:8001;
    }
}
```

### 7.4 启动与运维

**启动搜索服务器：**
```bash
D:\CodeTool\Anaconda\python.exe d:\NIR-Agent\deer-flow\nir_core\knowledge\_search_server.py
```

**重启 Docker 容器（代码更新后）：**
```bash
docker restart deer-flow-gateway deer-flow-frontend
```

**健康检查：**
```bash
# 搜索服务器
curl http://localhost:8089/health

# Gateway
docker exec deer-flow-gateway curl http://localhost:8001/health

# 端到端
docker exec deer-flow-gateway curl http://host.docker.internal:8089/health
```

**注意事项：**
- 搜索服务器需要手动启动，重启电脑后需重新运行
- 搜索服务器端口：8089（可通过 `--port` 参数修改）
- 嵌入模型路径：`D:\Models\bge-m3`（本地加载，不联网）

---

## 8. 依赖清单

### 8.1 核心库依赖 (`nir_core/pyproject.toml`)

```toml
[project.optional-dependencies]
knowledge = [
    "chromadb>=0.5.0",              # 本地向量数据库
    "sentence-transformers>=2.0",   # 本地嵌入模型
    "pypdf>=4.0",                   # PDF 解析
    "python-docx>=1.0",            # Word 解析
    "markdownify>=0.11",            # HTML → Markdown
    "langchain-community>=0.0.20",  # PyPDFLoader
]
```

### 8.2 嵌入模型

| 模型 | 维度 | 大小 | 中文支持 | 当前使用 |
|------|------|------|---------|---------|
| `BGE-M3` | 1024 | ~2.3GB | 优秀 | ✅ |
| `BAAI/bge-small-zh-v1.5` | 512 | ~100MB | 优秀 | 未来可选 |

模型下载：从 BAAI/bge-m3 下载到 `D:\Models\bge-m3`，通过
`NIR_KNOWLEDGE_EMBEDDING_MODEL` 配置本地路径。

---

## 9. 测试

### 9.1 单元测试 (54 个)

```
nir_core/tests/test_knowledge/
├── test_parser.py              # 15 个测试
│   ├── PDF 解析 (含中文规范化)
│   ├── DOCX / TXT / HTML / CSV 解析
│   └── 不支持的格式 → ValueError
├── test_chunker.py             # 16 个测试
│   ├── naive / section / qa 策略
│   ├── 重叠分块
│   └── 超长段落递归切分
├── test_entity_extractor.py    # 12 个测试
│   ├── 英文术语匹配
│   ├── 中文别名映射 (偏最小二乘→pls, 标准正态变量→snv, ...)
│   └── CJK 子串匹配 (非 \b 词边界)
├── test_cli.py                 # 5 个测试
│   ├── add / import-dir / list / search / delete
└── test_vectorstore.py         # 6 个测试
    ├── 添加 + 搜索 + 删除 + 列表
    └── 空库搜索
```

### 9.2 端到端验证

```
Docker 容器 → host.docker.internal:8089/health → 200 OK
Docker 容器 → host.docker.internal:8089/search → 正确返回结果
Gateway → /api/knowledge/documents → 200 OK
Gateway → /api/knowledge/stats → 200 OK
前端 → 设置 → 知识库 → 显示文档列表
```

---

## 10. 文件清单

```
nir_core/knowledge/
├── __init__.py                  # 模块入口
├── base.py                      # Chunk, SearchResult, KnowledgeRetriever Protocol
├── config.py                    # KnowledgeConfig + get_retriever 工厂
├── parser.py                    # 文档解析器 (PDF/Word/HTML/CSV → Markdown)
├── chunker.py                   # 智能分块器 (naive/section/qa)
├── entity_extractor.py          # NIR 领域实体提取 (中英文别名)
├── vectorstore.py               # ChromaDB 向量检索实现
├── cli.py                       # CLI 管理工具
├── _search_server.py            # HTTP 搜索服务器 (Docker-Host 桥接)
├── _import_papers.py            # 批量导入脚本 (设置环境变量)
├── .chromadb-bge-m3/            # ChromaDB 持久化数据 (gitignore)
└── .knowledge_catalog.bge-m3.sqlite3  # 治理目录 (gitignore)

nir_core/tests/test_knowledge/
├── __init__.py
├── test_parser.py
├── test_chunker.py
├── test_entity_extractor.py
├── test_vectorstore.py
└── test_cli.py

backend/
├── app/gateway/routers/knowledge.py    # FastAPI 知识库路由
├── packages/harness/deerflow/community/nir/tools.py  # nir_search_knowledge 工具
└── pyproject.toml                      # knowledge 可选依赖组

frontend/
├── src/components/workspace/settings/
│   ├── knowledge-settings-page.tsx     # 知识库设置页面
│   └── settings-dialog.tsx             # 设置对话框 (含 Knowledge tab)
└── src/core/i18n/locales/
    ├── zh-CN.ts                        # 中文翻译
    ├── en-US.ts                        # 英文翻译
    └── types.ts                        # i18n 类型定义

docker/nginx/nginx.conf                 # client_max_body_size 100M
```

---

## 11. 未来扩展

### 11.1 图谱检索 (Neo4j)

当需要以下能力时启用：
- 多跳查询："使用 SNV 的论文还用了什么方法？"
- 方法共现可视化："SNV 和 airPLS 常一起出现"
- 按实体关系过滤："在土壤数据集上验证过的方法"

扩展步骤（零现有代码修改）：

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `neo4j_client.py` | 新增 Neo4j 客户端 |
| 2 | `graph_retriever.py` | 新增 GraphEnhancedRetriever |
| 3 | `config.py` | `graph_backend = "neo4j"` (改 1 行) |
| 4 | `cli.py` | 加 `build-graph` 命令 |

### 11.2 嵌入模型升级

替换为 `BAAI/bge-small-zh-v1.5` (512 维，中文优化)：
- 修改 `_search_server.py` 中的 `_model_path`
- 运行 `cli rebuild` + `cli import-dir` 重建索引

### 11.3 搜索服务器自动启动

可选方案：
- Windows 服务 (nssm)
- 启动脚本 (`shell:startup`)
- Docker Compose `extra_service` (需要 Anaconda 镜像)

### 11.4 批量上传与进度

- 前端：拖拽上传区域 + 多文件选择 + 进度条
- 后端：批量接口 + WebSocket 进度推送

### 11.5 文档预览

- 前端：点击文档行 → 展开分块列表 + 高亮实体
- 后端：`GET /documents/{doc_id}/chunks` 端点
