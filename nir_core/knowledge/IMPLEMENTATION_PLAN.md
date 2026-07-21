# NIR 知识库 RAG 系统实现方案 v1.0

## 1. 目标

为 NIR-Agent 构建本地知识库 RAG 系统，支持：
- 论文 PDF/Word/Markdown 自动解析 → 智能分块 → 向量化 → 持久化存储
- 智能体通过 `nir_search_knowledge` 工具语义检索相关论文段落
- 支持 100+ 篇论文，检索延迟 <100ms
- 未来可扩展图谱检索（Neo4j），不修改现有代码

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    CLI 管理层                             │
│  python -m nir_core.knowledge.cli                       │
│  (add / import-dir / list / search / rebuild / delete)   │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│                    文档处理流水线                          │
│  Parser (PDF→MD) → Chunker (智能分块) → EntityExtractor  │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│              KnowledgeRetriever (抽象接口)                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │         ChromaDBRetriever (当前实现)             │    │
│  │  - 向量检索 (HNSW)                              │    │
│  │  - 元数据过滤 (where)                           │    │
│  │  - 本地持久化 (.chromadb/)                      │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │    GraphEnhancedRetriever (未来扩展, 零修改)      │    │
│  │  - ChromaDB 向量检索                             │    │
│  │  - Neo4j 图谱扩展 (关联实体)                     │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│              Agent 工具层 (tools.py)                      │
│  nir_search_knowledge(query, top_k=5)                   │
│  → 返回相关论文段落 + 来源 + 关联实体                      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
论文 PDF
  ↓ Parser.parse(file_path) → Markdown 文本
  ↓ Chunker.chunk(markdown, strategy="section") → list[Chunk]
  ↓ EntityExtractor.extract(markdown) → 实体列表 (methods, datasets, metrics)
  ↓ ChromaDBRetriever.add(chunks, metadata)
  ↓ 持久化到 .chromadb/
  
智能体分析时:
  nir_search_knowledge("SNV scatter correction soil")
  ↓ ChromaDBRetriever.search(query, top_k=5)
  ↓ 返回 top-5 相关段落 + 来源论文 + 元数据
  ↓ 智能体根据论文指导调整策略
```

### 2.3 可扩展性设计：接口抽象

核心原则：**Agent 工具层不感知后端实现**。

```python
# nir_core/knowledge/base.py

from __future__ import annotations
from typing import Protocol
from pydantic import BaseModel


class Chunk(BaseModel):
    """文档块"""
    id: str
    content: str
    source: str                    # 论文文件名
    chunk_index: int
    metadata: dict = {}            # title, authors, year, entities, ...


class SearchResult(BaseModel):
    """检索结果"""
    chunk: Chunk
    score: float
    related_entities: list[str] = []   # 预留：未来图谱扩展


class KnowledgeRetriever(Protocol):
    """检索层抽象接口
    
    当前实现: ChromaDBRetriever
    未来扩展: GraphEnhancedRetriever (ChromaDB + Neo4j)
    Agent 工具层只依赖此接口，不依赖具体实现。
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """语义检索，返回相关文档块
        
        Args:
            query: 自然语言查询
            top_k: 返回数量
            where: 元数据过滤 (如 {"year": {"$gte": 2020}})
        """
        ...

    def get_related_entities(
        self,
        entity: str,
        relation: str | None = None,
    ) -> list[str]:
        """获取关联实体
        
        当前实现: 返回空列表
        未来扩展: 从 Neo4j 图谱查询
        """
        ...

    def add_documents(
        self,
        chunks: list[Chunk],
    ) -> int:
        """添加文档块到知识库
        
        Returns: 添加的块数
        """
        ...

    def delete_document(self, doc_id: str) -> bool:
        """删除指定文档的所有块
        
        Returns: 是否删除成功
        """
        ...

    def list_documents(self) -> list[dict]:
        """列出知识库中的所有文档"""
        ...
```

### 2.4 当前实现：ChromaDBRetriever

```python
# nir_core/knowledge/vectorstore.py

import chromadb
from sentence_transformers import SentenceTransformer


class ChromaDBRetriever:
    """ChromaDB 向量检索实现
    
    - 本地持久化，零外部服务
    - HNSW 索引，10K chunks <10ms
    - 支持 where 元数据过滤
    """

    def __init__(
        self,
        db_path: str = "nir_core/knowledge/.chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "nir_papers",
    ):
        self._model = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query, top_k=5, where=None):
        query_emb = self._model.encode([query])
        results = self._collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            where=where,
        )
        return [
            SearchResult(
                chunk=Chunk(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    source=results["metadatas"][0][i].get("source", ""),
                    chunk_index=results["metadatas"][0][i].get("chunk_index", 0),
                    metadata=results["metadatas"][0][i],
                ),
                score=1 - results["distances"][0][i],  # cosine distance → similarity
                related_entities=[],  # 当前为空
            )
            for i in range(len(results["ids"][0]))
        ]

    def get_related_entities(self, entity, relation=None):
        return []  # 当前不支持

    def add_documents(self, chunks):
        for chunk in chunks:
            emb = self._model.encode([chunk.content])[0]
            self._collection.add(
                ids=[chunk.id],
                embeddings=[emb.tolist()],
                documents=[chunk.content],
                metadatas=[{**chunk.metadata, "source": chunk.source, "chunk_index": chunk.chunk_index}],
            )
        return len(chunks)

    def delete_document(self, doc_id):
        self._collection.delete(where={"doc_id": doc_id})
        return True

    def list_documents(self):
        all_meta = self._collection.get(include=["metadatas"])
        docs = {}
        for meta in all_meta["metadatas"]:
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "year": meta.get("year"),
                }
        return list(docs.values())
```

### 2.5 未来扩展：GraphEnhancedRetriever（零现有代码修改）

```python
# nir_core/knowledge/graph_retriever.py (未来新增，不改现有文件)

class GraphEnhancedRetriever:
    """ChromaDB + Neo4j 混合检索
    
    新增此文件，不改任何现有代码。
    通过 config.py 的 graph_backend 配置激活。
    """

    def __init__(self, chroma: ChromaDBRetriever, neo4j_client):
        self._chroma = chroma
        self._graph = neo4j_client

    def search(self, query, top_k=5, where=None):
        # 1. 向量检索 (ChromaDB)
        results = self._chroma.search(query, top_k, where)
        
        # 2. 图谱扩展 (Neo4j)
        for r in results:
            entities = r.chunk.metadata.get("entities", [])
            related = []
            for entity in entities:
                related.extend(self._graph.get_related(entity))
            r.related_entities = list(set(related))
        
        return results

    def get_related_entities(self, entity, relation=None):
        return self._graph.get_related(entity, relation)

    # add_documents / delete_document / list_documents 代理到 chroma
    def add_documents(self, chunks):
        count = self._chroma.add_documents(chunks)
        # 同时构建图谱边
        for chunk in chunks:
            entities = chunk.metadata.get("entities", [])
            self._graph.build_edges(chunk.id, entities)
        return count

    def delete_document(self, doc_id):
        self._chroma.delete_document(doc_id)
        self._graph.delete_doc_nodes(doc_id)
        return True

    def list_documents(self):
        return self._chroma.list_documents()
```

## 3. 模块详细设计

### 3.1 文档解析器 `nir_core/knowledge/parser.py`

**来源**：从 yuxi-knowledge `unified.py` 提取核心逻辑，去除 MinIO 依赖。

**支持的格式**：

| 格式 | 依赖 | 解析方式 |
|------|------|---------|
| `.pdf` | `langchain-community` + `pypdf` | PyPDFLoader |
| `.docx` | `python-docx` | 段落提取 → Markdown |
| `.txt` / `.md` | 无 | 直接读取 |
| `.html` | `markdownify` | HTML → Markdown |
| `.csv` | `pandas` | DataFrame → Markdown 表格 |

```python
# nir_core/knowledge/parser.py

from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".csv")


def parse_document(file_path: str | Path, params: dict | None = None) -> str:
    """统一文档解析入口，返回 Markdown 文本
    
    Args:
        file_path: 文件路径
        params: 解析参数 (如 enable_ocr)
    
    Returns:
        Markdown 格式的文档内容
    
    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = path.suffix.lower()
    
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    
    if ext == ".pdf":
        return _parse_pdf(path, params)
    
    if ext == ".docx":
        return _parse_docx(path)
    
    if ext in (".html", ".htm"):
        return _parse_html(path)
    
    if ext == ".csv":
        return _parse_csv(path)
    
    raise ValueError(f"Unsupported file type: {ext}")


def _parse_pdf(path: Path, params: dict | None = None) -> str:
    """PDF → Markdown (使用 PyPDFLoader)"""
    from langchain_community.document_loaders import PyPDFLoader
    
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs)


def _parse_docx(path: Path) -> str:
    """DOCX → Markdown (使用 python-docx)"""
    import docx
    
    doc = docx.Document(str(path))
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name.lower()
        if "heading 1" in style:
            lines.append(f"# {text}")
        elif "heading 2" in style:
            lines.append(f"## {text}")
        elif "heading 3" in style:
            lines.append(f"### {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def _parse_html(path: Path) -> str:
    """HTML → Markdown"""
    from markdownify import markdownify as md_convert
    
    content = path.read_text(encoding="utf-8")
    return md_convert(content, heading_style="ATX")


def _parse_csv(path: Path) -> str:
    """CSV → Markdown 表格"""
    import pandas as pd
    
    df = pd.read_csv(path)
    return df.to_markdown(index=False)
```

### 3.2 智能分块器 `nir_core/knowledge/chunker.py`

**来源**：从 yuxi-knowledge `general.py` + `dispatcher.py` 提取，简化为纯函数。

**分块策略**：

| 策略 | 适用场景 | 说明 |
|------|---------|------|
| `naive` | 通用 | 按段落+token 数合并，支持重叠 |
| `section` | 论文 | 按 Markdown 标题 (#, ##) 切分 |
| `qa` | 问答型文档 | 提取 Q→A 对 |

```python
# nir_core/knowledge/chunker.py

import re
import uuid
from typing import Literal

from nir_core.knowledge.base import Chunk


def chunk_document(
    markdown: str,
    source: str,
    strategy: Literal["naive", "section", "qa"] = "section",
    max_tokens: int = 512,
    overlap_percent: int = 0,
    doc_id: str | None = None,
) -> list[Chunk]:
    """将 Markdown 文档分块
    
    Args:
        markdown: 文档 Markdown 文本
        source: 源文件名
        strategy: 分块策略
        max_tokens: 每块最大 token 数 (naive 策略)
        overlap_percent: 块间重叠百分比 (naive 策略)
        doc_id: 文档 ID (不传则自动生成)
    
    Returns:
        文档块列表
    """
    doc_id = doc_id or str(uuid.uuid4())
    
    if strategy == "naive":
        chunks_text = _chunk_naive(markdown, max_tokens, overlap_percent)
    elif strategy == "section":
        chunks_text = _chunk_by_section(markdown, max_tokens)
    elif strategy == "qa":
        chunks_text = _chunk_qa(markdown)
    else:
        chunks_text = _chunk_naive(markdown, max_tokens, overlap_percent)
    
    return [
        Chunk(
            id=f"{doc_id}_chunk_{i}",
            content=text,
            source=source,
            chunk_index=i,
            metadata={"doc_id": doc_id},
        )
        for i, text in enumerate(chunks_text)
        if text.strip()
    ]


def _chunk_naive(text: str, max_tokens: int, overlap: int) -> list[str]:
    """按段落合并到 max_tokens，支持重叠"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0
    
    for para in paragraphs:
        para_len = len(para.split())  # 简单用词数估算 token
        if current_len + para_len > max_tokens and current:
            chunks.append("\n\n".join(current))
            # 重叠
            if overlap > 0:
                overlap_size = max(1, len(current) * overlap // 100)
                current = current[-overlap_size:]
                current_len = sum(len(c.split()) for c in current)
            else:
                current = []
                current_len = 0
        current.append(para)
        current_len += para_len
    
    if current:
        chunks.append("\n\n".join(current))
    
    return chunks


def _chunk_by_section(text: str, max_tokens: int) -> list[str]:
    """按 Markdown 标题分块，超长段落再用 naive"""
    # 按 # / ## / ### 标题分割
    sections = re.split(r'^(#{1,3}\s+.+)$', text, flags=re.MULTILINE)
    
    chunks = []
    current_header = ""
    current_content = ""
    
    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r'^#{1,3}\s+', part):
            # 新标题
            if current_content.strip():
                full_section = f"{current_header}\n\n{current_content}".strip()
                # 如果该节太长，用 naive 再切
                if len(full_section.split()) > max_tokens:
                    sub_chunks = _chunk_naive(full_section, max_tokens, 0)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(full_section)
            current_header = part
            current_content = ""
        else:
            current_content += part + "\n"
    
    # 处理最后一节
    if current_content.strip():
        full_section = f"{current_header}\n\n{current_content}".strip()
        if len(full_section.split()) > max_tokens:
            chunks.extend(_chunk_naive(full_section, max_tokens, 0))
        else:
            chunks.append(full_section)
    
    return chunks


def _chunk_qa(text: str) -> list[str]:
    """提取 Q→A 对，每对作为一个块"""
    # 匹配 "Q: ... A: ..." 或 "问：... 答：..." 格式
    qa_pattern = r'(?:^|\n)(?:Q[:：]|问[:：])\s*(.+?)(?=\n(?:A[:：]|答[:：]))\s*\n(?:A[:：]|答[:：])\s*(.+?)(?=\n(?:Q[:：]|问[:：])|$)'
    matches = re.findall(qa_pattern, text, re.DOTALL)
    
    if not matches:
        # 退化为 naive
        return _chunk_naive(text, 512, 0)
    
    return [f"Q: {q.strip()}\nA: {a.strip()}" for q, a in matches]
```

### 3.3 实体提取器 `nir_core/knowledge/entity_extractor.py`

**作用**：从论文中提取 NIR 领域实体，存入 metadata，为未来图谱建边准备。

```python
# nir_core/knowledge/entity_extractor.py

import re

# NIR 领域实体词典
_PREPROCESSING_METHODS = {
    "snv", "msc", "sg_smooth", "savitzky-golay", "derivative", "derivative1",
    "derivative2", "airpls", "asls", "detrend", "mean_center", "autoscale",
    "normalize", "snv_detrend", "osc", "ems",
}

_MODELING_METHODS = {
    "pls", "pcr", "svr", "svr_rbf", "svr_linear", "svr_poly",
    "ridge", "lasso", "elasticnet", "random_forest", "xgboost",
    "ann", "cnn", "lstm",
}

_VARIABLE_SELECTION = {
    "spa", "cars", "uve", "vip", "mcuves", "irf", "boruta",
}

_METRICS = {
    "r2", "rmse", "rmsecv", "rmsep", "rpd", "rpdq", "bias", "slope",
    "mae", "mse", "aicc",
}

_DATASETS = {
    "soil", "corn", "wheat", "rice", "barley", "oat",
    "meat", "milk", "fruit", "coffee", "tea",
}


def extract_entities(markdown: str) -> dict[str, list[str]]:
    """从 Markdown 文本提取 NIR 领域实体
    
    Returns:
        dict with keys: methods, models, variable_selection, metrics, datasets
    """
    text_lower = markdown.lower()
    
    def _find(terms: set[str]) -> list[str]:
        found = []
        for term in terms:
            # 匹配完整词（边界匹配）
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                found.append(term)
        return found
    
    return {
        "methods": _find(_PREPROCESSING_METHODS),
        "models": _find(_MODELING_METHODS),
        "variable_selection": _find(_VARIABLE_SELECTION),
        "metrics": _find(_METRICS),
        "datasets": _find(_DATASETS),
    }
```

### 3.4 配置 `nir_core/knowledge/config.py`

```python
# nir_core/knowledge/config.py

from dataclasses import dataclass


@dataclass
class KnowledgeConfig:
    """知识库配置
    
    当前: 仅 ChromaDB (向量检索)
    未来: 设 graph_backend="neo4j" 启用图谱扩展
    """
    # 向量检索后端 (当前唯一)
    vector_backend: str = "chromadb"
    
    # 图谱后端 (当前 None, 未来 "neo4j")
    graph_backend: str | None = None
    
    # ChromaDB 配置
    chroma_path: str = "nir_core/knowledge/.chromadb"
    collection_name: str = "nir_papers"
    
    # 嵌入模型
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # 分块配置
    chunk_strategy: str = "section"
    chunk_max_tokens: int = 512
    chunk_overlap: int = 0
    
    # Neo4j 配置 (未来使用, 当前忽略)
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None


_default_config: KnowledgeConfig | None = None


def get_config() -> KnowledgeConfig:
    global _default_config
    if _default_config is None:
        _default_config = KnowledgeConfig()
    return _default_config


def get_retriever(config: KnowledgeConfig | None = None):
    """根据配置返回检索器实例
    
    当前: 返回 ChromaDBRetriever
    未来: 如果 graph_backend="neo4j", 返回 GraphEnhancedRetriever
    """
    cfg = config or get_config()
    
    from nir_core.knowledge.vectorstore import ChromaDBRetriever
    chroma = ChromaDBRetriever(
        db_path=cfg.chroma_path,
        embedding_model=cfg.embedding_model,
        collection_name=cfg.collection_name,
    )
    
    if cfg.graph_backend == "neo4j" and cfg.neo4j_uri:
        # 未来扩展: 导入 GraphEnhancedRetriever
        # from nir_core.knowledge.graph_retriever import GraphEnhancedRetriever
        # from nir_core.knowledge.neo4j_client import Neo4jClient
        # neo4j = Neo4jClient(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
        # return GraphEnhancedRetriever(chroma, neo4j)
        raise NotImplementedError("Graph backend not yet implemented")
    
    return chroma
```

### 3.5 CLI 管理工具 `nir_core/knowledge/cli.py`

```python
# nir_core/knowledge/cli.py
"""NIR 知识库 CLI 管理工具

用法:
  python -m nir_core.knowledge.cli add paper.pdf --title "..." --year 2023
  python -m nir_core.knowledge.cli import-dir ./papers/
  python -m nir_core.knowledge.cli list
  python -m nir_core.knowledge.cli search "SNV scatter correction"
  python -m nir_core.knowledge.cli delete <doc_id>
  python -m nir_core.knowledge.cli rebuild
  python -m nir_core.knowledge.cli stats
"""

import argparse
import sys
from pathlib import Path

from nir_core.knowledge.config import get_retriever, get_config
from nir_core.knowledge.parser import parse_document, SUPPORTED_EXTENSIONS
from nir_core.knowledge.chunker import chunk_document
from nir_core.knowledge.entity_extractor import extract_entities


def cmd_add(args):
    """添加单篇文档"""
    retriever = get_retriever()
    cfg = get_config()
    
    markdown = parse_document(args.file_path)
    doc_id = Path(args.file_path).stem
    
    chunks = chunk_document(
        markdown,
        source=args.file_path,
        strategy=cfg.chunk_strategy,
        max_tokens=cfg.chunk_max_tokens,
        doc_id=doc_id,
    )
    
    # 提取实体
    entities = extract_entities(markdown)
    
    # 补充元数据
    for chunk in chunks:
        chunk.metadata.update({
            "title": args.title or doc_id,
            "year": args.year,
            "authors": args.authors or [],
            **entities,
        })
    
    count = retriever.add_documents(chunks)
    print(f"Added {count} chunks from {args.file_path}")


def cmd_import_dir(args):
    """批量导入目录下所有文档"""
    retriever = get_retriever()
    cfg = get_config()
    
    dir_path = Path(args.directory)
    files = [f for f in dir_path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if not files:
        print(f"No supported files found in {args.directory}")
        return
    
    print(f"Found {len(files)} files to import...")
    
    total_chunks = 0
    for i, file_path in enumerate(files, 1):
        try:
            markdown = parse_document(str(file_path))
            doc_id = file_path.stem
            
            chunks = chunk_document(
                markdown,
                source=str(file_path),
                strategy=cfg.chunk_strategy,
                max_tokens=cfg.chunk_max_tokens,
                doc_id=doc_id,
            )
            
            entities = extract_entities(markdown)
            for chunk in chunks:
                chunk.metadata.update({
                    "title": doc_id,
                    **entities,
                })
            
            retriever.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  [{i}/{len(files)}] {file_path.name} → {len(chunks)} chunks")
        except Exception as e:
            print(f"  [{i}/{len(files)}] {file_path.name} → ERROR: {e}")
    
    print(f"\nDone: {len(files)} files, {total_chunks} chunks total")


def cmd_list(args):
    """列出知识库中的所有文档"""
    retriever = get_retriever()
    docs = retriever.list_documents()
    
    if not docs:
        print("Knowledge base is empty.")
        return
    
    print(f"{'doc_id':<40} {'title':<50} {'year':>6}")
    print("-" * 100)
    for doc in docs:
        print(f"{doc.get('doc_id', ''):<40} {doc.get('title', ''):<50} {doc.get('year', ''):>6}")


def cmd_search(args):
    """搜索知识库"""
    retriever = get_retriever()
    results = retriever.search(args.query, top_k=args.top_k)
    
    if not results:
        print("No results found.")
        return
    
    for i, r in enumerate(results, 1):
        print(f"\n{'='*70}")
        print(f"Result {i} (score: {r.score:.4f})")
        print(f"Source: {r.chunk.source}")
        print(f"Entities: {r.chunk.metadata.get('methods', [])}")
        print(f"{'='*70}")
        print(r.chunk.content[:500])
        if len(r.chunk.content) > 500:
            print("...")


def cmd_delete(args):
    """删除指定文档"""
    retriever = get_retriever()
    success = retriever.delete_document(args.doc_id)
    print(f"Deleted: {success}" if success else "Not found")


def cmd_rebuild(args):
    """重建索引"""
    cfg = get_config()
    db_path = Path(cfg.chroma_path)
    
    if db_path.exists():
        import shutil
        shutil.rmtree(db_path)
        print(f"Cleared {db_path}")
    
    print("Rebuild complete. Use 'import-dir' to re-import documents.")


def cmd_stats(args):
    """显示知识库统计信息"""
    retriever = get_retriever()
    docs = retriever.list_documents()
    print(f"Documents: {len(docs)}")
    print(f"Database path: {get_config().chroma_path}")


def main():
    parser = argparse.ArgumentParser(description="NIR Knowledge Base CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # add
    p_add = subparsers.add_parser("add", help="Add a single document")
    p_add.add_argument("file_path", help="Path to document")
    p_add.add_argument("--title", help="Document title")
    p_add.add_argument("--year", type=int, help="Publication year")
    p_add.add_argument("--authors", nargs="*", help="Author names")
    p_add.set_defaults(func=cmd_add)
    
    # import-dir
    p_import = subparsers.add_parser("import-dir", help="Import all documents from a directory")
    p_import.add_argument("directory", help="Directory path")
    p_import.set_defaults(func=cmd_import_dir)
    
    # list
    p_list = subparsers.add_parser("list", help="List all documents")
    p_list.set_defaults(func=cmd_list)
    
    # search
    p_search = subparsers.add_parser("search", help="Search knowledge base")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search.set_defaults(func=cmd_search)
    
    # delete
    p_del = subparsers.add_parser("delete", help="Delete a document")
    p_del.add_argument("doc_id", help="Document ID")
    p_del.set_defaults(func=cmd_delete)
    
    # rebuild
    p_rebuild = subparsers.add_parser("rebuild", help="Rebuild index")
    p_rebuild.set_defaults(func=cmd_rebuild)
    
    # stats
    p_stats = subparsers.add_parser("stats", help="Show statistics")
    p_stats.set_defaults(func=cmd_stats)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

### 3.6 Agent 工具注册 `tools.py` 中新增

```python
# 在 tools.py 中添加

@tool("nir_search_knowledge", parse_docstring=True)
def nir_search_knowledge_tool(
    runtime: Runtime,
    query: str,
    top_k: int = 5,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> str:
    """Search the NIR knowledge base for relevant paper sections.

    Use this when you need domain-specific guidance from research papers,
    such as: which preprocessing method to choose for a specific sample type,
    typical R2 ranges for a given domain, or comparison between methods.

    Args:
        query: Natural language query (e.g., "SNV vs MSC for soil data")
        top_k: Number of results to return (default 5)

    Returns:
        JSON with matching paper sections, source, and entity metadata.
    """
    try:
        from nir_core.knowledge.config import get_retriever
        
        retriever = get_retriever()
        results = retriever.search(query, top_k=top_k)
        
        if not results:
            return _ok({"results": [], "message": "No matching documents found"})
        
        output = []
        for r in results:
            output.append({
                "content": r.chunk.content[:1000],
                "source": r.chunk.source,
                "score": round(r.score, 4),
                "entities": {
                    "methods": r.chunk.metadata.get("methods", []),
                    "models": r.chunk.metadata.get("models", []),
                    "datasets": r.chunk.metadata.get("datasets", []),
                },
                "related_entities": r.related_entities,
            })
        
        return _ok({
            "results": output,
            "count": len(output),
            "query": query,
        })
    except Exception as exc:
        return _error(f"Knowledge search failed: {exc}")
```

### 3.7 SKILL.md 更新

在 `nir-coordinator/SKILL.md` 的知识检索规则中添加：

```markdown
## ★ 知识库语义检索

当精简规则（docs/*.md）不足以指导决策时，调用 `nir_search_knowledge` 搜索论文库：

```
nir_search_knowledge(query="SNV vs MSC soil data scatter correction", top_k=3)
```

使用时机：
- 用户数据类型不在 soil-tips / food-tips 覆盖范围内
- 需要确认某预处理方法在特定场景下的效果
- 需要对比两种方法的优劣
- 需要查找特定 R² 范围是否合理

返回结果包含：论文段落 + 来源 + 实体（方法/模型/数据集）。
```

### 3.8 config.yaml 工具注册

在 `config.yaml` 的 bash 工具组中添加 `nir_search_knowledge`：

```yaml
# config.yaml (第 885-920 行附近)
- name: "nir_search_knowledge"
  description: "Search NIR knowledge base for relevant paper sections"
```

## 4. 依赖清单

### 4.1 新增依赖

```toml
# pyproject.toml (nir_core)
[project]
dependencies = [
    # ... 现有依赖 ...
    "chromadb>=0.5.0",              # 本地向量数据库
    "sentence-transformers>=2.0",   # 本地嵌入模型
    "pypdf>=4.0",                   # PDF 解析
    "python-docx>=1.0",            # Word 解析
    "markdownify>=0.11",            # HTML → Markdown
    "pandas>=2.0",                  # CSV → Markdown 表格 (已有)
    "langchain-community>=0.0.20",  # PyPDFLoader
]
```

### 4.2 嵌入模型

使用 `all-MiniLM-L6-v2`（384 维，模型大小 ~90MB）：
- 首次使用时自动从 HuggingFace 下载
- 纯本地运行，不需要 API key
- 中英文均可处理（中文效果稍弱，但 NIR 论文多为英文）

如需更优中文检索，可替换为 `BAAI/bge-small-zh-v1.5`（512 维，~100MB）。

## 5. 测试计划

### 5.1 单元测试

```
nir_core/tests/test_knowledge/
├── test_parser.py          # 文档解析测试
│   ├── test_parse_pdf
│   ├── test_parse_docx
│   ├── test_parse_txt
│   ├── test_parse_html
│   ├── test_parse_csv
│   └── test_unsupported_extension
├── test_chunker.py         # 分块器测试
│   ├── test_chunk_naive_basic
│   ├── test_chunk_naive_overlap
│   ├── test_chunk_by_section
│   ├── test_chunk_by_section_long_paragraph
│   ├── test_chunk_qa_format
│   └── test_chunk_qa_fallback
├── test_entity_extractor.py  # 实体提取测试
│   ├── test_extract_methods
│   ├── test_extract_models
│   ├── test_extract_datasets
│   ├── test_extract_metrics
│   └── test_no_entities
├── test_vectorstore.py     # ChromaDB 测试
│   ├── test_add_and_search
│   ├── test_search_with_where_filter
│   ├── test_delete_document
│   ├── test_list_documents
│   └── test_search_empty_db
├── test_cli.py             # CLI 测试
│   ├── test_cmd_add
│   ├── test_cmd_import_dir
│   ├── test_cmd_search
│   └── test_cmd_list
└── test_tools.py           # Agent 工具测试
    ├── test_nir_search_knowledge_basic
    ├── test_nir_search_knowledge_empty
    └── test_nir_search_knowledge_error
```

### 5.2 集成测试

```python
def test_end_to_end():
    """端到端：PDF → 解析 → 分块 → 索引 → 检索"""
    # 1. 解析测试 PDF
    markdown = parse_document("test_data/sample_paper.pdf")
    assert len(markdown) > 100
    
    # 2. 分块
    chunks = chunk_document(markdown, source="sample_paper.pdf")
    assert len(chunks) > 0
    
    # 3. 提取实体
    entities = extract_entities(markdown)
    assert "snv" in entities["methods"]
    
    # 4. 索引
    retriever = ChromaDBRetriever(db_path="/tmp/test_chromadb")
    retriever.add_documents(chunks)
    
    # 5. 检索
    results = retriever.search("SNV scatter correction")
    assert len(results) > 0
    assert "snv" in results[0].chunk.metadata.get("methods", [])
```

## 6. 实施顺序

| 步骤 | 文件 | 内容 | 依赖 |
|------|------|------|------|
| 1 | `nir_core/knowledge/__init__.py` | 模块初始化 | 无 |
| 2 | `nir_core/knowledge/base.py` | Chunk, SearchResult, KnowledgeRetriever Protocol | 无 |
| 3 | `nir_core/knowledge/parser.py` | 文档解析器 | 步骤 2 |
| 4 | `nir_core/knowledge/chunker.py` | 智能分块器 | 步骤 2 |
| 5 | `nir_core/knowledge/entity_extractor.py` | 实体提取器 | 无 |
| 6 | `nir_core/knowledge/config.py` | 配置 + get_retriever 工厂 | 步骤 2 |
| 7 | `nir_core/knowledge/vectorstore.py` | ChromaDB 实现 | 步骤 2, 6 |
| 8 | `nir_core/knowledge/cli.py` | CLI 管理工具 | 步骤 3-7 |
| 9 | `tools.py` | 注册 nir_search_knowledge 工具 | 步骤 7 |
| 10 | `config.yaml` | 工具注册 | 步骤 9 |
| 11 | `SKILL.md` | 知识检索规则更新 | 步骤 9 |
| 12 | `nir_core/tests/test_knowledge/` | 单元测试 | 步骤 3-8 |

## 7. 未来扩展：图谱检索（零现有代码修改）

### 7.1 启用条件

当以下需求出现时启用：
- 需要查询"使用 SNV 的论文还用了什么方法"（多跳查询）
- 需要可视化方法之间的关系（如 SNV→airPLS 常一起出现）
- 需要按实体关系过滤（如"在土壤数据集上验证过的方法"）

### 7.2 扩展步骤

| 步骤 | 文件 | 操作 | 现有代码修改 |
|------|------|------|------------|
| 1 | `nir_core/knowledge/neo4j_client.py` | 新增 Neo4j 客户端 | ❌ 新增 |
| 2 | `nir_core/knowledge/graph_retriever.py` | 新增 GraphEnhancedRetriever | ❌ 新增 |
| 3 | `nir_core/knowledge/config.py` | graph_backend 改为 "neo4j" | ✅ 改 1 行 |
| 4 | `nir_core/knowledge/cli.py` | 加 `build-graph` 命令 | ✅ 加 1 个命令 |
| 5 | `tools.py` | 不变 | ❌ |
| 6 | `SKILL.md` | 不变（related_entities 自动填充） | ❌ |

### 7.3 图谱 schema

```cypher
// 节点
(:Paper {doc_id, title, year})
(:Method {name})
(:Model {name})
(:Dataset {name})
(:Metric {name})

// 边
(:Paper)-[:USES_METHOD]->(:Method)
(:Paper)-[:USES_MODEL]->(:Model)
(:Paper)-[:EVALUATED_ON]->(:Dataset)
(:Paper)-[:REPORTS_METRIC]->(:Metric)
(:Method)-[:OFTEN_COMBINED_WITH]->(:Method)  // 共现关系
```

### 7.4 实体数据来源

当前 `entity_extractor.py` 提取的实体已存入 ChromaDB metadata，未来建图时直接读取：

```python
# 未来: neo4j_client.py

def build_graph_from_chromadb(chroma: ChromaDBRetriever):
    """从 ChromaDB 的 metadata 构建 Neo4j 图谱"""
    all_data = chroma._collection.get(include=["metadatas"])
    
    for i, meta in enumerate(all_data["metadatas"]):
        doc_id = meta.get("doc_id", "")
        title = meta.get("title", "")
        
        # 创建 Paper 节点
        graph.run("CREATE (p:Paper {doc_id: $id, title: $title})", id=doc_id, title=title)
        
        # 创建方法节点 + 边
        for method in meta.get("methods", []):
            graph.run("MERGE (m:Method {name: $name}) CREATE (p)-[:USES_METHOD]->(m)", name=method)
        
        # 同理处理 models, datasets, metrics
        ...
```

## 8. 文件清单

```
nir_core/knowledge/
├── __init__.py                  # 模块入口
├── base.py                      # Chunk, SearchResult, KnowledgeRetriever Protocol
├── config.py                    # KnowledgeConfig + get_retriever 工厂
├── parser.py                    # 文档解析器 (PDF/Word/HTML/CSV → Markdown)
├── chunker.py                   # 智能分块器 (naive/section/qa)
├── entity_extractor.py          # NIR 领域实体提取
├── vectorstore.py               # ChromaDB 向量检索实现
├── cli.py                       # CLI 管理工具
└── .chromadb/                   # ChromaDB 持久化数据 (自动生成)

nir_core/tests/test_knowledge/
├── __init__.py
├── test_parser.py
├── test_chunker.py
├── test_entity_extractor.py
├── test_vectorstore.py
└── test_cli.py

# 未来扩展 (零现有代码修改)
nir_core/knowledge/
├── neo4j_client.py              # Neo4j 客户端 (未来新增)
└── graph_retriever.py           # GraphEnhancedRetriever (未来新增)
```

## 9. 与现有知识库的整合

### 9.1 两层知识体系

```
精简规则层 (docs/*.md)          ← 快速参考，决策规则，已存在
    ↓ 如果规则不够
语义检索层 (nir_search_knowledge)  ← 论文段落，深度知识，新增
```

### 9.2 智能体检索流程

```
1. 读取 docs/chemometrics-rules.md (精简规则, ~200 字, 内联在 SKILL.md)
2. 如果需要深度知识 → 调用 nir_search_knowledge(query="...")
3. 获得论文段落 + 实体信息
4. 根据论文指导调整预处理/建模策略
5. 如果返回 related_entities (未来图谱) → 考虑关联方法
```

### 9.3 现有 docs/ 保持不变

`nir-knowledge/docs/` 下的 7 个 Markdown 文件保持原样，继续作为快速参考。新增的 `nir_search_knowledge` 工具作为深度知识补充。
