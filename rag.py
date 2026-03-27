"""
Z.R.I.C 引擎 — RAG 知识库模块 (rag.py)
从 main.py 中抽离的 RAG 相关：切片、embedding、检索、REST API。
由 main.py 通过 app.include_router(rag_router) 挂载。
"""

import math
import json
import time
import sqlite3
import threading
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from logger import get_logger

_log = get_logger("rag")

# numpy 优先，不可用时降级为纯 Python 模式
try:
    import numpy as np
    _HAS_NUMPY = True
    _log.info("numpy 可用，RAG 检索将使用向量化加速")
except ImportError:
    _HAS_NUMPY = False
    _log.info("numpy 不可用，RAG 检索使用纯 Python 模式（pip install numpy 可加速）")

rag_router = APIRouter(tags=["RAG 知识库"])

# ---------------------------------------------------------
# 数据库连接（由 main.py 注入）
# ---------------------------------------------------------
_db_file: str = ""
_embed_client = None  # OpenAI-compatible embedding client

# RAG 参数
RAG_CHUNK_SIZE    = 400
RAG_CHUNK_OVERLAP = 80
RAG_TOP_K         = 6

# 硅基流动 Embedding 配置
SILICONFLOW_EMB_MODEL = "BAAI/bge-m3"


def configure_rag(db_file: str, siliconflow_api_key: str,
                  siliconflow_base_url: str = "https://api.siliconflow.cn/v1"):
    """由 main.py 启动时调用，注入依赖。"""
    global _db_file, _embed_client
    _db_file = db_file
    if siliconflow_api_key:
        from openai import OpenAI as _OpenAI
        _embed_client = _OpenAI(api_key=siliconflow_api_key, base_url=siliconflow_base_url)


def get_db_connection():
    conn = sqlite3.connect(_db_file, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


# ---------------------------------------------------------
# 数据库表初始化
# ---------------------------------------------------------
def init_rag_tables():
    """创建 RAG 相关表（幂等）。由 main.py 的 init_db() 调用。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS rag_documents (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        title      TEXT NOT NULL,
        source     TEXT NOT NULL DEFAULT '',
        chunk_size INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS rag_chunks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id      INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL DEFAULT 0,
        chunk_text  TEXT NOT NULL,
        embedding   TEXT NOT NULL DEFAULT '[]',
        FOREIGN KEY(doc_id) REFERENCES rag_documents(id) ON DELETE CASCADE
    )''')
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# Pydantic 模型
# ---------------------------------------------------------
class RagIngestRequest(BaseModel):
    title:  str
    source: str = ""
    text:   str


class RagSearchRequest(BaseModel):
    scene_name: str
    content: str


# ---------------------------------------------------------
# 核心函数：切片、embedding、检索
# ---------------------------------------------------------
import re

def chunk_text(text: str, max_size: int = 600) -> list[str]:
    """
    语义切分：
    1. 优先按双换行符（\n\n）切分自然段落
    2. 如果文档没有双换行，fallback 到单换行（\n）切分
    3. 段落超过 max_size 时，按中文/英文句号切分
    4. 保留标点符号，不丢失语义边界
    """
    text = text.strip()
    if not text:
        return []

    # 1. 优先按 \n\n 切分
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # 2. 如果只得到一个段落（文档可能没有 \n\n），fallback 到 \n
    if len(paragraphs) <= 1 and len(text) > max_size:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    chunks = []
    for p in paragraphs:
        if len(p) <= max_size:
            chunks.append(p)
        else:
            # 3. 按句号/问号/叹号切分（保留标点）
            sentences = re.split(r'(?<=[。！？!?])', p)
            current = ""
            for s in sentences:
                if not s.strip():
                    continue
                if len(current) + len(s) <= max_size:
                    current += s
                else:
                    if current:
                        chunks.append(current.strip())
                    current = s
            if current:
                chunks.append(current.strip())

    return chunks


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    调用硅基流动 BAAI/bge-m3 获取 embedding 向量。
    带 rate-limit 保护：每批之间 sleep 0.3s 防止 429。
    """
    if not texts or not _embed_client:
        return []

    all_embeddings: list[list[float]] = []
    BATCH = 16  # 降低批量大小以减少 429 风险

    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        retry = 0
        while retry < 3:
            try:
                resp = _embed_client.embeddings.create(
                    model=SILICONFLOW_EMB_MODEL,
                    input=batch,
                )
                items = sorted(resp.data, key=lambda x: x.index)
                all_embeddings.extend(item.embedding for item in items)
                break
            except Exception as e:
                retry += 1
                if "429" in str(e) or "rate" in str(e).lower():
                    _log.debug("Embedding 429 限流，第 %d 次重试", retry)
                    time.sleep(1.0 * retry)  # 指数退避
                elif retry >= 3:
                    # 最终失败：用空向量占位
                    _log.warning("Embedding 批次最终失败（3次重试耗尽）: %s", e)
                    all_embeddings.extend([] for _ in batch)
                    break
                else:
                    _log.debug("Embedding 调用异常（第 %d 次重试）: %s", retry, e)
                    time.sleep(0.5)

        # 批间冷却：防止并发速率限制
        if i + BATCH < len(texts):
            time.sleep(0.3)

    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------
# 向量缓存（内存加速检索，避免每次全表 json.loads）
# ---------------------------------------------------------
class _VectorCache:
    """
    将 rag_chunks 表中的 embedding 预加载到内存。
    - numpy 可用时：存为 float32 矩阵，检索用矩阵乘法（毫秒级）
    - numpy 不可用时：存为 list[list[float]]，逐行计算（兼容模式）
    线程安全：通过 RLock 保护读写。
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._chunk_ids: list[int] = []       # chunk_id 列表，与矩阵行一一对应
        self._texts: list[str] = []           # chunk_text
        self._titles: list[str] = []          # document title
        self._doc_ids: list[int] = []         # doc_id（用于按文档删除）
        self._chunk_indexes: list[int] = []   # chunk_index
        self._matrix = None                   # numpy ndarray 或 list[list[float]]
        self._norms = None                    # 预计算的 L2 范数
        self._ready = False

    def reload(self, db_file: str = ""):
        """从数据库全量加载向量到内存。"""
        target_db = db_file or _db_file
        if not target_db:
            return
        try:
            conn = sqlite3.connect(target_db, timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT c.id, c.chunk_text, c.embedding, c.chunk_index, c.doc_id, d.title "
                "FROM rag_chunks c JOIN rag_documents d ON c.doc_id = d.id "
                "ORDER BY c.id"
            ).fetchall()
            conn.close()
        except Exception as e:
            _log.warning("向量缓存加载失败: %s", e)
            return

        ids, texts, titles, doc_ids, chunk_indexes, vectors = [], [], [], [], [], []
        for row in rows:
            try:
                emb = json.loads(row["embedding"])
                if not emb or not isinstance(emb, list):
                    continue
                ids.append(row["id"])
                texts.append(row["chunk_text"])
                titles.append(row["title"])
                doc_ids.append(row["doc_id"])
                chunk_indexes.append(row["chunk_index"])
                vectors.append(emb)
            except (json.JSONDecodeError, TypeError):
                continue

        with self._lock:
            self._chunk_ids = ids
            self._texts = texts
            self._titles = titles
            self._doc_ids = doc_ids
            self._chunk_indexes = chunk_indexes
            if _HAS_NUMPY and vectors:
                mat = np.array(vectors, dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # 避免除零
                self._matrix = mat / norms  # 预归一化，检索时只需 dot
                self._norms = norms
            else:
                self._matrix = vectors  # fallback: list[list[float]]
                self._norms = None
            self._ready = bool(ids)
            _log.info("向量缓存已加载: %d 个有效向量", len(ids))

    def search(self, query_vec: list[float], top_k: int = RAG_TOP_K,
               threshold: float = 0.3) -> list[dict]:
        """
        检索最相关的 top_k 个切片。
        返回 [{"score": float, "chunk_text": str, "title": str, "doc_id": int, "chunk_index": int}, ...]
        """
        with self._lock:
            if not self._ready or not self._chunk_ids:
                return []

            if _HAS_NUMPY and isinstance(self._matrix, np.ndarray):
                # numpy 向量化：query 归一化后矩阵乘法
                qvec = np.array(query_vec, dtype=np.float32)
                qnorm = np.linalg.norm(qvec)
                if qnorm == 0:
                    return []
                qvec = qvec / qnorm
                scores = self._matrix @ qvec  # (N,) cosine similarities
                # 取 top_k（先过滤阈值）
                mask = scores >= threshold
                if not mask.any():
                    return []
                indices = np.where(mask)[0]
                top_indices = indices[np.argsort(scores[indices])[::-1][:top_k]]
                return [
                    {
                        "score": round(float(scores[i]), 4),
                        "chunk_text": self._texts[i],
                        "title": self._titles[i],
                        "doc_id": self._doc_ids[i],
                        "chunk_index": self._chunk_indexes[i],
                    }
                    for i in top_indices
                ]
            else:
                # 纯 Python fallback
                scored = []
                for i, emb in enumerate(self._matrix):
                    score = cosine_similarity(query_vec, emb)
                    if score >= threshold:
                        scored.append((score, i))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [
                    {
                        "score": round(scored[j][0], 4),
                        "chunk_text": self._texts[scored[j][1]],
                        "title": self._titles[scored[j][1]],
                        "doc_id": self._doc_ids[scored[j][1]],
                        "chunk_index": self._chunk_indexes[scored[j][1]],
                    }
                    for j in range(min(top_k, len(scored)))
                ]

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._chunk_ids)


# 全局缓存实例
_vec_cache = _VectorCache()


def refresh_vector_cache():
    """外部调用入口：刷新向量缓存。由 main.py 在加载剧本/重建 embedding 后调用。"""
    _vec_cache.reload()


# ---------------------------------------------------------
# 核心检索函数（使用向量缓存加速）
# ---------------------------------------------------------


# ---------------------------------------------------------
# 混合检索核心（关键词预匹配 + 向量语义检索）
# ---------------------------------------------------------
def _hybrid_retrieve(conn, query_text: str, top_k: int = RAG_TOP_K,
                     vec_threshold: float = 0.3) -> list[dict]:
    """
    混合检索：Sparse（关键词精确匹配）+ Dense（向量语义检索）。
    返回 [{"score": float, "chunk_text": str, "title": str, ...}, ...]
    被 rag_retrieve（prompt 注入）和 rag_search（前端测试）共同调用。
    """
    seen_texts = set()  # 去重用
    exact_results = []
    sparse_quota = max(1, top_k // 2)  # 关键词匹配最多占一半名额

    # ── 1. 关键词预匹配（Sparse）──
    try:
        # 从 world_entities 提取实体名
        entities = conn.execute(
            "SELECT name FROM world_entities WHERE name IS NOT NULL AND name != ''"
        ).fetchall()
        # 从 rag_documents 提取文档标题
        doc_titles = conn.execute(
            "SELECT title FROM rag_documents WHERE title IS NOT NULL AND title != ''"
        ).fetchall()

        # 收集命中的关键词（过滤长度 < 2 的，防止"王"匹配"国王"等误命中）
        hit_keywords = set()
        for row in entities:
            name = row["name"]
            if len(name) >= 2 and name in query_text:
                hit_keywords.add(name)
        for row in doc_titles:
            title = row["title"]
            if len(title) >= 2 and title in query_text:
                hit_keywords.add(title)

        # 对命中关键词执行精确查找
        if hit_keywords:
            for kw in hit_keywords:
                if len(exact_results) >= sparse_quota:
                    break
                rows = conn.execute(
                    "SELECT c.chunk_text, c.chunk_index, c.doc_id, d.title "
                    "FROM rag_chunks c JOIN rag_documents d ON c.doc_id = d.id "
                    "WHERE c.chunk_text LIKE ? LIMIT 2",
                    (f"%{kw}%",)
                ).fetchall()
                for r in rows:
                    if r["chunk_text"] not in seen_texts and len(exact_results) < sparse_quota:
                        exact_results.append({
                            "score": 1.0,
                            "chunk_text": r["chunk_text"],
                            "title": f"[精确]{r['title']}",
                            "doc_id": r["doc_id"],
                            "chunk_index": r["chunk_index"],
                        })
                        seen_texts.add(r["chunk_text"])
    except Exception as e:
        _log.debug("RAG 关键词预匹配异常（已跳过）: %s", e)

    # ── 2. 向量语义检索（Dense）──
    dense_quota = top_k - len(exact_results)
    vec_results = []

    if dense_quota > 0:
        try:
            query_vecs = get_embeddings([query_text])
        except Exception as e:
            _log.warning("RAG 检索 embedding 失败: %s", e)
            query_vecs = []

        if query_vecs and query_vecs[0]:
            query_vec = query_vecs[0]

            if _vec_cache.is_ready:
                # 内存缓存检索（多取一些，去重后截断）
                raw = _vec_cache.search(query_vec, top_k=dense_quota + len(exact_results),
                                         threshold=vec_threshold)
                for r in raw:
                    if r["chunk_text"] not in seen_texts and len(vec_results) < dense_quota:
                        vec_results.append(r)
                        seen_texts.add(r["chunk_text"])
            else:
                # 缓存未就绪：数据库全表扫描
                rows = conn.execute(
                    "SELECT c.id, c.chunk_text, c.embedding, c.chunk_index, c.doc_id, d.title "
                    "FROM rag_chunks c JOIN rag_documents d ON c.doc_id = d.id"
                ).fetchall()
                scored = []
                for row in rows:
                    try:
                        emb = json.loads(row["embedding"])
                        if not emb:
                            continue
                        score = cosine_similarity(query_vec, emb)
                        if score >= vec_threshold and row["chunk_text"] not in seen_texts:
                            scored.append({
                                "score": round(score, 4),
                                "chunk_text": row["chunk_text"],
                                "title": row["title"],
                                "doc_id": row["doc_id"],
                                "chunk_index": row["chunk_index"],
                            })
                    except Exception:
                        continue
                scored.sort(key=lambda x: x["score"], reverse=True)
                vec_results = scored[:dense_quota]

    # ── 3. 合并：精确匹配置顶，向量补充 ──
    return exact_results + vec_results


def rag_retrieve(conn, query_text: str, top_k: int = RAG_TOP_K) -> str:
    """
    检索最相关的 top_k 个切片，格式化为 prompt 注入文本。
    使用混合检索：关键词精确匹配置顶 + 向量语义检索补充。
    """
    results = _hybrid_retrieve(conn, query_text, top_k=top_k)
    if not results:
        return ""
    return "\n\n".join(
        f"[来源：{r['title']}]\n{r['chunk_text']}" for r in results
    )


# ---------------------------------------------------------
# REST API 端点
# ---------------------------------------------------------
@rag_router.get("/api/rag/documents")
def rag_list_documents():
    conn = get_db_connection()
    docs = conn.execute("SELECT * FROM rag_documents ORDER BY created_at DESC").fetchall()
    result = []
    for d in docs:
        chunk_count = conn.execute(
            "SELECT COUNT(*) as c FROM rag_chunks WHERE doc_id=?", (d["id"],)
        ).fetchone()["c"]
        result.append({**dict(d), "chunk_count": chunk_count})
    conn.close()
    return {"status": "success", "documents": result}


@rag_router.post("/api/rag/ingest")
def rag_ingest(req: RagIngestRequest):
    """导入文档：切片 → embedding → 写入数据库。带 rate-limit 保护。"""
    t0 = time.time()

    chunks = chunk_text(req.text)
    if not chunks:
        return {"status": "error", "message": "文本为空，无法切片"}

    conn = get_db_connection()
    cur = conn.execute(
        "INSERT INTO rag_documents (title, source, chunk_size) VALUES (?,?,?)",
        (req.title[:100], req.source[:200], len(chunks))
    )
    doc_id = cur.lastrowid

    all_embeddings = get_embeddings(chunks)

    for idx, chunk in enumerate(chunks):
        emb = all_embeddings[idx] if idx < len(all_embeddings) else []
        conn.execute(
            "INSERT INTO rag_chunks (doc_id, chunk_index, chunk_text, embedding) VALUES (?,?,?,?)",
            (doc_id, idx, chunk, json.dumps(emb))
        )

    conn.commit()
    conn.close()

    # 导入完成后刷新向量缓存
    refresh_vector_cache()

    elapsed = round(time.time() - t0, 1)
    embedded_count = sum(1 for e in all_embeddings if e)
    return {
        "status": "success",
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "embedded": embedded_count,
        "elapsed_sec": elapsed,
    }


@rag_router.post("/api/rag/upload")
async def rag_upload(
    file:   UploadFile = File(...),
    title:  str        = Form(""),
    source: str        = Form(""),
):
    """上传 .txt / .pdf 文件并导入知识库。"""
    t0 = time.time()
    filename = file.filename or "unknown"
    raw = await file.read()

    # ── 提取文本 ──────────────────────────────────────────
    if filename.lower().endswith(".pdf"):
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            return {"status": "error", "message": "PDF 解析需要安装 pypdf：pip install pypdf"}
        except Exception as e:
            return {"status": "error", "message": f"PDF 解析失败：{e}"}
    else:
        # 尝试 UTF-8，fallback GBK
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("gbk")
            except Exception:
                return {"status": "error", "message": "文件编码无法识别，请转为 UTF-8 后重试"}

    text = text.strip()
    if not text:
        return {"status": "error", "message": "文件内容为空"}

    doc_title  = (title.strip()  or filename)[:100]
    doc_source = (source.strip() or filename)[:200]

    chunks = chunk_text(text)
    if not chunks:
        return {"status": "error", "message": "文本切片失败"}

    conn = get_db_connection()
    cur = conn.execute(
        "INSERT INTO rag_documents (title, source, chunk_size) VALUES (?,?,?)",
        (doc_title, doc_source, len(chunks))
    )
    doc_id = cur.lastrowid

    all_embeddings = get_embeddings(chunks)
    for idx, chunk in enumerate(chunks):
        emb = all_embeddings[idx] if idx < len(all_embeddings) else []
        conn.execute(
            "INSERT INTO rag_chunks (doc_id, chunk_index, chunk_text, embedding) VALUES (?,?,?,?)",
            (doc_id, idx, chunk, json.dumps(emb))
        )

    conn.commit()
    conn.close()
    refresh_vector_cache()

    elapsed = round(time.time() - t0, 1)
    embedded_count = sum(1 for e in all_embeddings if e)
    return {
        "status":      "success",
        "doc_id":      doc_id,
        "filename":    filename,
        "char_count":  len(text),
        "chunk_count": len(chunks),
        "embedded":    embedded_count,
        "elapsed_sec": elapsed,
    }


@rag_router.delete("/api/rag/documents/{doc_id}")
def rag_delete_document(doc_id: int):
    conn = get_db_connection()
    conn.execute("DELETE FROM rag_chunks WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM rag_documents WHERE id=?", (doc_id,))
    conn.commit(); conn.close()
    # 删除后刷新缓存
    refresh_vector_cache()
    return {"status": "success"}


@rag_router.post("/api/rag/search")
def rag_search(req: RagSearchRequest):
    """独立检索接口，供前端测试知识库效果。使用混合检索。"""
    query = f"{req.scene_name} {req.content}"
    conn = get_db_connection()
    results = _hybrid_retrieve(conn, query, top_k=RAG_TOP_K * 2, vec_threshold=0.0)
    conn.close()
    return {"status": "success", "results": results}
