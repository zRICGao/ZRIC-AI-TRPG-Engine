"""
Z.R.I.C 引擎 — 三级记忆系统模块 (memory.py)
L1 短期工作区 / 记忆折叠 / L3 淘汰 / REST API。
由 main.py 通过 app.include_router(memory_router) 挂载。
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from logger import get_logger

_log = get_logger("memory")

memory_router = APIRouter(tags=["记忆系统"])

# ---------------------------------------------------------
# 依赖注入（由 main.py 启动时设置）
# ---------------------------------------------------------
_db_file: str = ""
_deepseek_client = None
_get_embeddings = None  # rag.get_embeddings

# 记忆参数
MEMORY_FOLD_THRESHOLD = 4000
MEMORY_RECENT_LINES   = 6
MEMORY_SUMMARY_LIMIT  = 300
L1_MAX_ENTRIES  = 10
L1_TOKEN_BUDGET = 4000


def configure_memory(db_file: str, deepseek_client, fn_get_embeddings=None):
    """由 main.py 启动时调用，注入依赖。"""
    global _db_file, _deepseek_client, _get_embeddings
    _db_file = db_file
    _deepseek_client = deepseek_client
    _get_embeddings = fn_get_embeddings


def get_db_connection():
    conn = sqlite3.connect(_db_file, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


@contextmanager
def safe_db():
    conn = get_db_connection()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------
# L1 短期工作区
# ---------------------------------------------------------
def _l1_append(conn, scene_name: str, player_action: str,
               ai_summary: str, thought_process: str = "",
               entity_updates: str = "", timeline_id: int | None = None):
    """
    向 L1 短期工作区追加一条推演快照。
    timeline_id=None 表示主线；传入 int 则隔离到对应时间线。
    若超过 L1_MAX_ENTRIES，将最旧的同线记录提取出来，交给后台线程淘汰到 L3，主线程不阻塞。
    """
    conn.execute(
        "INSERT INTO memory_l1 (scene_name, player_action, ai_summary, "
        "thought_process, entity_updates, timeline_id) VALUES (?,?,?,?,?,?)",
        (scene_name[:80], player_action[:200], ai_summary[:500],
         thought_process[:800], entity_updates[:500], timeline_id)
    )
    conn.commit()

    # 检查是否需要淘汰旧记录到 L3（按时间线隔离计数）
    count = conn.execute(
        "SELECT COUNT(*) as c FROM memory_l1 WHERE timeline_id IS ?", (timeline_id,)
    ).fetchone()["c"]
    if count > L1_MAX_ENTRIES:
        # 将行数据转为普通 dict，以便跨线程传递
        overflow = [dict(r) for r in conn.execute(
            "SELECT * FROM memory_l1 WHERE timeline_id IS ? ORDER BY id ASC LIMIT ?",
            (timeline_id, count - L1_MAX_ENTRIES,)
        ).fetchall()]
        
        # 【异步改造】：启动后台线程处理 AI 摘要和 Embedding，主线程直接放行
        import threading
        def _async_evict_task(rows_to_evict):
            # 后台线程必须获取自己的数据库连接
            bg_conn = get_db_connection()
            try:
                _l1_evict_to_l3(bg_conn, rows_to_evict)
            except Exception as e:
                print(f"[后台任务] L1->L3 淘汰失败: {e}")
            finally:
                bg_conn.close()

        threading.Thread(target=_async_evict_task, args=(overflow,), daemon=True).start()


def _l1_evict_to_l3(conn, rows):
    """
    将 L1 溢出的记录批量转化为 L3 长期向量记忆。
    流程：拼接为摘要文本 → embedding → 写入 rag_chunks → 删除 L1 原记录。
    如果 embedding 失败，降级为纯文本写入 session_memory。
    """
    if not rows:
        return

    evict_text_parts = []
    evict_ids = []
    for r in rows:
        part = (
            f"[{r['scene_name']}] 玩家行动：{r['player_action']}。"
            f"结果：{r['ai_summary']}"
        )
        if r['entity_updates']:
            part += f" 实体变化：{r['entity_updates']}"
        evict_text_parts.append(part)
        evict_ids.append(r["id"])

    evict_text = "\n".join(evict_text_parts)

    # 用 AI 压缩为精炼摘要
    try:
        resp = _deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    f"你是跑团GM助手。请将以下跑团记录归纳为{MEMORY_SUMMARY_LIMIT}字以内的核心剧情摘要。"
                    "要求：保留人名、地点、关键物品、重要转折、实体状态变化，"
                    "去除冗余细节，用第三人称叙述，不加任何前缀或解释。"
                )},
                {"role": "user", "content": evict_text}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        _log.warning("L1→L3 记忆摘要 AI 调用失败，降级为截断: %s", e)
        summary = evict_text[:MEMORY_SUMMARY_LIMIT]

    # 尝试写入 L3（RAG 向量库）
    try:
        doc_title = f"长期记忆_{datetime.now().strftime('%m%d_%H%M')}"
        cur = conn.execute(
            "INSERT INTO rag_documents (title, source, chunk_size) VALUES (?,?,?)",
            (doc_title, "memory_l3_eviction", 1)
        )
        doc_id = cur.lastrowid

        vecs = _get_embeddings([summary]) if _get_embeddings else []
        emb_json = json.dumps(vecs[0]) if vecs and vecs[0] else "[]"

        conn.execute(
            "INSERT INTO rag_chunks (doc_id, chunk_index, chunk_text, embedding) "
            "VALUES (?,?,?,?)",
            (doc_id, 0, summary, emb_json)
        )
    except Exception as e:
        _log.warning("L1→L3 embedding 写入失败，降级到 session_memory: %s", e)
        row = conn.execute(
            "SELECT value FROM system_state WHERE key='session_memory'"
        ).fetchone()
        old_mem = row["value"] if row else ""
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key,value) VALUES ('session_memory',?)",
            (old_mem + f"\n【长期记忆归档】{summary}",)
        )

    for rid in evict_ids:
        conn.execute("DELETE FROM memory_l1 WHERE id=?", (rid,))
    conn.commit()


def l1_get_working_context(conn, timeline_id: int | None = None) -> str:
    """
    从 L1 工作区提取最近 N 条推演快照，格式化为可注入 system prompt 的文本。
    timeline_id=None 取主线记录；传入 int 只取该时间线记录。
    控制总字符数不超过 L1_TOKEN_BUDGET。
    """
    rows = conn.execute(
        "SELECT * FROM memory_l1 WHERE timeline_id IS ? ORDER BY id DESC LIMIT ?",
        (timeline_id, L1_MAX_ENTRIES,)
    ).fetchall()

    if not rows:
        return ""

    rows = list(reversed(rows))
    lines = []
    total = 0
    for r in rows:
        line = f"- [{r['scene_name']}] 玩家「{r['player_action']}」→ {r['ai_summary']}"
        if r['entity_updates']:
            line += f" | 实体变化：{r['entity_updates']}"
        if total + len(line) > L1_TOKEN_BUDGET:
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines)


# ---------------------------------------------------------
# 记忆折叠 + 追加
# ---------------------------------------------------------
def fold_memory_with_ai(current_mem: str) -> str:
    """
    将过长记忆折叠为「AI摘要 + 最近N行原文」。
    若 AI 调用失败，降级为按行截断。
    """
    lines = [l for l in current_mem.splitlines() if l.strip()]
    recent_lines  = lines[-MEMORY_RECENT_LINES:]
    archive_lines = lines[:-MEMORY_RECENT_LINES]

    if not archive_lines:
        return current_mem

    archive_text = "\n".join(archive_lines)
    recent_text  = "\n".join(recent_lines)

    try:
        resp = _deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    f"你是跑团GM助手。请将以下跑团日志归纳为{MEMORY_SUMMARY_LIMIT}字以内的核心剧情摘要。"
                    "要求：保留人名、地点、关键物品、重要转折，去除冗余细节，用第三人称叙述，不加任何前缀或解释。"
                )},
                {"role": "user", "content": archive_text}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        summary = resp.choices[0].message.content.strip()
        return f"【剧情摘要】{summary}\n【近期详细记录】\n{recent_text}"
    except Exception as e:
        _log.warning("记忆折叠 AI 调用失败，降级为截断: %s", e)
        fallback_lines = lines[-(MEMORY_RECENT_LINES * 3):]
        return "【早期记录已截断】\n" + "\n".join(fallback_lines)


def _async_fold_memory_task(target_type: str, text_to_fold: str, tl_id: int = None):
    """后台专属任务：调用 AI 压缩记忆，然后写回数据库"""
    bg_conn = get_db_connection()
    try:
        # 调用大模型进行折叠（耗时操作，现在在后台发生）
        folded_text = fold_memory_with_ai(text_to_fold)
        
        if target_type == "session":
            bg_conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('session_memory', ?)", (folded_text,))
        elif target_type == "timeline" and tl_id is not None:
            bg_conn.execute("UPDATE timelines SET memory=? WHERE id=?", (folded_text, tl_id))
            
        bg_conn.commit()
    except Exception as e:
        print(f"[后台任务] 记忆 AI 折叠失败: {e}")
    finally:
        bg_conn.close()


def append_to_memory(conn, new_log):
    """追加新主线记忆：立即存库防止丢失，超限则丢给后台折叠"""
    row = conn.execute("SELECT value FROM system_state WHERE key = 'session_memory'").fetchone()
    current_mem = row["value"] if row else ""
    updated_mem = current_mem + f"\n- {new_log}"

    # 1. 立即保存原文（哪怕超长了也先存进去，保证前端可以秒刷）
    conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('session_memory', ?)", (updated_mem,))
    conn.commit()

    # 2. 如果超过阈值，派发后台线程去慢慢折叠
    if len(updated_mem) > MEMORY_FOLD_THRESHOLD:
        import threading
        threading.Thread(target=_async_fold_memory_task, args=("session", updated_mem), daemon=True).start()


# ---------------------------------------------------------
# 时间线记忆追加（供 timeline 模块调用）
# ---------------------------------------------------------
def _tl_append_memory(conn, tl_id: int, log: str):
    """追加时间线记忆：立即存库防止丢失，超限则丢给后台折叠"""
    row = conn.execute("SELECT memory FROM timelines WHERE id=?", (tl_id,)).fetchone()
    if not row: return
    
    current = row["memory"] or ""
    updated = current + f"\n- {log}"
    
    # 1. 立即保存原文
    conn.execute("UPDATE timelines SET memory=? WHERE id=?", (updated, tl_id))
    conn.commit()

    # 2. 如果超过阈值，派发后台线程
    if len(updated) > MEMORY_FOLD_THRESHOLD:
        import threading
        threading.Thread(target=_async_fold_memory_task, args=("timeline", updated, tl_id), daemon=True).start()


# ---------------------------------------------------------
# REST API 端点
# ---------------------------------------------------------
class StringContentRequest(BaseModel):
    content: str


@memory_router.get("/api/game/memory")
def get_memory():
    with safe_db() as conn:
        row = conn.execute("SELECT value FROM system_state WHERE key = 'session_memory'").fetchone()
    return {"status": "success", "content": row["value"] if row else ""}


@memory_router.put("/api/game/memory")
def update_memory(req: StringContentRequest):
    with safe_db() as conn:
        conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('session_memory', ?)", (req.content,))
        conn.commit()
    return {"status": "success"}


@memory_router.get("/api/game/memory-l1")
def get_memory_l1():
    """获取 L1 短期工作区所有条目（前端展示用）"""
    with safe_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM memory_l1 ORDER BY id DESC LIMIT ?", (L1_MAX_ENTRIES,)
        ).fetchall()]
    return {"status": "success", "entries": list(reversed(rows))}


@memory_router.delete("/api/game/memory-l1")
def clear_memory_l1():
    """清空 L1 短期工作区"""
    with safe_db() as conn:
        conn.execute("DELETE FROM memory_l1")
        conn.commit()
    return {"status": "success"}
