"""
Z.R.I.C 引擎 — 多时间线并行系统模块 (timeline.py)
时间线 CRUD / 跳转 / 合并 / 记忆管理。
由 main.py 通过 app.include_router(timeline_router) 挂载。
注意：时间线推演端点留在 main.py（深度依赖 agent 模块）。
"""

import json
import sqlite3
import fastapi
from contextlib import contextmanager
from fastapi import APIRouter
from pydantic import BaseModel
from logger import get_logger

_log = get_logger("timeline")

timeline_router = APIRouter(tags=["多时间线"])

# ---------------------------------------------------------
# 依赖注入
# ---------------------------------------------------------
_db_file: str = ""
_deepseek_client = None
_fn_tl_append_memory = None  # memory.tl_append_memory
_MEMORY_SUMMARY_LIMIT = 300


def configure_timeline(db_file: str, deepseek_client,
                       fn_tl_append_memory=None,
                       memory_summary_limit: int = 300):
    """由 main.py 启动时调用，注入依赖。"""
    global _db_file, _deepseek_client, _fn_tl_append_memory, _MEMORY_SUMMARY_LIMIT
    _db_file = db_file
    _deepseek_client = deepseek_client
    _fn_tl_append_memory = fn_tl_append_memory
    _MEMORY_SUMMARY_LIMIT = memory_summary_limit


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
# Pydantic 数据模型
# ---------------------------------------------------------
class TimelineCreateRequest(BaseModel):
    label: str = "时间线"
    color: str = "#5b9cf5"
    char_ids: str = ""
    current_node_id: int = 0

class TimelineUpdateRequest(BaseModel):
    label: str
    color: str
    char_ids: str
    status: str = "active"

class TimelineJumpRequest(BaseModel):
    node_id: int

class TimelineMergeRequest(BaseModel):
    source_id: int
    target_id: int

class StringContentRequest(BaseModel):
    content: str


# ---------------------------------------------------------
# REST API 端点
# ---------------------------------------------------------
@timeline_router.get("/api/timelines")
def get_timelines():
    with safe_db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM timelines ORDER BY id").fetchall()]
        all_chars = {c["id"]: dict(c) for c in conn.execute("SELECT * FROM characters").fetchall()}
    for tl in rows:
        ids = [int(x) for x in tl["char_ids"].split(",") if x.strip().isdigit()]
        tl["characters"] = [all_chars[i] for i in ids if i in all_chars]
    return {"status": "success", "timelines": rows}


@timeline_router.post("/api/timelines")
def create_timeline(req: TimelineCreateRequest):
    with safe_db() as conn:
        node_id = req.current_node_id if req.current_node_id else None
        if node_id and not conn.execute("SELECT id FROM nodes WHERE id=?", (node_id,)).fetchone():
            raise fastapi.HTTPException(status_code=400, detail="指定的初始节点不存在")
        cur = conn.execute(
            "INSERT INTO timelines (label, color, current_node_id, memory, char_ids, status) VALUES (?,?,?,?,?,?)",
            (req.label[:40], req.color[:20], node_id,
             f"【时间线「{req.label}」记忆已初始化】\n", req.char_ids, "active")
        )
        new_id = cur.lastrowid
        conn.commit()
    return {"status": "success", "id": new_id}


@timeline_router.put("/api/timelines/{tl_id}")
def update_timeline(tl_id: int, req: TimelineUpdateRequest):
    with safe_db() as conn:
        conn.execute(
            "UPDATE timelines SET label=?, color=?, char_ids=?, status=? WHERE id=?",
            (req.label[:40], req.color[:20], req.char_ids, req.status, tl_id)
        )
        conn.commit()
    return {"status": "success"}


@timeline_router.delete("/api/timelines/{tl_id}")
def delete_timeline(tl_id: int):
    with safe_db() as conn:
        conn.execute("DELETE FROM timelines WHERE id=?", (tl_id,))
        conn.commit()
    return {"status": "success"}


@timeline_router.post("/api/timelines/{tl_id}/jump")
def timeline_jump(tl_id: int, req: TimelineJumpRequest):
    with safe_db() as conn:
        node = conn.execute("SELECT * FROM nodes WHERE id=?", (req.node_id,)).fetchone()
        if not node:
            raise fastapi.HTTPException(status_code=400, detail="节点不存在")
        conn.execute("UPDATE timelines SET current_node_id=? WHERE id=?", (req.node_id, tl_id))
        if _fn_tl_append_memory:
            _fn_tl_append_memory(conn, tl_id, f"时间线跳转至场景[{node['name']}]")
        conn.commit()
    return {"status": "success"}


@timeline_router.get("/api/timelines/{tl_id}/memory")
def get_timeline_memory(tl_id: int):
    with safe_db() as conn:
        row = conn.execute("SELECT memory FROM timelines WHERE id=?", (tl_id,)).fetchone()
    return {"status": "success", "content": row["memory"] if row else ""}


@timeline_router.put("/api/timelines/{tl_id}/memory")
def update_timeline_memory(tl_id: int, req: StringContentRequest):
    with safe_db() as conn:
        conn.execute("UPDATE timelines SET memory=? WHERE id=?", (req.content, tl_id))
        conn.commit()
    return {"status": "success"}


@timeline_router.post("/api/timelines/merge")
def merge_timelines(req: TimelineMergeRequest):
    """将 source 时间线的记忆合并入 target，然后标记 source 为 merged。"""
    conn = get_db_connection()
    try:
        src = conn.execute("SELECT * FROM timelines WHERE id=?", (req.source_id,)).fetchone()
        tgt = conn.execute("SELECT * FROM timelines WHERE id=?", (req.target_id,)).fetchone()
        if not src or not tgt:
            raise fastapi.HTTPException(status_code=400, detail="时间线不存在")

        merged_mem = tgt["memory"] or ""
        src_mem    = src["memory"]  or ""
        try:
            resp = _deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": (
                        "你是跑团GM助手。以下是两条分叉时间线的记忆流，请将它们合并为一段连贯的剧情摘要，"
                        "标注哪些事件发生在哪条支线，保留所有关键事件、人名、物品，"
                        f"控制在{_MEMORY_SUMMARY_LIMIT * 2}字以内，用第三人称叙述。"
                    )},
                    {"role": "user", "content":
                        f"【主线记忆】\n{merged_mem}\n\n【合并记忆（来自支线「{src['label']}」）】\n{src_mem}"}
                ],
                temperature=0.3, max_tokens=800,
            )
            merged_text = resp.choices[0].message.content.strip()
        except Exception as e:
            _log.warning("时间线合并 AI 调用失败，降级为拼接: %s", e)
            merged_text = merged_mem + f"\n\n【并入支线「{src['label']}」记忆】\n" + src_mem

        src_ids = set(x for x in src["char_ids"].split(",") if x.strip())
        tgt_ids = set(x for x in tgt["char_ids"].split(",") if x.strip())
        merged_ids = ",".join(sorted(src_ids | tgt_ids))

        conn.execute("UPDATE timelines SET memory=?, char_ids=? WHERE id=?",
                     (f"【时间线汇合摘要】\n{merged_text}", merged_ids, req.target_id))
        conn.execute("UPDATE timelines SET status='merged' WHERE id=?", (req.source_id,))
        conn.commit()
        return {"status": "success", "merged_memory": merged_text}
    finally:
        conn.close()


@timeline_router.put("/api/timelines/{tl_id}/room")
def set_timeline_room(tl_id: int, room_id: int | None = None):
    """更新时间线当前所在房间。"""
    with safe_db() as conn:
        conn.execute("UPDATE timelines SET current_room_id=? WHERE id=?", (room_id, tl_id))
        conn.commit()
    return {"status": "success"}
