"""
Z.R.I.C 引擎 — 世界实体注册表模块 (entity.py)
世界实体 CRUD / NPC 情绪状态机 / AI 实体提取 / 实体文本格式化。
由 main.py 通过 app.include_router(entity_router) 挂载。
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from logger import get_logger

_log = get_logger("entity")

entity_router = APIRouter(tags=["世界实体"])

# ---------------------------------------------------------
# 依赖注入
# ---------------------------------------------------------
_db_file: str = ""
_deepseek_client = None


def configure_entity(db_file: str, deepseek_client):
    """由 main.py 启动时调用，注入依赖。"""
    global _db_file, _deepseek_client
    _db_file = db_file
    _deepseek_client = deepseek_client


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
class WorldEntityUpsertRequest(BaseModel):
    entity_type: str = "npc"
    name: str
    location: str = ""
    status: str = "active"
    last_seen_by: str = ""
    state_desc: str = ""
    room_id: int | None = None

class UpdatePersonaRequest(BaseModel):
    desc: str = ""
    emotion: dict = {}
    breakpoint: dict = {}
    memory: list = []


# ---------------------------------------------------------
# 核心函数：实体文本格式化（供 AI 推演上下文注入）
# ---------------------------------------------------------
def get_world_entities_text(conn, *search_texts) -> str:
    """
    从 world_entities 表提取与当前场景相关的实体快照，格式化后供 AI 注入。
    支持情绪状态机、NPC 记忆、压力破绽预警。
    """
    rows = conn.execute(
        "SELECT * FROM world_entities ORDER BY updated_at DESC"
    ).fetchall()
    if not rows:
        return ""

    # 分离活跃实体和未登场实体
    active_rows = [r for r in rows if r["status"] != "pending"]
    pending_rows = [r for r in rows if r["status"] == "pending"]

    words = [w for w in " ".join(search_texts).lower().split() if len(w) > 1]
    relevant = [
        r for r in active_rows
        if not words or any(
            w in f"{r['name']} {r['location']} {r['state_desc']}".lower()
            for w in words
        )
    ] or list(active_rows)[:12]

    type_label = {"npc": "人物", "location": "地点", "event": "事件"}
    lines = []
    for r in relevant[:12]:
        tag  = type_label.get(r["entity_type"], r["entity_type"])
        seen = f"（最后接触方：{r['last_seen_by']}）" if r["last_seen_by"] else ""

        raw_sd = r["state_desc"] or ""
        if raw_sd.strip().startswith("{"):
            try:
                sd = json.loads(raw_sd)
            except json.JSONDecodeError:
                sd = {"desc": raw_sd}
        else:
            sd = {"desc": raw_sd}

        desc = sd.get("desc", raw_sd) or ""
        emo = sd.get("emotion", {})
        memories = sd.get("memory", [])
        breakpoint_cfg = sd.get("breakpoint", {})

        line = f"- [{tag}] {r['name']} | 位置：{r['location'] or '不明'} | 状态：{r['status']}"

        if r["entity_type"] == "npc" and emo:
            trust = emo.get("trust", 0)
            fear = emo.get("fear", 0)
            irritation = emo.get("irritation", 0)
            if trust or fear or irritation:
                emo_parts = []
                if trust: emo_parts.append(f"信任:{trust}")
                if fear: emo_parts.append(f"恐惧:{fear}")
                if irritation: emo_parts.append(f"烦躁:{irritation}")
                line += f" | 情绪({', '.join(emo_parts)})"

            bp_threshold = breakpoint_cfg.get("threshold", 70)
            bp_field = breakpoint_cfg.get("trigger_field", "irritation")
            bp_reaction = breakpoint_cfg.get("reaction", "")
            bp_value = emo.get(bp_field, 0)
            if bp_value >= bp_threshold and bp_reaction:
                line += f"\n  ⚠️ 【压力破防中！】{r['name']}的{bp_field}({bp_value})已超过阈值({bp_threshold})→破防反应：{bp_reaction}"
            elif bp_value >= bp_threshold * 0.8 and bp_reaction:
                line += f"\n  ⚡ 【即将破防】{r['name']}的{bp_field}已达{bp_value}（阈值{bp_threshold}），情绪明显不稳"

        if desc:
            line += f" | {desc}"
        line += seen

        if memories and r["entity_type"] == "npc":
            recent = memories[-3:]
            line += "\n  记忆：" + " / ".join(recent)

        lines.append(line)

    # 未登场实体：仅注入名字和存在性，不暴露详情
    if pending_rows:
        type_label_p = {"npc": "人物", "location": "地点", "event": "事件"}
        for r in pending_rows[:8]:
            tag = type_label_p.get(r["entity_type"], r["entity_type"])
            lines.append(f"- [{tag}] {r['name']} | 状态：尚未登场（仅知其存在，详细信息待揭晓）")

    return "\n".join(lines)


# ---------------------------------------------------------
# 核心函数：AI 实体提取（推演后自动调用）
# ---------------------------------------------------------
def ai_extract_and_upsert_entities(
    conn, scene_name: str, content: str,
    player_action: str, ai_branches_text: str,
    timeline_label: str
):
    """
    推演完成后，用 AI 从场景+结果中提取涉及的命名实体并写入/更新注册表。
    写入/更新时保留已有的 emotion 和 memory 数据。
    """
    existing = conn.execute(
        "SELECT name, location, status, state_desc FROM world_entities"
    ).fetchall()
    existing_lines = []
    for r in existing[:20]:
        raw = r["state_desc"] or ""
        if raw.strip().startswith("{"):
            try:
                desc = json.loads(raw).get("desc", raw)
            except json.JSONDecodeError:
                desc = raw
        else:
            desc = raw
        existing_lines.append(f"- {r['name']}（{r['location']}，{r['status']}）：{desc}")
    existing_block = "\n".join(existing_lines) or "（暂无已知实体）"

    prompt_sys = (
        "你是跑团世界状态追踪助手。根据本次剧情，提取所有被提及的命名实体（人物/地点/事件）。\n"
        "规则：\n"
        "1. 只提取有专有名字的实体，忽略泛指（'一个路人'、'几个守卫'）\n"
        "2. 若实体已在【已知实体】中，更新其状态描述和位置\n"
        "3. status 只能取：active / dead / moved / resolved / pending\n"
        "4. state_desc 用一句话概括当前状态，含关键信息（比如'刚与A小队交谈过，知道密道入口'）\n"
        "5. 若本次剧情没有涉及任何命名实体，返回 {\"entities\": []}\n\n"
        f"【已知实体（请勿重复创建，应更新）】\n{existing_block}\n\n"
        "返回严格 JSON（无 markdown）：\n"
        '{"entities": [{"entity_type": "npc|location|event", "name": "名字", '
        '"location": "当前所在地点", "status": "active", "state_desc": "一句话状态描述"}]}'
    )
    prompt_user = (
        f"场景名：{scene_name}\n"
        f"场景内容：{content[:250]}\n"
        f"玩家行动：{player_action}\n"
        f"推演结果摘要：{ai_branches_text[:350]}"
    )

    try:
        resp = _deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user",   "content": prompt_user},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        parsed   = json.loads(resp.choices[0].message.content)
        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        for e in entities:
            name = str(e.get("name", "")).strip()[:60]
            if not name:
                continue
            etype      = str(e.get("entity_type", "npc"))[:20]
            location   = str(e.get("location",    ""))[:80]
            status     = str(e.get("status",       "active"))[:20]
            new_desc   = str(e.get("state_desc",   ""))[:200]

            # 自动匹配 room_id
            matched_room_id = None
            if location:
                room_match = conn.execute(
                    "SELECT id FROM map_rooms WHERE label LIKE ? LIMIT 1",
                    (f"%{location[:20]}%",)
                ).fetchone()
                if room_match:
                    matched_room_id = room_match["id"]

            exists = conn.execute(
                "SELECT id, room_id, state_desc FROM world_entities WHERE name=?", (name,)
            ).fetchone()
            if exists:
                old_raw = exists["state_desc"] or ""
                if old_raw.strip().startswith("{"):
                    try:
                        old_sd = json.loads(old_raw)
                    except json.JSONDecodeError:
                        old_sd = {"desc": old_raw}
                else:
                    old_sd = {"desc": old_raw}
                old_sd["desc"] = new_desc
                final_sd = json.dumps(old_sd, ensure_ascii=False)

                final_room_id = exists["room_id"] if exists["room_id"] else matched_room_id
                conn.execute(
                    "UPDATE world_entities "
                    "SET location=?, status=?, state_desc=?, last_seen_by=?, updated_at=?, room_id=? "
                    "WHERE id=?",
                    (location, status, final_sd, timeline_label, now_str,
                     final_room_id, exists["id"]),
                )
            else:
                initial_sd = json.dumps({
                    "desc": new_desc,
                    "emotion": {"trust": 0, "fear": 0, "irritation": 0},
                    "memory": []
                }, ensure_ascii=False)
                conn.execute(
                    "INSERT INTO world_entities "
                    "(entity_type, name, location, status, last_seen_by, state_desc, updated_at, room_id) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (etype, name, location, status, timeline_label,
                     initial_sd, now_str, matched_room_id),
                )
        conn.commit()
    except Exception as e:
        _log.warning("实体提取失败（已降级跳过）: %s", e, exc_info=True)


# ---------------------------------------------------------
# REST API 端点
# ---------------------------------------------------------
@entity_router.get("/api/world-entities")
def list_world_entities():
    with safe_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM world_entities ORDER BY entity_type, updated_at DESC"
        ).fetchall()]
    for r in rows:
        r.setdefault("room_id", None)
    return {"status": "success", "entities": rows}


@entity_router.post("/api/world-entities")
def upsert_world_entity(req: WorldEntityUpsertRequest):
    with safe_db() as conn:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        existing = conn.execute(
            "SELECT id FROM world_entities WHERE name=?", (req.name,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE world_entities SET entity_type=?, location=?, status=?, "
                "last_seen_by=?, state_desc=?, updated_at=?, room_id=? WHERE id=?",
                (req.entity_type, req.location, req.status,
                 req.last_seen_by, req.state_desc, now_str, req.room_id, existing["id"])
            )
        else:
            conn.execute(
                "INSERT INTO world_entities (entity_type, name, location, status, "
                "last_seen_by, state_desc, updated_at, room_id) VALUES (?,?,?,?,?,?,?,?)",
                (req.entity_type, req.name, req.location, req.status,
                 req.last_seen_by, req.state_desc, now_str, req.room_id)
            )
        conn.commit()
    return {"status": "success"}


@entity_router.delete("/api/world-entities/{entity_id}")
def delete_world_entity(entity_id: int):
    with safe_db() as conn:
        conn.execute("DELETE FROM world_entities WHERE id=?", (entity_id,))
        conn.commit()
    return {"status": "success"}


@entity_router.put("/api/world-entities/{entity_id}/room")
def set_entity_room(entity_id: int, room_id: int | None = None):
    """将世界实体绑定到地图房间。"""
    with safe_db() as conn:
        conn.execute("UPDATE world_entities SET room_id=? WHERE id=?", (room_id, entity_id))
        conn.commit()
    return {"status": "success"}


@entity_router.get("/api/world-entities/{entity_id}/persona")
def get_entity_persona(entity_id: int):
    """获取 NPC 的完整 persona 数据。"""
    with safe_db() as conn:
        row = conn.execute("SELECT * FROM world_entities WHERE id=?", (entity_id,)).fetchone()
    if not row:
        return {"status": "error", "message": "实体不存在"}
    raw = row["state_desc"] or ""
    if raw.strip().startswith("{"):
        try:
            sd = json.loads(raw)
        except json.JSONDecodeError:
            sd = {"desc": raw}
    else:
        sd = {"desc": raw}
    sd.setdefault("emotion", {"trust": 0, "fear": 0, "irritation": 0})
    sd.setdefault("memory", [])
    sd.setdefault("breakpoint", {"threshold": 70, "trigger_field": "irritation", "reaction": ""})
    return {"status": "success", "name": row["name"], "entity_id": entity_id, "persona": sd}


@entity_router.put("/api/world-entities/{entity_id}/persona")
def update_entity_persona(entity_id: int, req: UpdatePersonaRequest):
    """更新 NPC 的 persona 数据。"""
    with safe_db() as conn:
        row = conn.execute("SELECT state_desc FROM world_entities WHERE id=?", (entity_id,)).fetchone()
        if not row:
            return {"status": "error", "message": "实体不存在"}
        raw = row["state_desc"] or ""
        if raw.strip().startswith("{"):
            try:
                sd = json.loads(raw)
            except json.JSONDecodeError:
                sd = {"desc": raw}
        else:
            sd = {"desc": raw}
        if req.desc is not None:
            sd["desc"] = req.desc
        if req.emotion:
            emo = sd.get("emotion", {"trust": 0, "fear": 0, "irritation": 0})
            for k in ("trust", "fear", "irritation"):
                if k in req.emotion:
                    emo[k] = max(-100, min(100, int(req.emotion[k])))
            sd["emotion"] = emo
        if req.breakpoint:
            sd["breakpoint"] = {
                "threshold": int(req.breakpoint.get("threshold", 70)),
                "trigger_field": str(req.breakpoint.get("trigger_field", "irritation")),
                "reaction": str(req.breakpoint.get("reaction", ""))[:200],
            }
        if req.memory is not None:
            sd["memory"] = [str(m)[:80] for m in req.memory][:10]
        conn.execute("UPDATE world_entities SET state_desc=? WHERE id=?",
                     (json.dumps(sd, ensure_ascii=False), entity_id))
        conn.commit()
    return {"status": "success"}
