"""
Z.R.I.C 引擎 — 世界实体注册表模块 (entity.py)
世界实体 CRUD / NPC 情绪状态机 / AI 实体提取 / 实体文本格式化。
由 main.py 通过 app.include_router(entity_router) 挂载。
"""

import json
import json_repair
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
    aliases: list = []

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

        try:
            aliases = json.loads(r["aliases"] or "[]") if "aliases" in r.keys() else []
        except (json.JSONDecodeError, TypeError):
            aliases = []
        alias_tag = f"（又称：{'、'.join(aliases[:3])}）" if aliases else ""
        line = f"- [{tag}] {r['name']}{alias_tag} | 位置：{r['location'] or '不明'} | 状态：{r['status']}"

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
        "SELECT name, location, status, state_desc, aliases FROM world_entities"
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
        try:
            aliases = json.loads(r["aliases"] or "[]")
        except (json.JSONDecodeError, TypeError):
            aliases = []
        alias_str = f"，又称：{'、'.join(aliases)}" if aliases else ""
        existing_lines.append(f"- {r['name']}{alias_str}（{r['location']}，{r['status']}）：{desc}")
    existing_block = "\n".join(existing_lines) or "（暂无已知实体）"

    prompt_sys = (
        "你是跑团世界状态追踪助手。根据本次剧情，提取所有被提及的命名实体（人物/地点/事件）。\n"
        "规则：\n"
        "1. 只提取有专有名字的实体，忽略泛指（'一个路人'、'几个守卫'）\n"
        "2. 若实体已在【已知实体】中（含别名），更新其状态描述和位置，name 字段必须使用已知实体的规范名称\n"
        "3. 【关键】若文本用外貌描述或昵称（如'眼睛男生'、'高个子'、'戴眼镜的'）指代已知实体，"
        "请直接使用已知实体的规范名称，并将该描述词写入 aliases 列表\n"
        "4. status 只能取：active / dead / moved / resolved / pending\n"
        "5. state_desc 用一句话概括当前状态，含关键信息\n"
        "6. aliases 填写本场景中出现的该实体别名/外貌描述词列表（不含规范名称本身），若无则填 []\n"
        "7. 若本次剧情没有涉及任何命名实体，返回 {\"entities\": []}\n\n"
        f"【已知实体（请勿重复创建，应更新；别名也可用于匹配）】\n{existing_block}\n\n"
        "返回严格 JSON（无 markdown）：\n"
        '{"entities": [{"entity_type": "npc|location|event", "name": "规范名称", '
        '"location": "当前所在地点", "status": "active", "state_desc": "一句话状态描述", '
        '"aliases": ["别名1", "别名2"]}]}'
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
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        parsed   = json_repair.loads(resp.choices[0].message.content)
        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 预加载全部实体，供别名三路查找使用（避免循环内多次全表扫描）
        all_rows = conn.execute(
            "SELECT id, name, room_id, state_desc, aliases FROM world_entities"
        ).fetchall()

        def _find_by_alias(target_name: str, new_aliases: list):
            """
            三路别名查找：
            路径1 - 精确名字匹配（已在外层做，这里不重复）
            路径2 - target_name 出现在某实体的 aliases 里 → 找到规范名
            路径3 - new_aliases 里的某项是已有实体的规范名 → 合并并把 target_name 当别名
            返回 (canonical_name, row) 或 (target_name, None)
            """
            # 路径2：别人的 aliases 里有我的名字
            for row in all_rows:
                try:
                    row_aliases = json.loads(row["aliases"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    row_aliases = []
                if target_name in row_aliases:
                    return row["name"], row

            # 路径3：我的 aliases 里有别人的规范名
            for alias in new_aliases:
                for row in all_rows:
                    if row["name"] == alias:
                        # 把 target_name 加入 new_aliases，以便后续写入实体
                        if target_name not in new_aliases:
                            new_aliases.append(target_name)
                        return row["name"], row
            return target_name, None

        for e in entities:
            name = str(e.get("name", "")).strip()[:60]
            if not name:
                continue
            etype    = str(e.get("entity_type", "npc"))[:20]
            location = str(e.get("location",    ""))[:80]
            status   = str(e.get("status",       "active"))[:20]
            new_desc = str(e.get("state_desc",   ""))[:200]
            new_aliases = [
                str(a).strip()[:40] for a in e.get("aliases", [])
                if isinstance(a, str) and str(a).strip() and str(a).strip() != name
            ]

            # 自动匹配 room_id
            matched_room_id = None
            if location:
                room_match = conn.execute(
                    "SELECT id FROM map_rooms WHERE label LIKE ? LIMIT 1",
                    (f"%{location[:20]}%",)
                ).fetchone()
                if room_match:
                    matched_room_id = room_match["id"]

            # 路径1：精确名字匹配
            exists = conn.execute(
                "SELECT id, room_id, state_desc, aliases FROM world_entities WHERE name=?", (name,)
            ).fetchone()

            # 路径2/3：别名匹配（仅在精确匹配失败时触发）
            if not exists:
                name, exists = _find_by_alias(name, new_aliases)
                if exists:
                    _log.info("别名匹配：将「%s」归并到已知实体「%s」", e.get("name"), name)

            if exists:
                # 合并 state_desc（保留 emotion/memory/breakpoint）
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

                # 合并别名：去重保序
                try:
                    old_aliases = json.loads(exists["aliases"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    old_aliases = []
                merged = list(dict.fromkeys(old_aliases + new_aliases))
                merged_aliases = json.dumps(merged, ensure_ascii=False)

                final_room_id = exists["room_id"] if exists["room_id"] else matched_room_id
                conn.execute(
                    "UPDATE world_entities "
                    "SET location=?, status=?, state_desc=?, last_seen_by=?, updated_at=?, room_id=?, aliases=? "
                    "WHERE id=?",
                    (location, status, final_sd, timeline_label, now_str,
                     final_room_id, merged_aliases, exists["id"]),
                )
            else:
                initial_sd = json.dumps({
                    "desc": new_desc,
                    "emotion": {"trust": 0, "fear": 0, "irritation": 0},
                    "memory": []
                }, ensure_ascii=False)
                conn.execute(
                    "INSERT INTO world_entities "
                    "(entity_type, name, location, status, last_seen_by, state_desc, updated_at, room_id, aliases) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (etype, name, location, status, timeline_label,
                     initial_sd, now_str, matched_room_id,
                     json.dumps(new_aliases, ensure_ascii=False)),
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
        new_aliases = [str(a).strip()[:40] for a in req.aliases if str(a).strip()]
        existing = conn.execute(
            "SELECT id, aliases FROM world_entities WHERE name=?", (req.name,)
        ).fetchone()
        if existing:
            try:
                old_aliases = json.loads(existing["aliases"] or "[]")
            except (json.JSONDecodeError, TypeError):
                old_aliases = []
            merged = json.dumps(list(dict.fromkeys(old_aliases + new_aliases)), ensure_ascii=False)
            conn.execute(
                "UPDATE world_entities SET entity_type=?, location=?, status=?, "
                "last_seen_by=?, state_desc=?, updated_at=?, room_id=?, aliases=? WHERE id=?",
                (req.entity_type, req.location, req.status,
                 req.last_seen_by, req.state_desc, now_str, req.room_id, merged, existing["id"])
            )
        else:
            conn.execute(
                "INSERT INTO world_entities (entity_type, name, location, status, "
                "last_seen_by, state_desc, updated_at, room_id, aliases) VALUES (?,?,?,?,?,?,?,?,?)",
                (req.entity_type, req.name, req.location, req.status,
                 req.last_seen_by, req.state_desc, now_str, req.room_id,
                 json.dumps(new_aliases, ensure_ascii=False))
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


# ---------------------------------------------------------
# 情绪状态机：阻尼 + 衰减 + 事件烈度穿透 + 背叛放大
# ---------------------------------------------------------

# (damping, decay_rate)
# damping：同向堆叠阻尼系数（越高越难推到极值）
# decay_rate：每 tick 衰减比例（仅 fear/irritation 正值生效）
_PROFILE: dict[str, tuple[float, float]] = {
    "trust":      (0.7, 0.00),
    "fear":       (0.3, 0.03),
    "irritation": (0.3, 0.08),
}
_HARD_MAX = 100


def apply_emotion_delta(emo: dict, key: str, delta: int) -> dict:
    """
    将 delta 叠加到 emo[key]，应用阻尼和背叛放大。
    返回修改后的 emo（in-place 同时返回）。

    阻尼逻辑：
    - 同向堆叠（delta 与 current 同号，且 current 已离 0 较远）：越接近极值阻尼越强。
      但事件烈度 magnitude = min(|delta|/20, 1.0) 可穿透阻尼，大事件几乎不受阻尼。
    - 反向拉回（delta 与 current 异号）：无阻尼，直接入账。
      trust 反向时额外应用背叛放大：信任越深，背叛伤害越大。
    """
    damping, _ = _PROFILE.get(key, (0.5, 0.0))
    current = emo.get(key, 0)

    same_direction = (delta > 0 and current > 0) or (delta < 0 and current < 0)

    if same_direction:
        # 烈度穿透：|delta| >= 20 时 magnitude=1，完全绕过阻尼
        magnitude = min(abs(delta) / 20.0, 1.0)
        saturation = abs(current) / _HARD_MAX  # 0→1，越满越堵
        effective_damping = damping * saturation * (1.0 - magnitude)
        actual_delta = delta * (1.0 - effective_damping)
    else:
        # 反向通路：无阻尼，trust 额外背叛放大
        if key == "trust" and current > 0 and delta < 0:
            # trust=+100 时系数 3.0，trust=+50 时系数 2.0
            betrayal_multiplier = 1.0 + 2.0 * (current / _HARD_MAX)
            actual_delta = delta * betrayal_multiplier
        else:
            actual_delta = delta

    new_val = current + actual_delta
    emo[key] = max(-_HARD_MAX, min(_HARD_MAX, round(new_val)))
    return emo


def tick_emotion_decay(emo: dict) -> dict:
    """
    对 fear 和 irritation 的正值执行一次自然衰减。
    trust 永不衰减；负值永不衰减；衰减到 <1 直接归零。
    """
    for key in ("fear", "irritation"):
        _, decay_rate = _PROFILE[key]
        val = emo.get(key, 0)
        if val > 0 and decay_rate > 0:
            val *= (1.0 - decay_rate)
            emo[key] = 0 if val < 1 else round(val)
    return emo


if __name__ == "__main__":
    # 自测：验证四个序列
    print("=== 自测：情绪状态机 ===")

    # 序列1：trust 从 0 连续 +10，应逐步减速
    emo = {"trust": 0, "fear": 0, "irritation": 0}
    print("trust 连续 +10×6：", end=" ")
    for _ in range(6):
        apply_emotion_delta(emo, "trust", 10)
        print(emo["trust"], end=" ")
    print()

    # 序列2：trust=+50 收到 +20（救命级），应实入账 ≈+20
    emo = {"trust": 50, "fear": 0, "irritation": 0}
    apply_emotion_delta(emo, "trust", 20)
    print(f"trust=50 + 20(救命级) → {emo['trust']}  (期望 ≈70)")

    # 序列3：trust=+100 收到 -20（重大背叛），应从 +100 砸到 ≈0
    emo = {"trust": 100, "fear": 0, "irritation": 0}
    apply_emotion_delta(emo, "trust", -20)
    print(f"trust=100 + (-20)(背叛) → {emo['trust']}  (期望 ≈0)")

    # 序列4：fear=80 衰减 20 tick，应还剩 ≈44
    emo = {"trust": 0, "fear": 80, "irritation": 0}
    for _ in range(20):
        tick_emotion_decay(emo)
    print(f"fear=80 衰减 20 tick → {emo['fear']}  (期望 ≈44)")

    # 序列5：irritation=80 衰减 10 tick，期望 ≈34；15 tick 接近消散
    emo = {"trust": 0, "fear": 0, "irritation": 80}
    for _ in range(10):
        tick_emotion_decay(emo)
    print(f"irritation=80 衰减 10 tick → {emo['irritation']}  (期望 ≈34)")
    for _ in range(5):
        tick_emotion_decay(emo)
    print(f"irritation=80 衰减 15 tick → {emo['irritation']}  (期望接近消散)")
