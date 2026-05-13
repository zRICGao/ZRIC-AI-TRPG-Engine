"""
Z.R.I.C 引擎 — 触发器系统模块 (trigger.py)

支持功能：
- DAG 条件树：{"op": "and|or|not", "children": [...]}，旧 AND 列表自动迁移
- fire_count / cooldown / prerequisite_trigger_ids / exclude_trigger_ids
- trigger_judgements 表记录 AI 推理过程（含 prompt_hash 去重缓存）
"""

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime

import fastapi
from fastapi import APIRouter
from pydantic import BaseModel

from logger import get_logger
from rag import refresh_vector_cache

_log = get_logger("trigger")

trigger_router = APIRouter(tags=["触发器系统"])

# ---------------------------------------------------------
# 依赖注入
# ---------------------------------------------------------
_db_file: str = ""
_deepseek_client = None
_fn_get_system_context = None
_fn_append_to_memory = None


def configure_trigger(db_file: str, deepseek_client,
                      fn_get_system_context=None,
                      fn_append_to_memory=None):
    global _db_file, _deepseek_client, _fn_get_system_context, _fn_append_to_memory
    _db_file = db_file
    _deepseek_client = deepseek_client
    _fn_get_system_context = fn_get_system_context
    _fn_append_to_memory = fn_append_to_memory
    _ensure_schema()


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
# Schema 迁移（幂等，启动时自动执行）
# ---------------------------------------------------------
def _ensure_schema():
    with safe_db() as conn:
        # triggers 表新增字段
        for col, typedef in [
            ("fire_count",               "INTEGER NOT NULL DEFAULT 0"),
            ("cooldown",                 "INTEGER NOT NULL DEFAULT 0"),
            ("last_fired_at",            "INTEGER NOT NULL DEFAULT 0"),  # Unix 时间戳，避免时区歧义
            ("prerequisite_trigger_ids", "TEXT NOT NULL DEFAULT '[]'"),
            ("exclude_trigger_ids",      "TEXT NOT NULL DEFAULT '[]'"),
            ("max_fire_count",           "INTEGER NOT NULL DEFAULT 0"),
            ("actions",                  "TEXT NOT NULL DEFAULT '[]'"),
        ]:
            try:
                conn.execute(f"ALTER TABLE triggers ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass  # 列已存在

        # trigger_judgements 表（AI 判定审计记录，不作缓存读路径）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trigger_judgements (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_id    INTEGER,
                scene_id      INTEGER,
                scene_name    TEXT,
                timestamp     TEXT,
                reasoning     TEXT,
                result        INTEGER,
                condition_hash TEXT  -- 仅供去重聚合查询，不参与判定逻辑
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_judge_trigger "
            "ON trigger_judgements(trigger_id)"
        )
        try:
            conn.execute("ALTER TABLE trigger_judgements ADD COLUMN condition_hash TEXT")
        except sqlite3.OperationalError:
            pass  # 列已存在

        # game_flags 表：供 set_flag 动作读写命名状态位
        conn.execute("""
            CREATE TABLE IF NOT EXISTS game_flags (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.commit()


# ---------------------------------------------------------
# Pydantic 数据模型
# ---------------------------------------------------------
class ConditionItem(BaseModel):
    type: str
    value: str = ""

class TriggerCreateRequest(BaseModel):
    label: str = "未命名触发器"
    target_node_id: int = 0             # passive 模式可为 0（无跳转目标）
    mode: str = "soft"                  # soft | hard | passive
    conditions: list | dict = []        # 接受列表（旧）或树（新）
    cond_type: str = ""
    cond_value: str = ""
    cooldown: int = 0
    prerequisite_trigger_ids: list[int] = []
    exclude_trigger_ids: list[int] = []
    max_fire_count: int = 0             # 0=无限；>0=触发N次后永久封锁
    actions: list = []                  # 触发时执行的副作用动作列表

class TriggerUpdateRequest(BaseModel):
    label: str
    target_node_id: int = 0
    mode: str
    conditions: list | dict = []
    cond_type: str = ""
    cond_value: str = ""
    cooldown: int = 0
    prerequisite_trigger_ids: list[int] = []
    exclude_trigger_ids: list[int] = []
    max_fire_count: int = 0
    actions: list = []

class CheckTriggersRequest(BaseModel):
    scene_id: int
    scene_name: str
    scene_content: str


# ---------------------------------------------------------
# 条件树：解析 & 规范化
# ---------------------------------------------------------
def _parse_condition_tree(raw: str | None) -> dict:
    """
    将数据库 conditions 字段解析为标准树节点。
    旧格式 [...] 自动升级为 {"op": "and", "children": [...]}。
    """
    if not raw:
        return {"op": "and", "children": []}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"op": "and", "children": []}

    if isinstance(data, list):
        # 旧 AND 列表
        leaves = [{"type": c.get("type", ""), "value": c.get("value", "")}
                  for c in data if c.get("type")]
        return {"op": "and", "children": leaves}

    if isinstance(data, dict) and "op" in data:
        return data

    return {"op": "and", "children": []}


def _normalize_conditions(req) -> str:
    """将请求中的 conditions 字段统一序列化为 JSON 树。"""
    conds = getattr(req, "conditions", [])
    if isinstance(conds, dict) and "op" in conds:
        return json.dumps(conds, ensure_ascii=False)
    if isinstance(conds, list) and conds:
        leaves = [{"type": c.type if hasattr(c, "type") else c.get("type",""),
                   "value": c.value if hasattr(c, "value") else c.get("value","")}
                  for c in conds]
        return json.dumps({"op": "and", "children": leaves}, ensure_ascii=False)
    # 旧单字段回退
    if getattr(req, "cond_type", ""):
        return json.dumps({"op": "and", "children": [
            {"type": req.cond_type, "value": req.cond_value}
        ]}, ensure_ascii=False)
    return json.dumps({"op": "and", "children": []}, ensure_ascii=False)


def _collect_ai_leaves(node: dict) -> list[str]:
    """递归收集树中所有 type=='ai' 的叶节点 value。"""
    if "op" in node:
        result = []
        for child in node.get("children", []):
            result.extend(_collect_ai_leaves(child))
        return result
    if node.get("type") == "ai":
        return [node.get("value", "").strip()]
    return []


# ---------------------------------------------------------
# 单叶节点本地判定
# ---------------------------------------------------------
def _judge_leaf(node: dict, scene_id: int, chars: list, all_inv: str) -> bool | None:
    """返回 True/False 或 None（需 AI 判断）。"""
    ctype = node.get("type", "")
    cval  = node.get("value", "").strip()

    if ctype == "scene":
        try:
            return scene_id == int(cval)
        except ValueError:
            return False

    if ctype == "item":
        kws = [k.strip().lower() for k in cval.split(",") if k.strip()]
        return any(k in all_inv for k in kws)

    if ctype == "stat":
        _OPS = [("<=", lambda a, b: a <= b), (">=", lambda a, b: a >= b),
                ("<",  lambda a, b: a < b),  (">",  lambda a, b: a > b),
                ("==", lambda a, b: a == b)]
        for part in cval.split(","):
            part = part.strip().lower()
            for op_str, op_fn in _OPS:
                if op_str in part:
                    attr, _, thr_str = part.partition(op_str)
                    attr = attr.strip()
                    try:
                        thr = int(thr_str.strip())
                        if any(c["role"] == "PC" and op_fn(int(c.get(attr) or 0), thr)
                               for c in chars):
                            return True
                    except (ValueError, TypeError):
                        pass
                    break
        return False

    if ctype == "ai":
        return None  # 交给 AI 批判

    return False


# ---------------------------------------------------------
# 递归树求值
# ---------------------------------------------------------
def _eval_tree(node: dict, scene_id: int, chars: list, all_inv: str,
               ai_cache: dict, _depth: int = 0) -> bool:
    if _depth > 20:
        _log.warning("条件树深度超限（>20），截断求值返回 False")
        return False
    if "op" in node:
        op = node["op"]
        children = node.get("children", [])
        if not children:
            return False
        if op == "and":
            return all(_eval_tree(c, scene_id, chars, all_inv, ai_cache, _depth+1) for c in children)
        if op == "or":
            return any(_eval_tree(c, scene_id, chars, all_inv, ai_cache, _depth+1) for c in children)
        if op == "not":
            return not _eval_tree(children[0], scene_id, chars, all_inv, ai_cache, _depth+1)
        _log.warning("条件树遇到未知 op: %s，跳过返回 False", op)
        return False
    # 叶节点
    r = _judge_leaf(node, scene_id, chars, all_inv)
    if r is None:  # ai 类型
        return ai_cache.get(node.get("value", "").strip(), False)
    return r


# ---------------------------------------------------------
# 三值预求值（True / False / _MAYBE）— AI 剪枝
# ---------------------------------------------------------
_MAYBE = object()  # 哨兵：AI 条件结果未知，需调用后确定


def _eval_tree_3val(node: dict, scene_id: int, chars: list, all_inv: str,
                    ai_cache: dict, _depth: int = 0):
    """
    在调用 AI 前做剪枝：非 AI 条件按短路逻辑求值，AI 条件返回 _MAYBE。
    返回 True/False 表示已由非 AI 条件确定结果；返回 _MAYBE 则需要 AI。
    已在 ai_cache 中的 AI 条件直接读缓存，视为已知值。
    """
    if _depth > 20:
        return False
    if "op" in node:
        op = node["op"]
        children = node.get("children", [])
        if not children:
            return False
        if op == "and":
            has_maybe = False
            for c in children:
                r = _eval_tree_3val(c, scene_id, chars, all_inv, ai_cache, _depth + 1)
                if r is False:
                    return False        # AND 短路：某非 AI 条件已失败，整体为 False
                if r is _MAYBE:
                    has_maybe = True
            return _MAYBE if has_maybe else True
        if op == "or":
            has_maybe = False
            for c in children:
                r = _eval_tree_3val(c, scene_id, chars, all_inv, ai_cache, _depth + 1)
                if r is True:
                    return True        # OR 短路：某非 AI 条件已满足，整体为 True
                if r is _MAYBE:
                    has_maybe = True
            return _MAYBE if has_maybe else False
        if op == "not":
            r = _eval_tree_3val(children[0], scene_id, chars, all_inv, ai_cache, _depth + 1)
            if r is _MAYBE:
                return _MAYBE
            return not r
        return False
    # 叶节点
    r = _judge_leaf(node, scene_id, chars, all_inv)
    if r is None:                      # ai 类型
        key = node.get("value", "").strip()
        if key in ai_cache:
            return bool(ai_cache[key]) # 命中跨触发器缓存，视为已知
        return _MAYBE
    return r


# ---------------------------------------------------------
# AI 批量判断（judgements 表仅作审计，不作缓存读路径）
# ---------------------------------------------------------
def _batch_judge_ai(values: list[str], scene_name: str, scene_content: str,
                    conn, trigger_id: int | None = None,
                    scene_id: int = 0) -> dict[str, bool]:
    """
    返回 {value: bool} 字典。
    AI 成功时写入 trigger_judgements（审计用）；降级/失败时不落库，避免污染。
    """
    if not values:
        return {}

    # ── 调用 AI ────────────────────────────────────────────────────────────
    worldview, party_status, _, session_memory, \
        _, world_entities_text, _, _ = \
        _fn_get_system_context(conn, scene_name, scene_content)

    n = len(values)
    conditions_text = "\n".join(f"{i+1}. {v}" for i, v in enumerate(values))

    static_system_prompt = (
        "你是一个严谨的剧情逻辑判定器。\n"
        "你的任务是根据玩家的【最新动作】，判断一组触发条件是否成立。\n\n"
        "--- 基础世界观约束 ---\n"
        f"{worldview}\n\n"
        "--- 当前世界实体状态 ---\n"
        f"{world_entities_text}\n\n"
        "--- 判定规则 ---\n"
        "1. 必须【绝对优先】以 User 提供的【核心动作与场景】作为判定依据。\n"
        "2. 必须严格返回 JSON，先进行一句话短推理，再输出结果。\n"
        "3. JSON 格式：\n"
        "{\n"
        "  \"reasoning\": \"简短的逻辑推演（限50字以内）\",\n"
        "  \"results\": [true, false, ...]\n"
        "}\n"
        f"4. results 数组长度必须恰好为 {n}，顺序与条件一致。"
    )

    dynamic_user_prompt = (
        "【近期记忆】\n"
        f"{session_memory}\n\n"
        "【队伍状态】\n"
        f"{party_status}\n\n"
        "====================================\n"
        "【核心动作与场景 (判定最高优先级)】\n"
        f"场景名称：{scene_name}\n"
        f"玩家动作：{scene_content}\n"
        "====================================\n\n"
        f"请针对上述信息，依次判断以下 {n} 个条件是否成立：\n"
        f"{conditions_text}"
    )

    raw_results: list[bool] = [False] * n
    reasoning = ""
    ai_succeeded = False

    try:
        resp = _deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": static_system_prompt},
                {"role": "user",   "content": dynamic_user_prompt},
            ],
            temperature=0.1,
            max_tokens=600 + 10 * n,
            response_format={"type": "json_object"},
        )
        if resp.choices[0].finish_reason == "length":
            _log.warning("AI 触发器推理被截断，降级为全 False，不写缓存")
        else:
            parsed = json.loads(resp.choices[0].message.content)
            reasoning = parsed.get("reasoning", "")
            _log.info("触发器推演逻辑: %s", reasoning)
            results = parsed.get("results", [])
            if len(results) == n:
                raw_results = [bool(r) for r in results]
                ai_succeeded = True
            else:
                _log.warning("AI 返回 results 长度 %d ≠ 期望 %d，降级", len(results), n)
    except Exception as e:
        _log.warning("AI 触发器批量判断失败: %s", e)

    result_map: dict[str, bool] = dict(zip(values, raw_results))

    # ── 仅在 AI 成功时写入审计表，失败/降级不落库避免缓存污染 ──────────────
    if ai_succeeded:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, (val, res) in enumerate(zip(values, raw_results)):
            note = reasoning if i == 0 else f"(batch #{trigger_id} 共 {n} 条)"
            conn.execute(
                "INSERT INTO trigger_judgements "
                "(trigger_id, scene_id, scene_name, timestamp, reasoning, result, condition_hash) "
                "VALUES (?,?,?,?,?,?,?)",
                (trigger_id, scene_id, scene_name, ts, note, int(res),
                 hashlib.md5(val.strip().encode()).hexdigest()),
            )
        # 注意：不在此处 commit，由调用方（check_triggers）统一提交

    return result_map


# ---------------------------------------------------------
# AI 即时生成节点内容辅助
# ---------------------------------------------------------
def _generate_node_ai(prompt: str, scene_name: str, conn) -> dict | None:
    """使用 AI 即兴生成一个剧情节点，返回 {name, content} 或 None（失败时）。"""
    if not _deepseek_client or not _fn_get_system_context:
        return None
    try:
        worldview, party_status, _, session_memory, \
            _, world_entities_text, _, _ = \
            _fn_get_system_context(conn, scene_name, prompt)

        system_prompt = (
            "你是一个专业的 TRPG 叙事生成器。根据提示即兴创作一个新剧情节点。\n"
            "语言沉浸、简洁有力，禁止出戏或解释性旁白。\n\n"
            "--- 世界观 ---\n"
            f"{worldview}\n\n"
            "--- 当前世界实体状态 ---\n"
            f"{world_entities_text}\n\n"
            "返回严格 JSON（仅此两字段）：\n"
            '{"name": "节点标题（10字以内）", "content": "场景骨架（60-100字）"}'
        )
        user_prompt = (
            "【近期记忆】\n"
            f"{session_memory}\n\n"
            "【队伍状态】\n"
            f"{party_status}\n\n"
            "【当前场景】\n"
            f"{scene_name}\n\n"
            "【节点生成提示】\n"
            f"{prompt}"
        )
        resp = _deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.75,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        if resp.choices[0].finish_reason == "length":
            _log.warning("gen_node AI 截断")
            return None
        parsed = json.loads(resp.choices[0].message.content)
        name    = str(parsed.get("name", "触发生成节点")).strip()[:80]
        content = str(parsed.get("content", "")).strip()
        return {"name": name, "content": content} if content else None
    except Exception as e:
        _log.warning("gen_node AI 调用失败: %s", e)
        return None


# ---------------------------------------------------------
# 触发器动作执行（触发时副作用）
# 返回 side_effects dict 供 check_triggers 聚合后下发给前端
# ---------------------------------------------------------
def _execute_actions(actions_raw: str, conn, trigger_id: int,
                     scene_id: int, scene_name: str) -> dict:
    """
    动作类型一览：
      set_flag      — {key, value}：写入 game_flags 表
      mod_stat      — {target, attr, delta}：改角色 hp/san，target="PC" 作用全体玩家角色
      add_memory    — {text}：写入会话记忆
      fire_trigger  — {id}：强制激活另一触发器（不递归执行其动作，防止循环）
      inject_text   — {node_id?, text}：向指定节点追加叙事文字（node_id=0 → 当前场景）
      inject_option — {node_id?, text, next_node_id}：向指定节点注入选项
      gen_node      — {prompt, mode}：AI 即时生成新节点并返回供前端跳转
      mod_inventory — {target, op, item}：op="add"|"remove"，修改角色 inventory 纯文本字段
      mod_npc       — {name, field, value?, delta?}：改实体 status / emotion_* / memory_add
      entity_script — {name, action}：appear | leave | die | attack
    """
    side_effects: dict = {
        "text_injections":  [],   # [{node_id, text}]
        "option_injections": [],  # [{node_id, option_id, text, next_node_id}]
        "generated_nodes":  [],   # [{trigger_id, node_id, node_name, node_content, mode}]
        "log":              [],   # [str] 人类可读的动作执行摘要，下发给前端显示
        "reveal_rag_ids":   [],   # [int] reveal_rag 解锁的 doc_id，commit 后刷新向量缓存
    }

    try:
        actions = json.loads(actions_raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return side_effects

    for act in actions:
        atype = act.get("type", "")
        try:
            # ── 原有四种 ──────────────────────────────────────────────────────
            if atype == "set_flag":
                key = str(act.get("key", "")).strip()
                val = str(act.get("value", "")).strip()
                if key:
                    conn.execute(
                        "INSERT OR REPLACE INTO game_flags (key, value) VALUES (?, ?)",
                        (key, val)
                    )
                    _log.info("触发器 %d set_flag: %s = %s", trigger_id, key, val)
                    side_effects["log"].append(f"🚩 标志位 {key} = {val}")

            elif atype == "mod_stat":
                target = str(act.get("target", "PC")).strip()
                attr   = str(act.get("attr",   "hp")).strip().lower()
                delta  = int(act.get("delta",  0))
                if attr in ("hp", "san") and delta != 0:
                    if target.upper() == "PC":
                        conn.execute(
                            f"UPDATE characters SET {attr}=MAX(0,{attr}+?) WHERE role='PC'",
                            (delta,)
                        )
                    else:
                        conn.execute(
                            f"UPDATE characters SET {attr}=MAX(0,{attr}+?) WHERE name=?",
                            (delta, target)
                        )
                    _log.info("触发器 %d mod_stat: [%s] %s %+d", trigger_id, target, attr, delta)
                    label = "全体玩家" if target.upper() == "PC" else target
                    side_effects["log"].append(f"💢 {label} {attr.upper()} {delta:+d}")

            elif atype == "add_memory":
                text = str(act.get("text", "")).strip()
                if text and _fn_append_to_memory:
                    _fn_append_to_memory(conn, text)
                    _log.info("触发器 %d add_memory: %s…", trigger_id, text[:40])
                    side_effects["log"].append(f"📝 记忆已写入：{text[:30]}…" if len(text) > 30 else f"📝 记忆已写入：{text}")

            elif atype == "fire_trigger":
                tid = int(act.get("id", 0))
                if tid and tid != trigger_id:
                    row = conn.execute(
                        "SELECT fire_count, cooldown, max_fire_count FROM triggers WHERE id=?",
                        (tid,)
                    ).fetchone()
                    if row:
                        new_cnt   = (row["fire_count"] or 0) + 1
                        max_fc    = row["max_fire_count"] or 0
                        new_fired = 1 if (row["cooldown"] or 0) == 0 or (max_fc > 0 and new_cnt >= max_fc) else 0
                        conn.execute(
                            "UPDATE triggers SET fired=?, fire_count=?, last_fired_at=? WHERE id=?",
                            (new_fired, new_cnt, int(time.time()), tid)
                        )
                        _log.info("触发器 %d fire_trigger: 激活 #%d（第%d次）", trigger_id, tid, new_cnt)
                        side_effects["log"].append(f"⚡ 触发器 #{tid} 已激活（第{new_cnt}次）")

            # ── 叙事层 ────────────────────────────────────────────────────────
            elif atype == "inject_text":
                target_node = int(act.get("node_id", 0) or 0) or scene_id
                text = str(act.get("text", "")).strip()
                if text and target_node:
                    row = conn.execute(
                        "SELECT content, expanded_content FROM nodes WHERE id=?",
                        (target_node,)
                    ).fetchone()
                    if row:
                        if row["expanded_content"]:
                            conn.execute(
                                "UPDATE nodes SET expanded_content=expanded_content||? WHERE id=?",
                                ("\n\n" + text, target_node)
                            )
                            # 同步追加到战报史册最近一条记录的深入调查快照
                            conn.execute(
                                "UPDATE chronicle_log SET expanded_content=expanded_content||? "
                                "WHERE id=(SELECT id FROM chronicle_log WHERE scene_id=? "
                                "ORDER BY id DESC LIMIT 1)",
                                ("\n\n" + text, target_node)
                            )
                        else:
                            conn.execute(
                                "UPDATE nodes SET content=content||? WHERE id=?",
                                ("\n\n" + text, target_node)
                            )
                            # 同步追加到战报史册最近一条记录的场景快照
                            conn.execute(
                                "UPDATE chronicle_log SET scene_content=scene_content||? "
                                "WHERE id=(SELECT id FROM chronicle_log WHERE scene_id=? "
                                "ORDER BY id DESC LIMIT 1)",
                                ("\n\n" + text, target_node)
                            )
                        side_effects["text_injections"].append(
                            {"node_id": target_node, "text": text}
                        )
                        _log.info("触发器 %d inject_text → 节点%d: %s…", trigger_id, target_node, text[:30])
                        side_effects["log"].append(f"📖 文本已注入当前场景")

            elif atype == "inject_option":
                target_node = int(act.get("node_id", 0) or 0) or scene_id
                opt_text    = str(act.get("text", "")).strip()
                next_nid    = int(act.get("next_node_id", 0) or 0)
                if opt_text and next_nid and target_node:
                    cur = conn.execute(
                        "INSERT INTO options (node_id, text, next_node_id) VALUES (?,?,?)",
                        (target_node, opt_text, next_nid)
                    )
                    side_effects["option_injections"].append({
                        "node_id":     target_node,
                        "option_id":   cur.lastrowid,
                        "text":        opt_text,
                        "next_node_id": next_nid,
                    })
                    _log.info("触发器 %d inject_option → 节点%d: [%s]→%d",
                              trigger_id, target_node, opt_text, next_nid)
                    side_effects["log"].append(f"🔀 新选项已注入：「{opt_text[:20]}」→ 节点#{next_nid}")

            elif atype == "gen_node":
                prompt   = str(act.get("prompt", "")).strip()
                nav_mode = str(act.get("mode", "soft")).strip()
                if prompt:
                    result = _generate_node_ai(prompt, scene_name, conn)
                    if result:
                        cur = conn.execute(
                            "INSERT INTO nodes (name, summary, content) VALUES (?,?,?)",
                            (result["name"], "", result["content"])
                        )
                        new_nid = cur.lastrowid
                        side_effects["generated_nodes"].append({
                            "trigger_id":   trigger_id,
                            "node_id":      new_nid,
                            "node_name":    result["name"],
                            "node_content": result["content"],
                            "mode":         nav_mode,
                        })
                        _log.info("触发器 %d gen_node → 新节点%d「%s」", trigger_id, new_nid, result["name"])
                        side_effects["log"].append(f"✨ AI 已生成节点「{result['name']}」(#{new_nid})")

            # ── 实体层 ────────────────────────────────────────────────────────
            elif atype == "mod_inventory":
                target = str(act.get("target", "PC")).strip()
                op     = str(act.get("op",     "add")).strip()
                item   = str(act.get("item",   "")).strip()
                if item:
                    if target.upper() == "PC":
                        chars = [dict(r) for r in conn.execute(
                            "SELECT id, inventory FROM characters WHERE role='PC'"
                        ).fetchall()]
                    else:
                        chars = [dict(r) for r in conn.execute(
                            "SELECT id, inventory FROM characters WHERE name=?", (target,)
                        ).fetchall()]
                    for c in chars:
                        inv = (c.get("inventory") or "").strip()
                        if op == "add":
                            new_inv = (inv + ", " + item).lstrip(", ")
                        else:
                            parts   = [p.strip() for p in inv.split(",")]
                            parts   = [p for p in parts if item.lower() not in p.lower()]
                            new_inv = ", ".join(p for p in parts if p)
                        conn.execute(
                            "UPDATE characters SET inventory=? WHERE id=?",
                            (new_inv, c["id"])
                        )
                    _log.info("触发器 %d mod_inventory: [%s] %s %s", trigger_id, target, op, item)
                    label = "全体玩家" if target.upper() == "PC" else target
                    op_str = "获得" if op == "add" else "失去"
                    side_effects["log"].append(f"🎒 {label} {op_str}「{item}」")

            elif atype == "mod_npc":
                name  = str(act.get("name", "")).strip()
                field = str(act.get("field", "status")).strip()
                if not name:
                    continue
                row = conn.execute(
                    "SELECT id, state_desc FROM world_entities WHERE name=?", (name,)
                ).fetchone()
                if not row:
                    _log.warning("触发器 %d mod_npc: 找不到实体「%s」", trigger_id, name)
                    side_effects["log"].append(f"👤 ⚠ 找不到实体「{name}」")
                    continue
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if field == "status":
                    val = str(act.get("value", "active")).strip()
                    conn.execute(
                        "UPDATE world_entities SET status=?, updated_at=? WHERE name=?",
                        (val, ts, name)
                    )
                    _log.info("触发器 %d mod_npc: %s.status → %s", trigger_id, name, val)
                    side_effects["log"].append(f"👤 {name} 状态 → {val}")

                elif field.startswith("emotion_"):
                    emo_key = field[len("emotion_"):]
                    delta   = int(act.get("delta", 0))
                    try:
                        sd = json.loads(row["state_desc"] or "{}")
                    except (json.JSONDecodeError, TypeError):
                        sd = {}
                    emo = sd.get("emotion", {"trust": 0, "fear": 0, "irritation": 0})
                    emo[emo_key] = max(-100, min(100, emo.get(emo_key, 0) + delta))
                    sd["emotion"] = emo
                    conn.execute(
                        "UPDATE world_entities SET state_desc=?, updated_at=? WHERE name=?",
                        (json.dumps(sd, ensure_ascii=False), ts, name)
                    )
                    _log.info("触发器 %d mod_npc: %s.%s %+d", trigger_id, name, emo_key, delta)
                    emo_names = {"trust": "信任", "fear": "恐惧", "irritation": "愤怒"}
                    side_effects["log"].append(f"👤 {name} {emo_names.get(emo_key, emo_key)} {delta:+d} → {emo[emo_key]}")

                elif field == "memory_add":
                    mem_text = str(act.get("value", "")).strip()
                    if mem_text:
                        try:
                            sd = json.loads(row["state_desc"] or "{}")
                        except (json.JSONDecodeError, TypeError):
                            sd = {}
                        mem = sd.get("memory", [])
                        mem.append(mem_text)
                        sd["memory"] = mem[-20:]
                        conn.execute(
                            "UPDATE world_entities SET state_desc=?, updated_at=? WHERE name=?",
                            (json.dumps(sd, ensure_ascii=False), ts, name)
                        )
                        _log.info("触发器 %d mod_npc memory_add: %s ← %s…", trigger_id, name, mem_text[:30])
                        side_effects["log"].append(f"👤 {name} 记忆已写入")

            elif atype == "entity_script":
                name   = str(act.get("name",   "")).strip()
                action = str(act.get("action", "appear")).strip()
                if not name:
                    continue
                row = conn.execute(
                    "SELECT id, state_desc FROM world_entities WHERE name=?", (name,)
                ).fetchone()
                if not row:
                    _log.warning("触发器 %d entity_script: 找不到实体「%s」", trigger_id, name)
                    side_effects["log"].append(f"⚙️ ⚠ 找不到实体「{name}」")
                    continue
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if action in ("appear", "leave", "die"):
                    status_map = {"appear": "active", "leave": "moved", "die": "dead"}
                    conn.execute(
                        "UPDATE world_entities SET status=?, updated_at=? WHERE name=?",
                        (status_map[action], ts, name)
                    )
                    _log.info("触发器 %d entity_script: %s → %s", trigger_id, name, action)
                    action_names = {"appear": "登场", "leave": "离场", "die": "死亡"}
                    side_effects["log"].append(f"⚙️ {name} {action_names[action]}")

                elif action == "attack":
                    try:
                        sd = json.loads(row["state_desc"] or "{}")
                    except (json.JSONDecodeError, TypeError):
                        sd = {}
                    emo = sd.get("emotion", {"trust": 0, "fear": 0, "irritation": 0})
                    emo["irritation"] = min(100, emo.get("irritation", 0) + 30)
                    emo["trust"]      = max(-100, emo.get("trust", 0) - 20)
                    mem = sd.get("memory", [])
                    mem.append(f"[触发器] 在场景「{scene_name}」对玩家发动攻击")
                    sd["emotion"] = emo
                    sd["memory"]  = mem[-20:]
                    conn.execute(
                        "UPDATE world_entities SET state_desc=?, updated_at=? WHERE name=?",
                        (json.dumps(sd, ensure_ascii=False), ts, name)
                    )
                    _log.info("触发器 %d entity_script: %s 发动攻击", trigger_id, name)
                    side_effects["log"].append(f"⚙️ {name} 发动攻击（愤怒+30，信任-20）")

            # ── 世界构建层 ───────────────────────────────────────────────────
            elif atype == "spawn_npc":
                name      = str(act.get("name", "")).strip()[:50]
                hp        = int(act.get("hp",  100))
                san       = int(act.get("san",  80))
                inventory = str(act.get("inventory", "")).strip()[:200]
                desc      = str(act.get("desc", "")).strip()
                if not name:
                    continue
                if conn.execute("SELECT id FROM characters WHERE name=?", (name,)).fetchone():
                    _log.warning("触发器 %d spawn_npc: 角色「%s」已存在，跳过", trigger_id, name)
                    side_effects["log"].append(f"🧑 NPC「{name}」已存在，跳过")
                    continue
                conn.execute(
                    "INSERT INTO characters (name, role, hp, san, inventory, status) VALUES (?,?,?,?,?,?)",
                    (name, "NPC", hp, san, inventory, "active")
                )
                state_desc = json.dumps({
                    "desc":       desc or f"{name}，新登场的 NPC",
                    "emotion":    {"trust": 0, "fear": 0, "irritation": 0},
                    "breakpoint": {"threshold": 70, "trigger_field": "irritation", "reaction": ""},
                    "memory":     [],
                }, ensure_ascii=False)
                if not conn.execute("SELECT id FROM world_entities WHERE name=?", (name,)).fetchone():
                    conn.execute(
                        "INSERT INTO world_entities (entity_type, name, status, state_desc, updated_at) "
                        "VALUES (?,?,?,?,?)",
                        ("npc", name, "active", state_desc,
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    )
                _log.info("触发器 %d spawn_npc: 创建 NPC「%s」HP=%d SAN=%d", trigger_id, name, hp, san)
                side_effects["log"].append(f"🧑 NPC「{name}」已登场（HP {hp} / SAN {san}）")

            elif atype == "mod_worldview":
                op   = str(act.get("op",   "append")).strip()
                text = str(act.get("text", "")).strip()
                if not text:
                    continue
                if op == "replace":
                    conn.execute(
                        "UPDATE system_state SET value=? WHERE key='worldview'", (text,)
                    )
                else:
                    conn.execute(
                        "UPDATE system_state SET value=value||? WHERE key='worldview'",
                        ("\n\n" + text,)
                    )
                _log.info("触发器 %d mod_worldview[%s]: %s…", trigger_id, op, text[:40])
                op_str = "已替换" if op == "replace" else "已追加"
                side_effects["log"].append(f"🌍 世界观{op_str}（{len(text)} 字）")

            elif atype == "add_lore":
                keywords = str(act.get("keywords", "")).strip()
                content  = str(act.get("content",  "")).strip()
                if keywords and content:
                    conn.execute(
                        "INSERT INTO lorebook (keywords, content) VALUES (?,?)",
                        (keywords, content)
                    )
                    _log.info("触发器 %d add_lore: [%s] %s…", trigger_id, keywords, content[:30])
                    side_effects["log"].append(f"📚 Lorebook 新增：[{keywords[:20]}]")

            elif atype == "reveal_rag":
                doc_id = int(act.get("doc_id", 0))
                if doc_id:
                    row = conn.execute("SELECT title FROM rag_documents WHERE id=?", (doc_id,)).fetchone()
                    conn.execute(
                        "UPDATE rag_documents SET hidden=0 WHERE id=?",
                        (doc_id,)
                    )
                    side_effects["reveal_rag_ids"].append(doc_id)
                    _log.info("触发器 %d reveal_rag: doc_id=%d 已解锁", trigger_id, doc_id)
                    title = row["title"] if row else f"#{doc_id}"
                    side_effects["log"].append(f"🔓 RAG 文档「{title}」已解锁并载入向量库")

        except Exception as e:
            _log.warning("触发器 %d action[%s] 执行失败: %s", trigger_id, atype, e)

    return side_effects


def _has_gen_node_action(actions) -> bool:
    """动作列表中含有 gen_node 时，跳转目标由 AI 运行时生成，无需固定 target_node_id。"""
    if isinstance(actions, str):
        try:
            actions = json.loads(actions)
        except (json.JSONDecodeError, TypeError):
            return False
    return any(isinstance(a, dict) and a.get("type") == "gen_node" for a in (actions or []))


# ---------------------------------------------------------
# 触发器完整检查（含 DAG、前驱、互斥、冷却、次数上限）
# ---------------------------------------------------------
def _check_trigger(trigger_row, scene_id: int, scene_name: str,
                   scene_content: str, chars: list, all_inv: str,
                   ai_cache: dict, conn) -> bool:
    t = trigger_row

    # ── 前驱：所有前驱触发器必须已 fire ────────────────────────────────────
    try:
        prereq_ids = json.loads(t["prerequisite_trigger_ids"] or "[]")
    except (TypeError, json.JSONDecodeError):
        prereq_ids = []
    if prereq_ids:
        for pid in prereq_ids:
            row = conn.execute("SELECT fire_count, fired FROM triggers WHERE id=?", (pid,)).fetchone()
            if not row or (row["fire_count"] == 0 and row["fired"] == 0):
                return False

    # ── 互斥：所有排斥触发器必须未 fire ────────────────────────────────────
    try:
        excl_ids = json.loads(t["exclude_trigger_ids"] or "[]")
    except (TypeError, json.JSONDecodeError):
        excl_ids = []
    if excl_ids:
        for eid in excl_ids:
            row = conn.execute("SELECT fire_count, fired FROM triggers WHERE id=?", (eid,)).fetchone()
            if row and (row["fire_count"] > 0 or row["fired"] == 1):
                return False

    # ── 冷却检查 ────────────────────────────────────────────────────────────
    cooldown = t["cooldown"] if "cooldown" in t.keys() else 0
    if cooldown and cooldown > 0:
        last_ts = t["last_fired_at"] if "last_fired_at" in t.keys() else 0
        if last_ts and (time.time() - int(last_ts)) < cooldown:
            return False

    # ── 触发次数上限 ─────────────────────────────────────────────────────────
    max_fire = t["max_fire_count"] if "max_fire_count" in t.keys() else 0
    if max_fire > 0:
        cur_count = t["fire_count"] if "fire_count" in t.keys() else 0
        if cur_count >= max_fire:
            return False

    # ── 条件树求值 ──────────────────────────────────────────────────────────
    tree = _parse_condition_tree(t["conditions"] if "conditions" in t.keys() else "[]")
    if not tree.get("children"):
        return False

    # ── 三值预求值：非 AI 条件短路剪枝，结果确定时直接返回，跳过 AI 调用 ──
    pre = _eval_tree_3val(tree, scene_id, chars, all_inv, ai_cache)
    if pre is False:
        return False
    if pre is True:
        return True

    # 只有 _MAYBE（非 AI 条件无法单独确定结果）才收集 AI 条件并批量请求
    ai_leaves = _collect_ai_leaves(tree)
    new_ai = [v for v in ai_leaves if v and v not in ai_cache]
    if new_ai:
        unique = list(dict.fromkeys(new_ai))
        batch = _batch_judge_ai(unique, scene_name, scene_content, conn,
                                trigger_id=t["id"], scene_id=scene_id)
        ai_cache.update(batch)

    return _eval_tree(tree, scene_id, chars, all_inv, ai_cache)


# ---------------------------------------------------------
# REST API 端点
# ---------------------------------------------------------
@trigger_router.get("/api/game/triggers")
def get_triggers():
    with safe_db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM triggers ORDER BY id").fetchall()]
    for r in rows:
        tree = _parse_condition_tree(r.get("conditions", "[]"))
        r["conditions"] = tree
        r.setdefault("fire_count", 0)
        r.setdefault("cooldown", 0)
        r.setdefault("last_fired_at", "")
        r.setdefault("prerequisite_trigger_ids", [])
        r.setdefault("exclude_trigger_ids", [])
        r.setdefault("max_fire_count", 0)
        r.setdefault("actions", [])
        for fld in ("prerequisite_trigger_ids", "exclude_trigger_ids", "actions"):
            if isinstance(r[fld], str):
                try:
                    r[fld] = json.loads(r[fld])
                except (json.JSONDecodeError, TypeError):
                    r[fld] = []
    return {"status": "success", "triggers": rows}


@trigger_router.post("/api/game/trigger")
def create_trigger(req: TriggerCreateRequest):
    with safe_db() as conn:
        if req.target_node_id and req.mode != "passive" and not _has_gen_node_action(req.actions):
            if not conn.execute("SELECT id FROM nodes WHERE id=?", (req.target_node_id,)).fetchone():
                raise fastapi.HTTPException(status_code=400, detail="目标节点不存在")
        conditions_json = _normalize_conditions(req)
        c = conn.execute(
            "INSERT INTO triggers "
            "(label, target_node_id, mode, cond_type, cond_value, conditions, "
            " cooldown, prerequisite_trigger_ids, exclude_trigger_ids, max_fire_count, actions) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (req.label[:80], req.target_node_id or 0, req.mode,
             req.cond_type, req.cond_value[:200],
             conditions_json, req.cooldown,
             json.dumps(req.prerequisite_trigger_ids),
             json.dumps(req.exclude_trigger_ids),
             req.max_fire_count,
             json.dumps(req.actions, ensure_ascii=False))
        )
        conn.commit()
    return {"status": "success", "id": c.lastrowid}


@trigger_router.put("/api/game/trigger/{tid}")
def update_trigger(tid: int, req: TriggerUpdateRequest):
    with safe_db() as conn:
        if req.target_node_id and req.mode != "passive" and not _has_gen_node_action(req.actions):
            if not conn.execute("SELECT id FROM nodes WHERE id=?", (req.target_node_id,)).fetchone():
                raise fastapi.HTTPException(status_code=400, detail="目标节点不存在")
        conditions_json = _normalize_conditions(req)
        conn.execute(
            "UPDATE triggers SET label=?, target_node_id=?, mode=?, "
            "cond_type=?, cond_value=?, conditions=?, "
            "cooldown=?, prerequisite_trigger_ids=?, exclude_trigger_ids=?, "
            "max_fire_count=?, actions=? "
            "WHERE id=?",
            (req.label[:80], req.target_node_id or 0, req.mode,
             req.cond_type, req.cond_value[:200],
             conditions_json, req.cooldown,
             json.dumps(req.prerequisite_trigger_ids),
             json.dumps(req.exclude_trigger_ids),
             req.max_fire_count,
             json.dumps(req.actions, ensure_ascii=False),
             tid)
        )
        conn.commit()
    return {"status": "success"}


@trigger_router.delete("/api/game/trigger/{tid}")
def delete_trigger(tid: int):
    with safe_db() as conn:
        conn.execute("DELETE FROM triggers WHERE id=?", (tid,))
        conn.commit()
    return {"status": "success"}


@trigger_router.post("/api/game/trigger/{tid}/reset")
def reset_trigger(tid: int):
    with safe_db() as conn:
        conn.execute(
            "UPDATE triggers SET fired=0, fire_count=0, last_fired_at=0 WHERE id=?",
            (tid,)
        )
        conn.commit()
    return {"status": "success"}


@trigger_router.post("/api/game/triggers/reset-all")
def reset_all_triggers():
    """将所有触发器重置为未触发状态（调试用）。"""
    with safe_db() as conn:
        conn.execute("UPDATE triggers SET fired=0, fire_count=0, last_fired_at=0")
        conn.commit()
    return {"status": "success"}


@trigger_router.get("/api/game/trigger-judgements")
def get_trigger_judgements(limit: int = 50):
    """查询最近的 AI 判定记录，供 GM 追溯。"""
    with safe_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM trigger_judgements ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]
    return {"status": "success", "judgements": rows}


@trigger_router.post("/api/game/check-triggers")
def check_triggers(req: CheckTriggersRequest):
    """检查所有符合条件的触发器（DAG 条件树）。"""
    conn = get_db_connection()
    try:
        # 取未永久封锁的触发器：
        # - fired=0：从未触发或无冷却可用
        # - cooldown>0 AND 未达 max_fire 上限：可重复触发
        pending = conn.execute(
            "SELECT * FROM triggers WHERE fired=0 "
            "OR (cooldown>0 AND (max_fire_count=0 OR fire_count < max_fire_count))"
        ).fetchall()
        if not pending:
            conn.close()
            return {"status": "success", "fired": []}

        chars = [dict(r) for r in conn.execute("SELECT * FROM characters").fetchall()]
        all_inv = " ".join([(c.get("inventory") or "").lower() for c in chars])
        ai_cache: dict[str, bool] = {}

        fired_results    = []
        text_injections  = []
        option_injections = []
        generated_nodes  = []
        any_reveal_rag   = False

        for t in pending:
            if not _check_trigger(t, req.scene_id, req.scene_name,
                                   req.scene_content, chars, all_inv,
                                   ai_cache, conn):
                continue

            # ── 执行触发器动作副作用，收集前端需感知的副作用 ───────────────────
            side = _execute_actions(
                t["actions"] if "actions" in t.keys() else "[]",
                conn, t["id"], req.scene_id, req.scene_name
            )
            text_injections.extend(side["text_injections"])
            option_injections.extend(side["option_injections"])
            if side["reveal_rag_ids"]:
                any_reveal_rag = True

            # gen_node 条目携带 actions_log，供前端在生成节点 toast 中展示
            for gn in side["generated_nodes"]:
                gn["actions_log"] = side["log"]
            generated_nodes.extend(side["generated_nodes"])

            # ── 更新触发状态 ─────────────────────────────────────────────────
            now_ts    = int(time.time())
            cooldown  = t["cooldown"]       if "cooldown"       in t.keys() else 0
            max_fire  = t["max_fire_count"] if "max_fire_count" in t.keys() else 0
            new_count = (t["fire_count"]    if "fire_count"     in t.keys() else 0) + 1

            if cooldown == 0:
                new_fired = 1                                  # 无冷却：单次触发，永久封锁
            elif max_fire > 0 and new_count >= max_fire:
                new_fired = 1                                  # 达到次数上限，永久封锁
            else:
                new_fired = t["fired"]                         # 有冷却且未达上限，保持状态

            conn.execute(
                "UPDATE triggers SET fired=?, fire_count=?, last_fired_at=? WHERE id=?",
                (new_fired, new_count, now_ts, t["id"])
            )

            # ── 构建结果条目 ─────────────────────────────────────────────────
            # gen_node 触发器：跳转通知走 generated_nodes 通道，避免双重弹窗
            # 且 target_node_id=0，fired_results 里的 hard 模式会错误跳转到节点0
            if not side["generated_nodes"]:
                target_node = conn.execute(
                    "SELECT * FROM nodes WHERE id=?", (t["target_node_id"],)
                ).fetchone() if t["target_node_id"] else None

                fired_results.append({
                    "trigger_id":       t["id"],
                    "label":            t["label"],
                    "mode":             t["mode"],
                    "target_node_id":   t["target_node_id"],
                    "target_node_name": target_node["name"] if target_node else "（无跳转目标）",
                    "fire_count":       new_count,
                    "actions_log":      side["log"],
                })

            # passive 模式无导航意图，不写记忆；soft/hard 模式写简洁的自然语言场景提示
            # hard 模式场景切换的详细记忆由前端 log-scene-visit 统一写入，此处仅补充触发背景
            if t["mode"] != "passive" and _fn_append_to_memory:
                if side["generated_nodes"]:
                    gn = side["generated_nodes"][0]
                    _fn_append_to_memory(
                        conn,
                        f"触发「{t['label']}」，进入「{gn['node_name']}」。"  #触发+触发器名称
                    )
                elif t["target_node_id"]:
                    target_node = conn.execute(
                        "SELECT name FROM nodes WHERE id=?", (t["target_node_id"],)
                    ).fetchone()
                    scene_label = target_node["name"] if target_node else f"场景{t['target_node_id']}"
                    _fn_append_to_memory(
                        conn,
                        f"触发「{t['label']}」，场景切换至「{scene_label}」。"
                    )

        conn.commit()
        conn.close()
        if any_reveal_rag:
            refresh_vector_cache()
        return {
            "status":           "success",
            "fired":            fired_results,
            "text_injections":  text_injections,
            "option_injections": option_injections,
            "generated_nodes":  generated_nodes,
        }
    except Exception:
        conn.close()
        raise
