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
        conn.commit()


# ---------------------------------------------------------
# Pydantic 数据模型
# ---------------------------------------------------------
class ConditionItem(BaseModel):
    type: str
    value: str = ""

class TriggerCreateRequest(BaseModel):
    label: str = "未命名触发器"
    target_node_id: int
    mode: str = "soft"
    conditions: list | dict = []        # 接受列表（旧）或树（新）
    cond_type: str = ""
    cond_value: str = ""
    cooldown: int = 0
    prerequisite_trigger_ids: list[int] = []
    exclude_trigger_ids: list[int] = []

class TriggerUpdateRequest(BaseModel):
    label: str
    target_node_id: int
    mode: str
    conditions: list | dict = []
    cond_type: str = ""
    cond_value: str = ""
    cooldown: int = 0
    prerequisite_trigger_ids: list[int] = []
    exclude_trigger_ids: list[int] = []

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
# 触发器完整检查（含 DAG、前驱、互斥、冷却）
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
        for fld in ("prerequisite_trigger_ids", "exclude_trigger_ids"):
            if isinstance(r[fld], str):
                try:
                    r[fld] = json.loads(r[fld])
                except (json.JSONDecodeError, TypeError):
                    r[fld] = []
    return {"status": "success", "triggers": rows}


@trigger_router.post("/api/game/trigger")
def create_trigger(req: TriggerCreateRequest):
    with safe_db() as conn:
        if not conn.execute("SELECT id FROM nodes WHERE id=?", (req.target_node_id,)).fetchone():
            raise fastapi.HTTPException(status_code=400, detail="目标节点不存在")
        conditions_json = _normalize_conditions(req)
        c = conn.execute(
            "INSERT INTO triggers "
            "(label, target_node_id, mode, cond_type, cond_value, conditions, "
            " cooldown, prerequisite_trigger_ids, exclude_trigger_ids) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (req.label[:80], req.target_node_id, req.mode,
             req.cond_type, req.cond_value[:200],
             conditions_json, req.cooldown,
             json.dumps(req.prerequisite_trigger_ids),
             json.dumps(req.exclude_trigger_ids))
        )
        conn.commit()
    return {"status": "success", "id": c.lastrowid}


@trigger_router.put("/api/game/trigger/{tid}")
def update_trigger(tid: int, req: TriggerUpdateRequest):
    with safe_db() as conn:
        if not conn.execute("SELECT id FROM nodes WHERE id=?", (req.target_node_id,)).fetchone():
            raise fastapi.HTTPException(status_code=400, detail="目标节点不存在")
        conditions_json = _normalize_conditions(req)
        conn.execute(
            "UPDATE triggers SET label=?, target_node_id=?, mode=?, "
            "cond_type=?, cond_value=?, conditions=?, "
            "cooldown=?, prerequisite_trigger_ids=?, exclude_trigger_ids=? "
            "WHERE id=?",
            (req.label[:80], req.target_node_id, req.mode,
             req.cond_type, req.cond_value[:200],
             conditions_json, req.cooldown,
             json.dumps(req.prerequisite_trigger_ids),
             json.dumps(req.exclude_trigger_ids),
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
        # 取未永久封锁的触发器（fired=0 或 cooldown>0 允许重复触发）
        pending = conn.execute(
            "SELECT * FROM triggers WHERE fired=0 OR cooldown>0"
        ).fetchall()
        if not pending:
            conn.close()
            return {"status": "success", "fired": []}

        chars = [dict(r) for r in conn.execute("SELECT * FROM characters").fetchall()]
        all_inv = " ".join([(c.get("inventory") or "").lower() for c in chars])
        ai_cache: dict[str, bool] = {}

        fired_results = []
        for t in pending:
            if not _check_trigger(t, req.scene_id, req.scene_name,
                                   req.scene_content, chars, all_inv,
                                   ai_cache, conn):
                continue

            now_ts = int(time.time())
            # 只有 cooldown==0 时才永久标 fired=1（单次触发语义保留）
            new_fired = 1 if (t["cooldown"] if "cooldown" in t.keys() else 0) == 0 else t["fired"]
            new_count = (t["fire_count"] if "fire_count" in t.keys() else 0) + 1
            conn.execute(
                "UPDATE triggers SET fired=?, fire_count=?, last_fired_at=? WHERE id=?",
                (new_fired, new_count, now_ts, t["id"])
            )

            target_node = conn.execute(
                "SELECT * FROM nodes WHERE id=?", (t["target_node_id"],)
            ).fetchone()
            fired_results.append({
                "trigger_id":       t["id"],
                "label":            t["label"],
                "mode":             t["mode"],
                "target_node_id":   t["target_node_id"],
                "target_node_name": target_node["name"] if target_node else "未知节点",
                "fire_count":       new_count,
            })

            if _fn_append_to_memory:
                _fn_append_to_memory(
                    conn,
                    f"关键触发器「{t['label']}」已触发（第{new_count}次），"
                    f"剧情指向节点[{t['target_node_id']}]。"
                )

        conn.commit()
        conn.close()
        return {"status": "success", "fired": fired_results}
    except Exception:
        conn.close()
        raise
