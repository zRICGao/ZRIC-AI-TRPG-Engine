"""
Z.R.I.C 引擎 — 触发器系统模块 (trigger.py)
关键节点触发器 CRUD + 条件检查。
支持 AND 门结构：conditions 数组中多个条件全部满足才触发。
由 main.py 通过 app.include_router(trigger_router) 挂载。
"""

import json
import sqlite3
import fastapi
from contextlib import contextmanager
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
class ConditionItem(BaseModel):
    type: str       # scene / item / stat / ai
    value: str = ""

class TriggerCreateRequest(BaseModel):
    label: str = "未命名触发器"
    target_node_id: int
    mode: str = "soft"
    conditions: list[ConditionItem] = []
    # 向后兼容旧单条件字段
    cond_type: str = ""
    cond_value: str = ""

class TriggerUpdateRequest(BaseModel):
    label: str
    target_node_id: int
    mode: str
    conditions: list[ConditionItem] = []
    cond_type: str = ""
    cond_value: str = ""

class CheckTriggersRequest(BaseModel):
    scene_id: int
    scene_name: str
    scene_content: str


# ---------------------------------------------------------
# 内部工具
# ---------------------------------------------------------
def _get_conditions(trigger_row) -> list[dict]:
    """从数据库行提取条件列表。优先 conditions JSON，回退旧字段。"""
    raw = trigger_row["conditions"] if "conditions" in trigger_row.keys() else "[]"
    try:
        conds = json.loads(raw) if raw else []
    except (json.JSONDecodeError, TypeError):
        conds = []
    if conds and isinstance(conds, list):
        return [{"type": c.get("type", ""), "value": c.get("value", "")} for c in conds if c.get("type")]
    ct = trigger_row["cond_type"] if "cond_type" in trigger_row.keys() else ""
    cv = trigger_row["cond_value"] if "cond_value" in trigger_row.keys() else ""
    if ct:
        return [{"type": ct, "value": cv}]
    return []


def _normalize_request_conditions(req) -> str:
    """将请求中的 conditions（或旧字段）统一序列化为 JSON。"""
    conds = []
    if req.conditions:
        conds = [{"type": c.type, "value": c.value} for c in req.conditions if c.type]
    elif req.cond_type:
        conds = [{"type": req.cond_type, "value": req.cond_value}]
    return json.dumps(conds, ensure_ascii=False)


# ---------------------------------------------------------
# 单条件本地判定
# ---------------------------------------------------------
def _judge_single_condition(cond: dict, scene_id: int, chars: list, all_inv: str) -> bool | None:
    """返回 True/False 或 None（需要 AI 判断）。"""
    ctype = cond["type"]
    cval = cond["value"].strip()

    if ctype == "scene":
        try:
            return scene_id == int(cval)
        except ValueError:
            return False

    elif ctype == "item":
        kws = [k.strip().lower() for k in cval.split(",") if k.strip()]
        return any(k in all_inv for k in kws)

    elif ctype == "stat":
        hit = False
        for part in cval.split(","):
            part = part.strip().lower()
            for attr in ["hp", "san"]:
                if part.startswith(attr + "<"):
                    try:
                        threshold = int(part.split("<")[1])
                        if any(c["role"] == "PC" and int(c[attr] or 0) < threshold for c in chars):
                            hit = True
                    except (ValueError, IndexError):
                        pass
        return hit

    elif ctype == "ai":
        return None

    return False


# ---------------------------------------------------------
# AI 批量判断 (KV-Cache + 结构化思维链 + 防弹级容错)
# ---------------------------------------------------------
def _batch_judge_ai(values: list[str], scene_name: str,
                    scene_content: str, conn) -> list[bool]:
    if not values:
        return []

    # 获取系统上下文
    worldview, party_status, relevant_lore, session_memory, \
        l1_context, world_entities_text, rag_context, map_context = \
        _fn_get_system_context(conn, scene_name, scene_content)

    n = len(values)
    conditions_text = "\n".join(f"{i+1}. {v}" for i, v in enumerate(values))

    # ==========================================
    # KV-Cache 拓扑设计：严格的 System/User 角色隔离
    # ==========================================

    # 【静态前缀区】 (映射为 System 角色)
    static_system_prompt = (
        "你是一个严谨的剧情逻辑判定器。\n"
        "你的任务是根据玩家的【最新动作】，判断一组触发条件是否成立。\n\n"
        "--- 基础世界观约束 ---\n"
        f"{worldview}\n\n"
        "--- 判定规则 ---\n"
        "1. 必须【绝对优先】以 User 提供的【核心动作与场景】作为判定依据。即使玩家动作与世界观有冲突，也以玩家动作为准。\n"
        "2. 必须严格返回 JSON。为了保证判定准确，请务必先进行一句话短推理，再输出结果。\n"
        "3. JSON 格式必须严格遵循以下结构：\n"
        "{\n"
        "  \"reasoning\": \"简短的逻辑推演（限50字以内，不要长篇大论）\",\n"
        "  \"results\": [true, false, ...]\n"
        "}\n"
        f"4. results 数组长度必须恰好匹配条件数量 ({n} 个)，顺序严格一致。"
    )

    # 【动态后缀区】 (映射为 User 角色)
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

    try:
        resp = _deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": static_system_prompt},
                {"role": "user", "content": dynamic_user_prompt}
            ],
            temperature=0.1,
            max_tokens=600 + 10 * n, # 大幅度放宽长度
            response_format={"type": "json_object"}
        )
        
        # 【拦截物理截断】
        if resp.choices[0].finish_reason == "length":
            _log.warning("AI 触发器推理过长被截断，启用降级防护。")
            return [False] * n
            
        raw_ai_content = resp.choices[0].message.content
        parsed = json.loads(raw_ai_content)
        
        # 调试开关：排查触发器不准时，可以取消注释下面这行，看 AI 的内心戏
        # _log.info(f"触发器推演逻辑: {parsed.get('reasoning')}")
        
        results = parsed.get("results", [])
        if len(results) != n:
            return [False] * n
        return [bool(r) for r in results]
        
    # 【拦截残缺 JSON 解析崩溃】
    except json.JSONDecodeError as e:
        _log.error(f"AI 返回的 JSON 格式破损: {e}")
        try:
            _log.error(f"破损内容: {raw_ai_content[:100]}...{raw_ai_content[-100:]}")
        except:
            pass
        return [False] * n
    except Exception as e:
        _log.warning(f"AI 触发器批量判断失败: {e}")
        return [False] * n
    

# ---------------------------------------------------------
# AND 门检查
# ---------------------------------------------------------
def _check_trigger_and(trigger_row, scene_id: int, scene_name: str,
                        scene_content: str, chars: list, all_inv: str,
                        ai_cache: dict, conn) -> bool:
    """所有条件都满足才返回 True。ai_cache 跨触发器共享避免重复调用。"""
    conditions = _get_conditions(trigger_row)
    if not conditions:
        return False

    # 收集需要 AI 判断的新条件
    new_ai = [c["value"] for c in conditions
              if c["type"] == "ai" and c["value"] not in ai_cache]
    if new_ai:
        unique = list(dict.fromkeys(new_ai))
        results = _batch_judge_ai(unique, scene_name, scene_content, conn)
        for val, res in zip(unique, results):
            ai_cache[val] = res

    # AND 判定
    for c in conditions:
        if c["type"] == "ai":
            if not ai_cache.get(c["value"], False):
                return False
        else:
            if not _judge_single_condition(c, scene_id, chars, all_inv):
                return False
    return True


# ---------------------------------------------------------
# REST API 端点
# ---------------------------------------------------------
@trigger_router.get("/api/game/triggers")
def get_triggers():
    with safe_db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM triggers ORDER BY id").fetchall()]
    for r in rows:
        try:
            r["conditions"] = json.loads(r.get("conditions", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            r["conditions"] = []
        if not r["conditions"] and r.get("cond_type"):
            r["conditions"] = [{"type": r["cond_type"], "value": r["cond_value"]}]
    return {"status": "success", "triggers": rows}


@trigger_router.post("/api/game/trigger")
def create_trigger(req: TriggerCreateRequest):
    with safe_db() as conn:
        if not conn.execute("SELECT id FROM nodes WHERE id=?", (req.target_node_id,)).fetchone():
            raise fastapi.HTTPException(status_code=400, detail="目标节点不存在")
        conditions_json = _normalize_request_conditions(req)
        conds = json.loads(conditions_json)
        ft = conds[0]["type"] if conds else req.cond_type
        fv = conds[0]["value"] if conds else req.cond_value
        c = conn.execute(
            "INSERT INTO triggers (label, target_node_id, mode, cond_type, cond_value, conditions) VALUES (?,?,?,?,?,?)",
            (req.label[:80], req.target_node_id, req.mode, ft, fv[:200], conditions_json)
        )
        conn.commit()
    return {"status": "success", "id": c.lastrowid}


@trigger_router.put("/api/game/trigger/{tid}")
def update_trigger(tid: int, req: TriggerUpdateRequest):
    with safe_db() as conn:
        if not conn.execute("SELECT id FROM nodes WHERE id=?", (req.target_node_id,)).fetchone():
            raise fastapi.HTTPException(status_code=400, detail="目标节点不存在")
        conditions_json = _normalize_request_conditions(req)
        conds = json.loads(conditions_json)
        ft = conds[0]["type"] if conds else req.cond_type
        fv = conds[0]["value"] if conds else req.cond_value
        conn.execute(
            "UPDATE triggers SET label=?, target_node_id=?, mode=?, cond_type=?, cond_value=?, conditions=?, fired=0 WHERE id=?",
            (req.label[:80], req.target_node_id, req.mode, ft, fv[:200], conditions_json, tid)
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
        conn.execute("UPDATE triggers SET fired=0 WHERE id=?", (tid,))
        conn.commit()
    return {"status": "success"}


@trigger_router.post("/api/game/check-triggers")
def check_triggers(req: CheckTriggersRequest):
    """检查所有未触发的触发器（AND 多条件）。"""
    conn = get_db_connection()
    try:
        pending = conn.execute("SELECT * FROM triggers WHERE fired=0").fetchall()
        if not pending:
            conn.close()
            return {"status": "success", "fired": []}

        chars = conn.execute("SELECT * FROM characters").fetchall()
        all_inv = " ".join([(c["inventory"] or "").lower() for c in chars])
        ai_cache = {}

        fired_results = []
        for t in pending:
            if _check_trigger_and(t, req.scene_id, req.scene_name,
                                   req.scene_content, chars, all_inv, ai_cache, conn):
                conn.execute("UPDATE triggers SET fired=1 WHERE id=?", (t["id"],))
                target_node = conn.execute(
                    "SELECT * FROM nodes WHERE id=?", (t["target_node_id"],)
                ).fetchone()
                fired_results.append({
                    "trigger_id":       t["id"],
                    "label":            t["label"],
                    "mode":             t["mode"],
                    "target_node_id":   t["target_node_id"],
                    "target_node_name": target_node["name"] if target_node else "未知节点",
                })
                if _fn_append_to_memory:
                    _fn_append_to_memory(conn, f"关键触发器「{t['label']}」已触发，剧情指向节点[{t['target_node_id']}]。")

        conn.commit()
        conn.close()
        return {"status": "success", "fired": fired_results}
    except Exception as e:
        conn.close()
        raise
