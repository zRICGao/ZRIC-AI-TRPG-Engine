"""
Z.R.I.C 引擎 — AI Agent 模块 (agent.py)
包含：多模型管理（DeepSeek / Claude Opus 4.6）、system prompt 构建、
SSE 流式推演端点、结果后处理（NPC/stat/map/entity）。
由 main.py 通过 app.include_router(agent_router) 挂载。
"""

import json
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from logger import get_logger

_log = get_logger("agent")

agent_router = APIRouter(tags=["AI Agent"])

##########################################################
# 将此请求模型添加到 agent.py 顶部
class NPCChatRequest(BaseModel):
    npc_name: str
    player_message: str
    chat_history: str = "" # 前端传来的近期聊天记录
###########################################################

# ---------------------------------------------------------
# state_desc JSON 解析器（兼容旧版纯文本）
# ---------------------------------------------------------
def _parse_state_desc(raw: str) -> dict:
    """
    解析 world_entities.state_desc 字段。
    新版：JSON 格式 {"desc":"...", "emotion":{...}, "breakpoint":{...}, "memory":[...]}
    旧版：纯文本 → 自动升级为 {"desc": "原文本", "emotion": {...}, "memory": []}
    """
    if not raw:
        return {"desc": "", "emotion": {"trust": 0, "fear": 0, "irritation": 0}, "memory": []}
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            sd = json.loads(raw)
            sd.setdefault("desc", "")
            sd.setdefault("emotion", {"trust": 0, "fear": 0, "irritation": 0})
            sd.setdefault("memory", [])
            return sd
        except json.JSONDecodeError:
            pass
    # 旧版纯文本兼容
    return {"desc": raw, "emotion": {"trust": 0, "fear": 0, "irritation": 0}, "memory": []}


def _serialize_state_desc(sd: dict) -> str:
    """将 state_desc dict 序列化为 JSON 字符串。"""
    return json.dumps(sd, ensure_ascii=False)


def _format_emotion_label(value: int) -> str:
    """将情绪数值转为人类可读标签。"""
    if value >= 60: return "极高"
    if value >= 30: return "高"
    if value >= 10: return "中"
    if value >= -10: return "平"
    if value >= -30: return "低"
    return "极低"

# ---------------------------------------------------------
# 依赖注入（由 main.py 启动时设置）
# ---------------------------------------------------------
_db_file: str = ""
_deepseek_client = None   # OpenAI-compatible (DeepSeek)
_anthropic_client = None  # Anthropic native SDK
_persona_config: dict = {}

# 外部函数引用（由 main.py 注入，避免循环导入）
_get_map_context = None
_auto_place_room = None
_process_map_actions = None
_get_current_room_id = None
_ai_extract_and_upsert_entities = None
_build_persona_instruction = None
_l1_append = None
_l1_get_working_context = None
_append_to_memory = None
_tl_append_memory = None
_get_world_entities_text = None
_rag_retrieve = None
_get_embeddings = None          # rag.get_embeddings（NPC 记忆→L3 向量化用）
_refresh_vector_cache = None    # rag.refresh_vector_cache

# 当前激活的模型（服务器级默认值，可被请求级参数覆盖）
AVAILABLE_MODELS = {
    "deepseek": {"label": "DeepSeek Chat", "model_id": "deepseek-chat"},
    "claude-opus": {"label": "Claude Opus 4.6", "model_id": "claude-opus-4-6"},
}
_active_model: str = "deepseek"  # 服务器默认模型


def _resolve_model(request_model: str | None = None) -> str:
    """
    解析当前请求应使用的模型。
    优先级：请求体中的 model 参数 > 服务器全局默认值。
    若指定的模型不可用（如 Claude 未配置），回退到 deepseek。
    """
    model = request_model if request_model and request_model in AVAILABLE_MODELS else _active_model
    if model == "claude-opus" and not _anthropic_client:
        _log.debug("请求指定 claude-opus 但未配置，回退到 deepseek")
        return "deepseek"
    return model


def configure_agent(
    db_file: str,
    deepseek_client,
    anthropic_api_key: str = "",
    persona_config: dict = None,
    # 外部函数注入
    fn_get_map_context=None,
    fn_auto_place_room=None,
    fn_process_map_actions=None,
    fn_get_current_room_id=None,
    fn_ai_extract_and_upsert_entities=None,
    fn_build_persona_instruction=None,
    fn_l1_append=None,
    fn_l1_get_working_context=None,
    fn_append_to_memory=None,
    fn_tl_append_memory=None,
    fn_get_world_entities_text=None,
    fn_rag_retrieve=None,
    fn_get_embeddings=None,
    fn_refresh_vector_cache=None,
):
    """由 main.py 启动时调用，注入所有依赖。"""
    global _db_file, _deepseek_client, _anthropic_client, _persona_config
    global _get_map_context, _auto_place_room, _process_map_actions
    global _get_current_room_id, _ai_extract_and_upsert_entities
    global _build_persona_instruction, _l1_append, _l1_get_working_context
    global _append_to_memory, _tl_append_memory, _get_world_entities_text, _rag_retrieve
    global _get_embeddings, _refresh_vector_cache

    _db_file = db_file
    _deepseek_client = deepseek_client
    _persona_config = persona_config or {}

    # Anthropic Claude client (optional)
    if anthropic_api_key:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            _log.info("Claude Opus 模型已启用")
        except ImportError:
            _anthropic_client = None
            _log.warning("anthropic 包未安装，Claude Opus 模型不可用（pip install anthropic）")

    # 注入外部函数
    _get_map_context = fn_get_map_context
    _auto_place_room = fn_auto_place_room
    _process_map_actions = fn_process_map_actions
    _get_current_room_id = fn_get_current_room_id
    _ai_extract_and_upsert_entities = fn_ai_extract_and_upsert_entities
    _build_persona_instruction = fn_build_persona_instruction
    _l1_append = fn_l1_append
    _l1_get_working_context = fn_l1_get_working_context
    _append_to_memory = fn_append_to_memory
    _tl_append_memory = fn_tl_append_memory
    _get_world_entities_text = fn_get_world_entities_text
    _rag_retrieve = fn_rag_retrieve
    _get_embeddings = fn_get_embeddings
    _refresh_vector_cache = fn_refresh_vector_cache


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
# 模型切换 API
# ---------------------------------------------------------
class ModelSwitchRequest(BaseModel):
    model: str  # "deepseek" | "claude-opus"

@agent_router.get("/api/ai/models")
def list_models():
    """列出可用模型及当前激活模型。"""
    models = []
    for key, info in AVAILABLE_MODELS.items():
        available = True
        if key == "claude-opus" and not _anthropic_client:
            available = False
        models.append({**info, "key": key, "available": available})
    return {
        "status": "success",
        "models": models,
        "active": _active_model,
    }

@agent_router.post("/api/ai/models/switch")
def switch_model(req: ModelSwitchRequest):
    """切换当前激活的 AI 模型。"""
    global _active_model
    if req.model not in AVAILABLE_MODELS:
        return {"status": "error", "message": f"未知模型：{req.model}"}
    if req.model == "claude-opus" and not _anthropic_client:
        return {"status": "error", "message": "Claude Opus 未配置 API Key（需安装 anthropic 包并配置 ANTHROPIC_API_KEY）"}
    _active_model = req.model
    return {"status": "success", "active": _active_model, "label": AVAILABLE_MODELS[_active_model]["label"]}


# ---------------------------------------------------------
# 统一 AI 调用接口（支持 DeepSeek / Claude，自动路由）
# ---------------------------------------------------------
def _call_ai(system_prompt: str, user_prompt: str,
             temperature: float = 0.8, max_tokens: int = 2000,
             json_mode: bool = True,
             model_override: str | None = None) -> str:
    """
    统一调用 AI 模型，返回完整文本。
    model_override：请求级模型覆盖（优先于服务器全局默认值）。
    """
    model = _resolve_model(model_override)
    if model == "claude-opus" and _anthropic_client:
        return _call_claude(system_prompt, user_prompt, temperature, max_tokens)
    else:
        return _call_deepseek(system_prompt, user_prompt, temperature, max_tokens, json_mode)


def _call_deepseek(system_prompt: str, user_prompt: str,
                   temperature: float, max_tokens: int,
                   json_mode: bool = True) -> str:
    """调用 DeepSeek Chat API。"""
    if not _deepseek_client:
        raise RuntimeError("DeepSeek client 未初始化。请检查 API Key 配置和 configure_agent 是否已执行。")

    kwargs = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,  # 防止超长等待
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = _deepseek_client.chat.completions.create(**kwargs)
        result = response.choices[0].message.content
        if not result:
            raise ValueError("DeepSeek 返回了空内容")
        return result.strip()
    except Exception as e:
        raise RuntimeError(f"DeepSeek API 调用失败: {type(e).__name__}: {str(e)}")


def _call_claude(system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int) -> str:
    """调用 Anthropic Claude Opus 4.6 API。"""
    response = _anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    # Claude 返回 content blocks
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text
    return text.strip()


# ---------------------------------------------------------
# SSE 流式调用（用于前端打字机效果）
# ---------------------------------------------------------
def _stream_ai_sse(system_prompt: str, user_prompt: str,
                   temperature: float = 0.8, max_tokens: int = 2000,
                   model_override: str | None = None):
    """
    生成器：流式调用 AI，yield SSE 格式的 data 行。
    支持 DeepSeek (stream=True) 和 Claude (messages.stream)。
    最终 yield 一个 [DONE] 事件。
    """
    full_text = ""
    model = _resolve_model(model_override)

    if model == "claude-opus" and _anthropic_client:
        # Claude 流式
        try:
            with _anthropic_client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            ) as stream:
                for text_chunk in stream.text_stream:
                    full_text += text_chunk
                    yield f"data: {json.dumps({'type': 'text', 'content': text_chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
    else:
        # DeepSeek 流式
        try:
            response = _deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    full_text += text_chunk
                    yield f"data: {json.dumps({'type': 'text', 'content': text_chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    # 最终事件：完整文本 + DONE 标记
    yield f"data: {json.dumps({'type': 'done', 'full_text': full_text}, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------
# 上下文构建器
# ---------------------------------------------------------
def build_system_context(conn, *search_texts):
    """
    打包提取全知上下文（世界观、角色、百科、记忆、实体、RAG、地图）。
    返回 8 元组。
    """
    wv_row = conn.execute("SELECT value FROM system_state WHERE key = 'worldview'").fetchone()
    worldview = wv_row["value"] if wv_row else ""

    mem_row = conn.execute("SELECT value FROM system_state WHERE key = 'session_memory'").fetchone()
    session_memory = mem_row["value"] if mem_row else ""

    l1_context = _l1_get_working_context(conn) if _l1_get_working_context else ""

    chars = conn.execute("SELECT name, role, hp, san, inventory, status FROM characters").fetchall()
    active_chars = [c for c in chars if (c["status"] or "active") == "active"]
    benched_chars = [c for c in chars if (c["status"] or "") == "benched"]
    party_status = "\n".join([
        f"- {c['name']} (HP:{c['hp']}, SAN:{c['san']}) | 状态/物品: {c['inventory'] or '无'}"
        for c in active_chars
    ])
    if benched_chars:
        party_status += "\n【暂离角色（不在当前场景，不可直接互动）】\n" + "\n".join([
            f"- {c['name']}（暂离）" for c in benched_chars
        ])
    if not party_status.strip():
        party_status = "当前场景无活跃角色。"

    combined_text = " ".join(search_texts).lower()
    lores = conn.execute("SELECT keywords, content FROM lorebook").fetchall()
    injected_lore = []
    for lore in lores:
        keywords = [k.strip().lower() for k in lore["keywords"].split(",") if k.strip()]
        if any(k in combined_text for k in keywords):
            injected_lore.append(f"[{lore['keywords']}]: {lore['content']}")
    relevant_lore = "\n".join(injected_lore) if injected_lore else "无"

    world_entities_text = _get_world_entities_text(conn, *search_texts) if _get_world_entities_text else ""

    # RAG 检索（地图驱动增强）
    rag_context = ""
    try:
        rag_query = " ".join(search_texts)
        room_row = conn.execute("SELECT value FROM system_state WHERE key='current_room_id'").fetchone()
        current_room_id = int(room_row["value"]) if room_row and room_row["value"] else None
        if current_room_id:
            room = conn.execute("SELECT label, description FROM map_rooms WHERE id=?", (current_room_id,)).fetchone()
            if room and (room["label"] or room["description"]):
                rag_query = f"{rag_query} {room['label']} {room['description']}".strip()
        if _rag_retrieve:
            rag_context = _rag_retrieve(conn, rag_query)
    except Exception as e:
        _log.debug("RAG 检索上下文构建失败（已降级）: %s", e)

    # 地图空间感知
    map_context = ""
    try:
        if _get_map_context:
            map_context = _get_map_context(conn, current_room_id)
    except Exception as e:
        _log.debug("地图上下文构建失败（已降级）: %s", e)

    return worldview, party_status, relevant_lore, session_memory, l1_context, world_entities_text, rag_context, map_context


# ---------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------
# ─────────────────────────────────────────────────────────
# JSON Schema（优化版：60 字短叙事 + null 字段省略 + 对话精简）
#
# 设计原则：
#   1. 60 字短叙事：node_content 限制为约 60 字的核心场景骨架，
#      保留因果链和细节锚点，玩家选择后由 expand-branch 润色为完整叙事。
#   2. null 字段省略：无变化的字段直接不输出。
#   3. 对话模式精简：砍掉 stat_changes / map_actions / npc 字段。
# ─────────────────────────────────────────────────────────

_DYNAMIC_OPTIONS_JSON_SCHEMA = """{
  "thought_process": "1. ... 2. ... 3. ... 4. ... 5. ...（每步限一句话，总共不超过100字）",
  "branches": [
    {
      "text": "选项简述（20字内）",
      "node_name": "新场景标题（15字内）",
      "node_content": "场景骨架（约60字，包含关键动作、感官反馈、情绪反应等核心锚点，仅当必要时才补充环境描写）",
      "stat_changes": [{"char_name": "角色名", "hp_delta": 0, "san_delta": 0, "inventory_append": "", "inventory_remove": "", "reason": "原因(10字内)"}],
      "entity_updates": "该分支导致的实体状态变化",
      "emotion_deltas": [{"npc_name": "NPC名字", "trust": 0, "fear": 0, "irritation": 0}],
      "npc_memories": [{"npc_name": "NPC名字", "memory": "从该NPC视角记住的一句话（20字内）"}],
      "map_actions": {
        "movement": {"target_room_label": "相邻房间名称", "reason": "移动原因"},
        "new_room": {"label": "新发现的房间名(10字内)", "description": "新房间描述(30字内)"},
        "unlock_edge": {"from_label": "房间A名称", "to_label": "房间B名称", "key_used": "使用的钥匙名"}
      },
      "npc": {"name": "姓名(10字内)", "role": "NPC", "hp": 50, "san": 60, "inventory": "持有物(30字内)", "backstory": "背景(50字内)"}
    }
  ]
}
【极重要·字段省略规则】：
- 若某分支没有 stat_changes，则省略整个 key
- 若某分支没有 entity_updates / emotion_deltas / npc_memories，则省略
- 若无空间变化，省略 map_actions；若无新 NPC，省略 npc
- 只保留有实际内容的字段"""

# 【对话模式】JSON Schema — 砍掉 stat_changes / map_actions / npc，强化情绪和记忆
_DIALOGUE_JSON_SCHEMA = """{
  "thought_process": "1. 社交意图识别：... 2. NPC性格反应：... 3. 信息博弈：... 4. 关系位移：...（每步限一句话）",
  "branches": [
    {
      "text": "对话走向简述（20字内）",
      "node_name": "场景标题（15字内）",
      "node_content": "对话场景骨架（约60字，含NPC关键台词用「」包裹、表情/肢体语言）",
      "san_delta": {"char_name": "角色名", "delta": -5, "reason": "心理冲击原因"},
      "entity_updates": "NPC态度变化描述",
      "emotion_deltas": [{"npc_name": "NPC名字", "trust": 10, "fear": 0, "irritation": -5}],
      "npc_memories": [{"npc_name": "NPC名字", "memory": "从该NPC视角记住的一句话"}]
    }
  ]
}
【极重要·字段省略规则】：
- 对话模式下不允许出现 stat_changes / map_actions / npc 字段
- 若无 SAN 变化，省略 san_delta；其余同上
- 只保留有实际内容的字段"""

# 【行动模式】JSON Schema — 与混合模式相同
_ACTION_JSON_SCHEMA = _DYNAMIC_OPTIONS_JSON_SCHEMA

# ---------------------------------------------------------
# 三套推演指令模板
# ---------------------------------------------------------
_INSTRUCTION_DIALOGUE = """
=== 推演指令（社交对话模式）===
这是一次社交互动。玩家正在通过言语与场景中的角色交涉。

【核心约束】
- 不得产生任何物理伤害（hp_delta 必须为 0）
- 不得改变物品持有状态（inventory_append / inventory_remove 必须为空）
- 不得触发空间移动（map_actions 必须为 null）
- SAN 值可以因心理冲击而变化（例如得知可怕真相）
- entity_updates 应聚焦于 NPC 态度、情绪、信任度的变化

【情绪状态机】
每个分支必须输出 emotion_deltas 数组，记录该分支对在场 NPC 情绪的影响：
- trust（信任）：正值=更信任玩家，负值=更不信任
- fear（恐惧）：正值=更害怕，负值=放松
- irritation（烦躁）：正值=更恼火，负值=缓和
注意：单次 delta 绝对值通常在 1-15 之间，只有极端事件（生死威胁/深层创伤被触及）才超过 25。

【NPC 记忆】
每个分支必须输出 npc_memories 数组，用一句话从 NPC 的第一人称视角记录本次互动：
- 只记录该 NPC 亲眼所见/亲耳所闻的事，不能记录 NPC 不可能知道的信息
- 20字以内，例如："有人来打听地下室的事"、"那个法师送了我一块怀表"

【thought_process 必须包含以下 4 步推理】
1. 社交意图识别：玩家这句话的真实目的是什么？（威吓/套话/说服/欺骗/安慰）
2. NPC 性格反应：根据该 NPC 的性格特质，ta 会如何接招？（抗拒/动摇/配合/反击）
3. 信息博弈：这次对话是否会泄露/获取关键信息？哪些信息是 NPC 会守住的底线？
4. 关系位移：对话结束后，NPC 对玩家的态度会发生什么变化？

推演 2-4 个对话走向分支。每个分支的 node_content（约60字）必须包含 NPC 的关键台词（用「」包裹）和核心肢体语言。
【极其重要】每个分支的 entity_updates / emotion_deltas / npc_memories 必须写在该分支对象内部。不同对话走向可能导致不同的情绪后果。
【对话模式禁止字段】不要输出 stat_changes / map_actions / npc 字段。SAN 变化使用 san_delta 字段。"""

_INSTRUCTION_ACTION = """
=== 推演指令（物理行动模式）===
这是一次物理行动。玩家正在执行涉及身体、环境或物品的操作。

【核心约束】
- HP/SAN 变化、物品得失、空间移动均可正常发生
- NPC 的言语反应应简短，重点描述其物理行为（拔剑、逃跑、倒地）
- 若行动涉及空间移动，map_actions.movement 的目标必须在【地图空间感知】的相邻列表中
- 若发现新区域填 new_room，解锁通道填 unlock_edge

【thought_process 必须包含以下 5 步推理】
1. 可行性检查：角色当前 HP/SAN/物品是否支持此行动？
2. 世界观合规：此行动是否违反物理法则或世界规则？
3. 空间合规：行动是否发生在当前位置可到达的范围内？
4. 物理后果：环境和在场角色会受到什么直接物理影响？
5. 连锁反应：此行动会触发哪些后续事件？

推演 2-4 个结果分支。每个分支的 node_content（约60字）必须包含关键动作描写、伤害程度和环境变化等核心锚点。
【极其重要】每个分支的 stat_changes / entity_updates / map_actions / npc 必须写在该分支对象内部。不同结果可能导致完全不同的HP变化和空间移动。
【字段省略】没有变化的字段直接省略，不要写空数组或 null。"""

_INSTRUCTION_MIXED = """
=== 推演指令 ===
玩家做出了意外举动。你必须先在 thought_process 字段中进行严格的逻辑自检（内省），然后再输出推演结果。

【thought_process 必须包含以下 5 步推理（每步一句话）】
1. 角色状态检查：当前行动者的 HP/SAN/负面状态是否限制了此行动？
2. 世界观合规：此行动是否违反世界观中的物理法则、禁忌或社会规则？
3. 空间合规：若地图已启用，行动是否发生在当前位置可到达的范围内？
4. NPC 动机分析：场景中的 NPC 会如何反应？他们的动机是什么？
5. 后果推演：综合以上分析，合理的结局应当是什么？

完成以上内省后，推演 2-4 个合理分支。每个分支的 node_content（约60字）应包含核心因果描写。

【极其重要】每个分支的副作用字段必须写在该分支对象内部，而不是全局字段。
不同分支可能导致完全不同的后果（例如分支A扣HP，分支B不扣），系统会在玩家选择后只执行该分支的副作用。
若行动导致空间移动，map_actions.movement 的目标必须在【地图空间感知】的相邻列表中。
若发现新区域填 new_room，解锁通道填 unlock_edge。
【字段省略】没有变化的字段直接省略，不要写空数组或 null。"""


def _build_dynamic_system_prompt(worldview, party_status, relevant_lore,
                                  session_memory, l1_context,
                                  world_entities_text, rag_context, map_context,
                                  is_timeline=False, tl_label="",
                                  action_type="mixed",
                                  gm_correction=""):
    """构建推演用的 system prompt。根据 action_type 切换上下文策略和指令集。"""
    prefix = f"你是跑团GM，正在处理分叉时间线「{tl_label}」中的剧情推演。" if is_timeline else "你是一个跑团GM。"
    persona = ""
    if _build_persona_instruction:
        persona = _build_persona_instruction()

    # GM 纠正指令块（非空时注入到 prompt 末尾）
    gm_correction_block = ""
    if gm_correction and gm_correction.strip():
        gm_correction_block = f"""

=== 【GM 纠正指令——最高优先级，必须遵守】===
GM 对上一次推演结果不满意，明确要求你在本次推演中：
{gm_correction.strip()}
你必须严格执行此纠正指令。如果 GM 的纠正与世界观冲突，以 GM 指令为准。"""

    # ── 根据 action_type 选择上下文策略 ──
    if action_type == "dialogue":
        # 对话模式：砍地图、强化 persona
        context_block = f"""{prefix}
你必须严格遵守以下信息层级，绝不能编造或忽视任何已设定的内容。

=== 第一优先级：硬性约束（违反即为错误）===
【全局世界观——所有推演必须在此框架内，不得自行发明世界观外的设定】
{worldview}

【角色当前状态——推演时必须参考，尤其关注 SAN 值和性格特征】
{party_status}

{f"【已知世界实体（L2 实体档案）——必须遵守其当前状态，尤其注意 NPC 的性格和态度】{chr(10)}{world_entities_text}" if world_entities_text else ""}

=== 第二优先级：上下文参考 ===
{f"【L1 近期推演快照】{chr(10)}{l1_context}" if l1_context else ""}

【剧情记忆流】
{session_memory}

【相关设定词条】
{relevant_lore}
{f"【知识库检索结果（L3 长期记忆 + 背景设定）】{chr(10)}{rag_context}" if rag_context else ""}

=== 性格交互协议（对话模式强化）===
{persona if persona else "根据 NPC 的已有设定推演其社交反应，注意其说话方式、肢体语言和心理状态的一致性。"}

{_INSTRUCTION_DIALOGUE}

必须返回严格 JSON（thought_process 为第一个字段）：
{_DIALOGUE_JSON_SCHEMA}{gm_correction_block}"""

    elif action_type == "action":
        # 行动模式：强化地图、砍 persona
        context_block = f"""{prefix}
你必须严格遵守以下信息层级，绝不能编造或忽视任何已设定的内容。

=== 第一优先级：硬性约束（违反即为错误）===
【全局世界观——所有推演必须在此框架内，不得自行发明世界观外的设定】
{worldview}

【角色当前状态——推演时必须参考，HP/SAN/物品必须与此一致】
{party_status}

{f"【已知世界实体（L2 实体档案）——必须遵守其当前状态，不得矛盾】{chr(10)}{world_entities_text}" if world_entities_text else ""}
{f"【地图空间感知——AI必须遵守此空间结构推演，不可凭空创造不存在的房间或通道】{chr(10)}{map_context}" if map_context else ""}

=== 第二优先级：上下文参考 ===
{f"【L1 近期推演快照】{chr(10)}{l1_context}" if l1_context else ""}

【剧情记忆流】
{session_memory}

【相关设定词条】
{relevant_lore}
{f"【知识库检索结果（L3 长期记忆 + 背景设定）】{chr(10)}{rag_context}" if rag_context else ""}

{_INSTRUCTION_ACTION}

必须返回严格 JSON（thought_process 为第一个字段）：
{_ACTION_JSON_SCHEMA}{gm_correction_block}"""

    else:
        # 混合模式：沿用原逻辑
        context_block = f"""{prefix}
你必须严格遵守以下信息层级，绝不能编造或忽视任何已设定的内容。

=== 第一优先级：硬性约束（违反即为错误）===
【全局世界观——所有推演必须在此框架内，不得自行发明世界观外的设定】
{worldview}

【角色当前状态——推演时必须参考，HP/SAN/物品必须与此一致】
{party_status}

{f"【已知世界实体（L2 实体档案）——必须遵守其当前状态，不得矛盾】{chr(10)}{world_entities_text}" if world_entities_text else ""}
{f"【地图空间感知——AI必须遵守此空间结构推演，不可凭空创造不存在的房间或通道】{chr(10)}{map_context}" if map_context else ""}

=== 第二优先级：上下文参考 ===
{f"【L1 近期推演快照】{chr(10)}{l1_context}" if l1_context else ""}

【剧情记忆流】
{session_memory}

【相关设定词条】
{relevant_lore}
{f"【知识库检索结果（L3 长期记忆 + 背景设定）】{chr(10)}{rag_context}" if rag_context else ""}
{persona}

{_INSTRUCTION_MIXED}

必须返回严格 JSON（thought_process 为第一个字段）：
{_DYNAMIC_OPTIONS_JSON_SCHEMA}{gm_correction_block}"""

    return context_block


# ---------------------------------------------------------
# 推演结果后处理（NPC / stat / map / entity / memory）
# ---------------------------------------------------------
def _post_process_dynamic_result(conn, parsed: dict, scene_name: str,
                                  player_action: str, current_node_id: int,
                                  timeline_id: int | None = None,
                                  tl_id_for_memory: int | None = None,
                                  action_type: str = "mixed"):
    """
    Phase 1：推演完成后立刻执行。
    AI 返回 60 字短叙事 node_content + 副作用，创建节点和选项。
    副作用序列化存储到 pending_effects 表，等玩家选择后执行。

    前端调用链：
      Phase 1: 推演 → _post_process（创建节点 + pending_effects）
      Phase 3: expand-branch（读取 pending_effects + 60字骨架，润色为完整叙事）
      Phase 2: apply-branch-effects（执行副作用，删除 pending_effects）
    """
    thought_process = str(parsed.get("thought_process", "")) if isinstance(parsed, dict) else ""

    branches = parsed.get("branches", []) if isinstance(parsed, dict) else []
    if not branches:
        branches = next((v for v in parsed.values() if isinstance(v, list)), []) if isinstance(parsed, dict) else []
    if not branches:
        raise ValueError("AI 未返回有效的分支数据")

    if not conn.execute("SELECT id FROM nodes WHERE id = ?", (current_node_id,)).fetchone():
        raise ValueError("节点不存在")

    # ── 兼容旧版全局字段：如果 branches 里没有 per-branch 副作用，从全局字段回填 ──
    global_stat_changes = parsed.get("stat_changes") if isinstance(parsed, dict) else None
    global_entity_updates = parsed.get("entity_updates", "") if isinstance(parsed, dict) else ""
    global_map_actions = parsed.get("map_actions") if isinstance(parsed, dict) else None
    global_npc = parsed.get("npc") if isinstance(parsed, dict) else None

    new_options = []
    for branch in branches:
        n_name = branch.get("node_name", "未知节点")[:50]
        n_content = branch.get("node_content", "")[:1000]
        o_text = branch.get("text", n_name)[:100]

        # ── 收集副作用（优先 per-branch，fallback 到全局；省略的字段默认为空） ──
        b_stat_changes = branch.get("stat_changes", global_stat_changes)
        b_entity_updates = branch.get("entity_updates") or global_entity_updates or ""
        b_map_actions = branch.get("map_actions", global_map_actions)
        b_npc = branch.get("npc", global_npc)
        b_emotion_deltas = branch.get("emotion_deltas") or []
        b_npc_memories = branch.get("npc_memories") or []

        # ── 对话模式守卫 ──
        if action_type == "dialogue":
            san_delta_obj = branch.get("san_delta")
            if san_delta_obj and isinstance(san_delta_obj, dict):
                b_stat_changes = [{
                    "char_name": san_delta_obj.get("char_name", ""),
                    "hp_delta": 0,
                    "san_delta": int(san_delta_obj.get("delta", 0)),
                    "inventory_append": "", "inventory_remove": "",
                    "reason": san_delta_obj.get("reason", ""),
                }]
            elif b_stat_changes and isinstance(b_stat_changes, list):
                for c in b_stat_changes:
                    c["hp_delta"] = 0
                    c["inventory_append"] = ""
                    c["inventory_remove"] = ""
            b_map_actions = None
            b_npc = None

        fx_payload = {
            "stat_changes": b_stat_changes or [],
            "entity_updates": b_entity_updates or "",
            "emotion_deltas": b_emotion_deltas or [],
            "npc_memories": b_npc_memories or [],
            "map_actions": b_map_actions,
            "npc": b_npc,
            "action_type": action_type,
            "timeline_id": timeline_id,
            "tl_id_for_memory": tl_id_for_memory,
            "scene_name": scene_name,
            "player_action": player_action,
            "thought_process": thought_process,
        }

        summary_text = o_text[:100]

        # 60字短叙事：content 已含核心骨架，玩家选择后由 expand-branch 润色扩写
        cursor = conn.execute("INSERT INTO nodes (name, summary, content) VALUES (?, ?, ?)",
                              (n_name, summary_text, n_content))
        n_id = cursor.lastrowid

        conn.execute(
            "INSERT OR REPLACE INTO pending_effects (node_id, payload) VALUES (?, ?)",
            (n_id, json.dumps(fx_payload, ensure_ascii=False))
        )

        conn.execute("INSERT INTO options (node_id, text, next_node_id) VALUES (?, ?, ?)",
                     (current_node_id, o_text, n_id))
        new_options.append({
            "text": o_text, "next_node_id": n_id,
            "node_name": n_name, "node_content": n_content,
            "pending_stat_changes": b_stat_changes or [],
            "pending_entity_updates": b_entity_updates or "",
            "pending_map_actions": b_map_actions,
            "pending_npc": b_npc,
        })

    # L1 写入（记录推演发生，结果待定）
    type_tag = {"dialogue": "🗣️", "action": "🏃", "mixed": "🎭"}.get(action_type, "")
    ai_summary = " / ".join(b.get("text", "") for b in branches)[:300]
    if _l1_append:
        prefix = f"{type_tag}{scene_name}" if not tl_id_for_memory else f"{type_tag}TL{tl_id_for_memory}:{scene_name}"
        _l1_append(conn, prefix, player_action, f"[待选择] {ai_summary}", thought_process, "", timeline_id=tl_id_for_memory)

    conn.commit()
    return new_options, None, [], {"moved_to": None, "new_room": None, "unlocked_edge": None, "errors": []}, thought_process, ""


# ---------------------------------------------------------
# Phase 2：玩家选择分支后，执行该分支的副作用
# ---------------------------------------------------------
class ApplyBranchEffectsRequest(BaseModel):
    node_id: int  # 玩家选择跳转到的目标节点 ID

@agent_router.post("/api/ai/apply-branch-effects")
def apply_branch_effects(req: ApplyBranchEffectsRequest):
    """
    当玩家点击某个 AI 生成的分支选项时，前端调用此接口。
    从 pending_effects 表中提取该节点的副作用 payload 并执行。
    执行完后删除记录（防止重复执行）。
    """
    conn = get_db_connection()
    try:
        node = conn.execute("SELECT * FROM nodes WHERE id=?", (req.node_id,)).fetchone()
        if not node:
            conn.close()
            return {"status": "skipped", "message": "节点不存在"}

        # 从独立表读取副作用（兼容旧版 __FX__ 机制）
        fx_row = conn.execute(
            "SELECT id, payload FROM pending_effects WHERE node_id=?", (req.node_id,)
        ).fetchone()

        if fx_row:
            # 新版：从 pending_effects 表读取
            try:
                fx = json.loads(fx_row["payload"])
            except json.JSONDecodeError:
                conn.close()
                return {"status": "error", "message": "副作用数据损坏"}
        else:
            # 兼容旧版：尝试从 nodes.summary 的 __FX__ 前缀读取
            summary = node["summary"] or ""
            if "__FX__" not in summary:
                conn.close()
                return {"status": "skipped", "message": "无待执行副作用"}
            fx_json = summary.split("__FX__", 1)[1]
            try:
                fx = json.loads(fx_json)
            except json.JSONDecodeError:
                conn.close()
                return {"status": "error", "message": "副作用数据损坏（旧版格式）"}

        action_type = fx.get("action_type", "mixed")
        timeline_id = fx.get("timeline_id")
        tl_id_for_memory = fx.get("tl_id_for_memory")
        scene_name = fx.get("scene_name", "")
        player_action = fx.get("player_action", "")
        thought_process = fx.get("thought_process", "")

        applied_changes = []
        spawned_npc = None
        map_result = {"moved_to": None, "new_room": None, "unlocked_edge": None, "errors": []}
        entity_updates_text = str(fx.get("entity_updates", ""))

        # ── 执行 stat_changes ──
        raw_changes = fx.get("stat_changes")
        if raw_changes and isinstance(raw_changes, list):
            all_chars = conn.execute("SELECT * FROM characters").fetchall()
            for change in raw_changes:
                char_name = change.get("char_name", "").strip()
                matched = next((c for c in all_chars if c["name"] == char_name), None)
                if not matched:
                    continue
                new_hp = max(0, int(matched["hp"] or 0) + int(change.get("hp_delta", 0)))
                new_san = max(0, int(matched["san"] or 0) + int(change.get("san_delta", 0)))
                inv = matched["inventory"] or ""
                append_inv = change.get("inventory_append", "").strip()
                remove_inv = change.get("inventory_remove", "").strip()
                if append_inv:
                    inv = (inv + ", " + append_inv).strip(", ")
                if remove_inv:
                    inv = ", ".join(p.strip() for p in inv.split(",")
                                   if remove_inv.lower() not in p.lower()).strip(", ")
                conn.execute("UPDATE characters SET hp=?, san=?, inventory=? WHERE id=?",
                             (new_hp, new_san, inv, matched["id"]))
                applied_changes.append({
                    "char_name": char_name,
                    "hp_delta": int(change.get("hp_delta", 0)),
                    "san_delta": int(change.get("san_delta", 0)),
                    "inventory_append": append_inv, "inventory_remove": remove_inv,
                    "reason": change.get("reason", ""),
                    "new_hp": new_hp, "new_san": new_san, "new_inv": inv
                })

        # ── 执行 NPC 生成 ──
        npc_data = fx.get("npc")
        if npc_data and isinstance(npc_data, dict) and npc_data.get("name"):
            npc_name = npc_data.get("name", "神秘人")[:50]
            npc_hp = int(npc_data.get("hp", 50))
            npc_san = int(npc_data.get("san", 60))
            npc_inv = npc_data.get("inventory", "")[:200]
            npc_back = npc_data.get("backstory", "")
            cur = conn.execute(
                "INSERT INTO characters (name, role, hp, san, inventory) VALUES (?, 'NPC', ?, ?, ?)",
                (npc_name, npc_hp, npc_san, npc_inv))
            spawned_npc = {"id": cur.lastrowid, "name": npc_name, "role": "NPC",
                           "hp": npc_hp, "san": npc_san, "inventory": npc_inv, "backstory": npc_back}
            mem_text = f"NPC [{npc_name}] 登场于 [{scene_name}]。{npc_back}"
            if tl_id_for_memory and _tl_append_memory:
                _tl_append_memory(conn, tl_id_for_memory, mem_text)
            elif _append_to_memory:
                _append_to_memory(conn, mem_text)

        # ── 执行 map_actions ──
        if action_type != "dialogue" and _process_map_actions and _get_current_room_id:
            fake_parsed = {"map_actions": fx.get("map_actions")}
            current_room_id = _get_current_room_id(conn, timeline_id)
            map_result = _process_map_actions(conn, fake_parsed, current_room_id, timeline_id=timeline_id)
            if map_result.get("moved_to"):
                mem = f"移动至[{map_result['moved_to']['label']}]"
                if tl_id_for_memory and _tl_append_memory:
                    _tl_append_memory(conn, tl_id_for_memory, mem)
                elif _append_to_memory:
                    _append_to_memory(conn, f"玩家{mem}。")
            if map_result.get("new_room"):
                mem = f"发现新区域[{map_result['new_room']['label']}]"
                if tl_id_for_memory and _tl_append_memory:
                    _tl_append_memory(conn, tl_id_for_memory, mem)
                elif _append_to_memory:
                    _append_to_memory(conn, mem + "。")

        # ── 执行实体提取 ──
        if _ai_extract_and_upsert_entities and entity_updates_text:
            _ai_extract_and_upsert_entities(
                conn, scene_name, node["content"] or "", player_action, entity_updates_text,
                timeline_label="主线" if not tl_id_for_memory else f"TL{tl_id_for_memory}")

        # ── 执行情绪状态机 (emotion_deltas) ──
        emotion_deltas = fx.get("emotion_deltas", [])
        _applied_emotions = []
        if emotion_deltas and isinstance(emotion_deltas, list):
            for ed in emotion_deltas:
                npc_name = str(ed.get("npc_name", "")).strip()
                if not npc_name:
                    continue
                entity_row = conn.execute(
                    "SELECT id, state_desc FROM world_entities WHERE name=? AND entity_type='npc'",
                    (npc_name,)
                ).fetchone()
                if not entity_row:
                    continue
                # 解析 state_desc JSON（兼容旧纯文本格式）
                sd = _parse_state_desc(entity_row["state_desc"])
                emo = sd.get("emotion", {"trust": 0, "fear": 0, "irritation": 0})
                # 应用 delta（clamp 到 ±30 单次，-100~100 总值）
                for key in ("trust", "fear", "irritation"):
                    delta = max(-30, min(30, int(ed.get(key, 0))))
                    emo[key] = max(-100, min(100, emo.get(key, 0) + delta))
                sd["emotion"] = emo
                conn.execute("UPDATE world_entities SET state_desc=? WHERE id=?",
                             (json.dumps(sd, ensure_ascii=False), entity_row["id"]))
                _applied_emotions.append({"npc_name": npc_name, **emo})

        # ── 执行 NPC 记忆写入 (npc_memories) ──
        # 策略：state_desc.memory 保留最近 12 条（短期工作记忆），
        # 溢出的旧记忆写入 RAG 向量库（L3 长期记忆），按语义检索召回。
        NPC_MEMORY_KEEP = 16  # 短期记忆保留条数（与前端 /16 上限一致）

        npc_memories = fx.get("npc_memories", [])
        _applied_memories = []
        _l3_evict_batch = []  # 收集需要写入 L3 的溢出记忆

        if npc_memories and isinstance(npc_memories, list):
            for nm in npc_memories:
                npc_name = str(nm.get("npc_name", "")).strip()
                memory_text = str(nm.get("memory", "")).strip()[:80]
                if not npc_name or not memory_text:
                    continue
                entity_row = conn.execute(
                    "SELECT id, state_desc FROM world_entities WHERE name=? AND entity_type='npc'",
                    (npc_name,)
                ).fetchone()
                if not entity_row:
                    continue
                sd = _parse_state_desc(entity_row["state_desc"])
                memories = sd.get("memory", [])
                # 添加新记忆（带时间线标签）
                tl_tag = f"TL{tl_id_for_memory}" if tl_id_for_memory else "主线"
                memories.append(f"[{tl_tag}] {memory_text}")
                # 溢出的旧记忆收集起来，准备写入 L3
                if len(memories) > NPC_MEMORY_KEEP:
                    overflow = memories[:len(memories) - NPC_MEMORY_KEEP]
                    memories = memories[-NPC_MEMORY_KEEP:]
                    for old_mem in overflow:
                        _l3_evict_batch.append((npc_name, old_mem))
                sd["memory"] = memories
                conn.execute("UPDATE world_entities SET state_desc=? WHERE id=?",
                             (json.dumps(sd, ensure_ascii=False), entity_row["id"]))
                _applied_memories.append({"npc_name": npc_name, "memory": memory_text})

        # ── 后台异步：溢出的 NPC 记忆写入 L3 向量库 ──
        if _l3_evict_batch and _get_embeddings:
            import threading
            evict_copy = list(_l3_evict_batch)  # 拷贝，避免跨线程引用问题
            def _async_npc_memory_to_l3(batch):
                bg_conn = get_db_connection()
                try:
                    # 按 NPC 分组，每个 NPC 的溢出记忆合并为一个 chunk
                    from collections import defaultdict
                    grouped = defaultdict(list)
                    for name, mem in batch:
                        grouped[name].append(mem)

                    for npc_name, mems in grouped.items():
                        chunk_text = f"【{npc_name}的早期记忆】" + " / ".join(mems)
                        doc_title = f"NPC记忆_{npc_name}"

                        # 查找或创建该 NPC 的 L3 记忆文档
                        existing_doc = bg_conn.execute(
                            "SELECT id FROM rag_documents WHERE source=? LIMIT 1",
                            (f"npc_memory_{npc_name}",)
                        ).fetchone()

                        if existing_doc:
                            doc_id = existing_doc["id"]
                            # 获取当前最大 chunk_index
                            max_idx = bg_conn.execute(
                                "SELECT MAX(chunk_index) as m FROM rag_chunks WHERE doc_id=?",
                                (doc_id,)
                            ).fetchone()["m"] or 0
                            chunk_index = max_idx + 1
                        else:
                            cur = bg_conn.execute(
                                "INSERT INTO rag_documents (title, source, chunk_size) VALUES (?,?,?)",
                                (doc_title, f"npc_memory_{npc_name}", 0)
                            )
                            doc_id = cur.lastrowid
                            chunk_index = 0

                        # Embedding + 写入
                        try:
                            vecs = _get_embeddings([chunk_text])
                            emb_json = json.dumps(vecs[0]) if vecs and vecs[0] else "[]"
                        except Exception:
                            emb_json = "[]"

                        bg_conn.execute(
                            "INSERT INTO rag_chunks (doc_id, chunk_index, chunk_text, embedding) "
                            "VALUES (?,?,?,?)",
                            (doc_id, chunk_index, chunk_text, emb_json)
                        )
                        # 更新 chunk_size 计数
                        bg_conn.execute(
                            "UPDATE rag_documents SET chunk_size=(SELECT COUNT(*) FROM rag_chunks WHERE doc_id=?) WHERE id=?",
                            (doc_id, doc_id)
                        )

                    bg_conn.commit()
                    _log.info("NPC 记忆 L3 归档完成：%d 条记忆来自 %d 个 NPC",
                              len(batch), len(grouped))
                    # 刷新向量缓存，使新记忆立即可检索
                    if _refresh_vector_cache:
                        _refresh_vector_cache()
                except Exception as e:
                    _log.warning("NPC 记忆 L3 归档失败（已降级跳过）: %s", e)
                finally:
                    bg_conn.close()

            threading.Thread(target=_async_npc_memory_to_l3, args=(evict_copy,), daemon=True).start()

        # ── 写入记忆流（现在确定了选择） ──
        type_tag = {"dialogue": "🗣️", "action": "🏃", "mixed": "🎭"}.get(action_type, "")
        chosen_text = node["name"] or ""
        if tl_id_for_memory and _tl_append_memory:
            _tl_append_memory(conn, tl_id_for_memory,
                              f"{type_tag}在[{scene_name}]，玩家[{player_action}]→选择了[{chosen_text}]")
        elif _append_to_memory:
            _append_to_memory(conn,
                              f"{type_tag}在[{scene_name}]，玩家试图[{player_action}]→结果：[{chosen_text}]。")

        # ── 清除副作用记录（防重复执行） ──
        # 新版：删除 pending_effects 表中的记录
        conn.execute("DELETE FROM pending_effects WHERE node_id=?", (req.node_id,))
        # 兼容旧版：如果 summary 中有 __FX__ 残留也清除
        summary = node["summary"] or ""
        if "__FX__" in summary:
            clean_summary = player_action[:100] if player_action else ""
            conn.execute("UPDATE nodes SET summary=? WHERE id=?", (clean_summary, req.node_id))

        conn.commit()
        conn.close()
        return {
            "status": "success",
            "stat_changes": applied_changes,
            "spawned_npc": spawned_npc,
            "map_result": map_result,
            "entity_updates": entity_updates_text,
            "emotion_changes": _applied_emotions,
            "npc_memories": _applied_memories,
        }
    except Exception as e:
        conn.close()
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------
# Phase 3：玩家选择分支后，生成该分支的完整叙事
# （60字短叙事的核心配套端点——润色扩写）
#
# 【调用顺序】前端应先调 expand-branch（此时 pending_effects 还在，
#   可读取副作用作为叙事上下文），再调 apply-branch-effects（执行并删除副作用）。
# ---------------------------------------------------------
class ExpandBranchRequest(BaseModel):
    node_id: int              # 玩家选择的目标节点 ID
    scene_name: str = ""      # 父场景名
    scene_content: str = ""   # 父场景正文（关键上下文）
    player_action: str = ""   # 玩家动作
    model: str | None = None  # 请求级模型选择（可选）

class ExpandBranchStreamRequest(ExpandBranchRequest):
    pass


def _build_fx_context(fx_row) -> tuple[str, str]:
    """
    从 pending_effects 行中提取副作用上下文文本和 action_type。
    返回 (fx_context_str, action_type)。
    供 expand-branch 的非流式和流式端点共享。
    """
    if not fx_row:
        return "", "mixed"
    try:
        fx = json.loads(fx_row["payload"])
    except (json.JSONDecodeError, TypeError):
        return "", "mixed"

    action_type = fx.get("action_type", "mixed")
    fx_parts = []

    for sc in (fx.get("stat_changes") or []):
        parts = []
        if sc.get("hp_delta"):  parts.append(f"HP{sc['hp_delta']:+d}")
        if sc.get("san_delta"): parts.append(f"SAN{sc['san_delta']:+d}")
        if sc.get("inventory_append"): parts.append(f"获得[{sc['inventory_append']}]")
        if sc.get("inventory_remove"): parts.append(f"失去[{sc['inventory_remove']}]")
        if parts:
            fx_parts.append(f"{sc.get('char_name', '')}: {', '.join(parts)}")

    if fx.get("entity_updates"):
        fx_parts.append(f"实体变化：{fx['entity_updates']}")

    ma = fx.get("map_actions")
    if ma and isinstance(ma, dict):
        if ma.get("movement"):
            fx_parts.append(f"移动至：{ma['movement'].get('target_room_label', '')}")
        if ma.get("new_room"):
            fx_parts.append(f"发现新区域：{ma['new_room'].get('label', '')}")

    npc = fx.get("npc")
    if npc and isinstance(npc, dict):
        fx_parts.append(f"NPC登场：{npc.get('name', '')} — {npc.get('backstory', '')}")

    fx_context = ("【该分支的确定结果】\n" + "\n".join(fx_parts)) if fx_parts else ""
    return fx_context, action_type


def _build_expand_prompts(conn, req, node_name: str, current_content: str,
                           fx_context: str, action_type: str) -> tuple[str, str]:
    """
    构建 expand-branch 的 system/user prompt。
    注入完整上下文：世界观、角色、记忆流、设定词条、实体、地图、父场景正文、副作用。
    """
    ctx = build_system_context(conn, req.scene_name, node_name, req.player_action)
    worldview, party_status, relevant_lore, session_memory, \
        l1_context, world_entities_text, rag_context, map_context = ctx

    if action_type == "dialogue":
        style_hint = "【沉浸式对话】：除了台词本身（用「」包裹），补充说话时的神态变化、语气停顿或习惯性的小动作。环境描写仅作为自然的背景板，需绝对契合当前场景的氛围（如日常、严肃或轻松）。"
    elif action_type == "action":
        style_hint = "【动作与感知】：丰富动作的过程细节与直接的感官体验（如物品的触感、周遭的白噪音、温度等）。环境描写应与角色的动作产生实际交互（例如：拉开椅子发出摩擦声、端起冒热气的餐盘），拒绝流水账式的风景速写。"
    else:
        style_hint = "【综合推演】：自然流畅地交织动作、对话与周遭环境。保持符合当前场景基调的沉浸感，日常则温馨平实，危机则节奏紧凑。"

    system_prompt = f"""你是跑团GM。请为玩家选择的分支撰写一段完整的、引人入胜的场景描写。

【全局世界观】
{worldview}

【角色当前状态】
{party_status}

{f"【已知世界实体】{chr(10)}{world_entities_text}" if world_entities_text else ""}
{f"【地图空间感知】{chr(10)}{map_context}" if map_context else ""}
{f"【相关设定词条】{chr(10)}{relevant_lore}" if relevant_lore and relevant_lore != "无" else ""}
{f"【近期剧情记忆】{chr(10)}{session_memory}" if session_memory else ""}

{fx_context}

【写作要求】
- 基于下方【分支骨架】进行润色扩写，将其转化为富有代入感的文字描述。
- 弹性字数：字数由事件的内容量自然决定（短至三五十字、长至百余字均可），绝对不要刻意凑字数。
- {style_hint}
- 保留骨架中的所有关键细节（动作、伤害程度、环境元素），不得矛盾
- 必须与【确定结果】中的数值变化一致——不得编造额外的 HP/物品/空间变化
- 承接父场景的内容和氛围，保持叙事连贯,给出自然的动作收尾或留白
- 不要输出 JSON，直接输出纯文本场景描写
- 不要加任何前缀、标题或解释"""

    # user prompt 包含完整的因果链：父场景 → 玩家行动 → 骨架 → 请求扩写
    user_parts = []
    if req.scene_name:
        user_parts.append(f"父场景名称：{req.scene_name}")
    if req.scene_content:
        user_parts.append(f"父场景正文：{req.scene_content[:300]}")
    if req.player_action:
        user_parts.append(f"玩家动作：{req.player_action}")
    user_parts.append(f"分支标题：{node_name}")
    user_parts.append(f"分支骨架（请基于此润色扩写）：{current_content}")
    user_parts.append("请将上述骨架润色为一段完整的、有代入感且契合当前氛围的场景描写。")

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


@agent_router.post("/api/ai/expand-branch")
def expand_branch_content(req: ExpandBranchRequest):
    """
    60字短叙事 Phase 3：将玩家选中分支的60字骨架润色为完整场景叙事。

    【重要】此端点应在 apply-branch-effects 之前调用——
    因为它需要从 pending_effects 读取副作用上下文，而 apply 会删除该记录。

    调用链：推演(Phase1) → 玩家选择 → expand-branch(Phase3) → apply-branch-effects(Phase2)
    """
    conn = get_db_connection()
    try:
        node = conn.execute("SELECT * FROM nodes WHERE id=?", (req.node_id,)).fetchone()
        if not node:
            conn.close()
            return {"status": "error", "message": "节点不存在"}

        node_name = node["name"] or ""
        current_content = node["content"] or ""

        # 权威判断：只有存在 pending_effects 记录的节点才需要扩写
        # 没有记录 = 非 AI 生成的节点 / 已扩写过 / 手动编辑的节点 → 直接返回原内容
        fx_row = conn.execute(
            "SELECT payload FROM pending_effects WHERE node_id=?", (req.node_id,)
        ).fetchone()
        if not fx_row:
            conn.close()
            return {"status": "success", "content": current_content, "expanded": False}
        fx_context, action_type = _build_fx_context(fx_row)

        # 构建完整上下文的 prompt
        system_prompt, user_prompt = _build_expand_prompts(
            conn, req, node_name, current_content, fx_context, action_type
        )

        expanded_text = _call_ai(system_prompt, user_prompt,
                                  temperature=0.7, max_tokens=1000, json_mode=False,
                                  model_override=req.model)

        # 写回 nodes.content
        conn.execute("UPDATE nodes SET content=? WHERE id=?", (expanded_text, req.node_id))
        conn.commit()
        conn.close()
        return {"status": "success", "content": expanded_text, "expanded": True}

    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        return {"status": "error", "message": str(e),
                "content": current_content if 'current_content' in dir() else ""}


@agent_router.post("/api/ai/expand-branch/stream")
def expand_branch_stream(req: ExpandBranchStreamRequest):
    """
    60字短叙事 Phase 3 的流式版本——前端可用打字机效果展示场景叙事生成过程。

    【重要】同非流式版本：必须在 apply-branch-effects 之前调用。
    """
    conn = get_db_connection()
    node = conn.execute("SELECT * FROM nodes WHERE id=?", (req.node_id,)).fetchone()
    if not node:
        conn.close()
        def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'content': '节点不存在'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    node_name = node["name"] or ""
    current_content = node["content"] or ""

    # 权威判断：只有存在 pending_effects 记录才扩写
    fx_row = conn.execute(
        "SELECT payload FROM pending_effects WHERE node_id=?", (req.node_id,)
    ).fetchone()
    if not fx_row:
        conn.close()
        def skip_gen():
            yield f"data: {json.dumps({'type': 'text', 'content': current_content}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'full_text': current_content, 'expanded': False}, ensure_ascii=False)}\n\n"
        return StreamingResponse(skip_gen(), media_type="text/event-stream")
    fx_context, action_type = _build_fx_context(fx_row)

    # 构建完整上下文的 prompt
    system_prompt, user_prompt = _build_expand_prompts(
        conn, req, node_name, current_content, fx_context, action_type
    )
    conn.close()

    def generate():
        full_text = ""
        for sse_line in _stream_ai_sse(system_prompt, user_prompt,
                                        temperature=0.7, max_tokens=1000,
                                        model_override=req.model):
            yield sse_line
            try:
                data = json.loads(sse_line.replace("data: ", "").strip())
                if data.get("type") == "done":
                    full_text = data.get("full_text", "")
            except Exception:
                pass

        # 写回 nodes.content
        if full_text:
            try:
                conn2 = get_db_connection()
                conn2.execute("UPDATE nodes SET content=? WHERE id=?",
                              (full_text, req.node_id))
                conn2.commit()
                conn2.close()
            except Exception as e:
                _log.warning("expand-branch 写回失败: %s", e)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
class DynamicActionRequest(BaseModel):
    current_node_id: int
    scene_name: str = ""
    content: str = ""
    player_action: str = ""
    gm_correction: str = ""  # GM 纠正指令（重新推演时填写）
    action_type: str = "mixed"  # "dialogue" | "action" | "mixed" — 动作经济分离
    model: str | None = None   # 请求级模型选择（可选，覆盖服务器默认值）

class ExpandTextRequest(BaseModel):
    scene_name: str = ""
    content: str = ""
    model: str | None = None   # 请求级模型选择（可选）


# ---------------------------------------------------------
# SSE 流式推演端点（核心）
# ---------------------------------------------------------
@agent_router.post("/api/ai/dynamic-options")
def dynamic_options_handler(request: DynamicActionRequest):
    """
    AI 分支推演。非流式模式（JSON 完整返回）。
    用于需要精确解析 JSON 的场景（如 response_format=json_object）。
    """
    conn = get_db_connection()
    ctx = build_system_context(conn, request.scene_name, request.content, request.player_action)
    worldview, party_status, relevant_lore, session_memory, l1_context, world_entities_text, rag_context, map_context = ctx

    system_prompt = _build_dynamic_system_prompt(
        worldview, party_status, relevant_lore, session_memory,
        l1_context, world_entities_text, rag_context, map_context,
        action_type=request.action_type,
        gm_correction=request.gm_correction
    )
    user_prompt = f"当前场景：{request.scene_name}\n场景内容：{request.content}\n玩家动作：{request.player_action}\n先思考，再生成分支。"

    try:
        ai_result = _call_ai(system_prompt, user_prompt, temperature=0.8, max_tokens=900,
                             json_mode=True, model_override=request.model)

        if ai_result.startswith("```"):
            ai_result = ai_result.split("```")[1]
            ai_result = ai_result[4:] if ai_result.startswith("json") else ai_result

        parsed = json.loads(ai_result.strip())
        conn2 = get_db_connection()
        new_options, spawned_npc, applied_changes, map_result, thought_process, entity_updates_text = \
            _post_process_dynamic_result(conn2, parsed, request.scene_name,
                                         request.player_action, request.current_node_id,
                                         action_type=request.action_type)
        conn2.close()

        return {
            "status": "success", "message": "推演成功",
            "new_options": new_options, "spawned_npc": spawned_npc,
            "stat_changes": applied_changes,
            "thought_process": thought_process,
            "entity_updates": entity_updates_text,
            "map_result": map_result,
        }
    except Exception as e:
        return {"status": "error", "message": f"推演失败: {str(e)}", "new_options": []}
    finally:
        conn.close()


@agent_router.post("/api/ai/dynamic-options/stream")
def dynamic_options_stream(request: DynamicActionRequest):
    """
    SSE 流式推演端点。
    前端通过 EventSource / fetch + ReadableStream 消费。
    事件格式：
      data: {"type":"text","content":"一段文本"}     ← 逐字推送（打字机效果）
      data: {"type":"done","full_text":"完整文本"}   ← AI 输出完毕
      data: {"type":"result","data":{...}}           ← 后处理结果（节点/NPC/状态等）
    """
    conn = get_db_connection()
    ctx = build_system_context(conn, request.scene_name, request.content, request.player_action)
    worldview, party_status, relevant_lore, session_memory, l1_context, world_entities_text, rag_context, map_context = ctx
    conn.close()

    # 流式推演不使用 json_mode（无法流式+JSON校验共存），
    # 改为在 prompt 中强制 JSON 格式
    system_prompt = _build_dynamic_system_prompt(
        worldview, party_status, relevant_lore, session_memory,
        l1_context, world_entities_text, rag_context, map_context,
        action_type=request.action_type,
        gm_correction=request.gm_correction
    )
    user_prompt = f"当前场景：{request.scene_name}\n场景内容：{request.content}\n玩家动作：{request.player_action}\n先思考，再生成分支。\n注意：你必须只输出 JSON，不要有任何额外文字或 markdown 代码块。"

    def generate():
        full_text = ""
        # Phase 1: 流式输出 AI 文本
        for sse_line in _stream_ai_sse(system_prompt, user_prompt,
                                        temperature=0.8, max_tokens=900,
                                        model_override=request.model):
            yield sse_line
            # 提取 full_text
            try:
                data = json.loads(sse_line.replace("data: ", "").strip())
                if data.get("type") == "done":
                    full_text = data.get("full_text", "")
            except Exception:
                pass  # SSE 解析中间态，正常跳过非 JSON 行

        # Phase 2: 后处理
        if full_text:
            try:
                # 清理可能的 markdown 代码块
                cleaned = full_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()

                parsed = json.loads(cleaned)
                conn2 = get_db_connection()
                new_options, spawned_npc, applied_changes, map_result, thought_process, entity_updates_text = \
                    _post_process_dynamic_result(conn2, parsed, request.scene_name,
                                                 request.player_action, request.current_node_id,
                                                 action_type=request.action_type)
                conn2.close()

                result_data = {
                    "status": "success",
                    "new_options": new_options,
                    "spawned_npc": spawned_npc,
                    "stat_changes": applied_changes,
                    "thought_process": thought_process,
                    "entity_updates": entity_updates_text,
                    "map_result": map_result,
                }
                yield f"data: {json.dumps({'type': 'result', 'data': result_data}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'后处理失败: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------
# 扩写端点
# ---------------------------------------------------------
@agent_router.post("/api/ai/expand-text")
def expand_scene_text(request: ExpandTextRequest):
    conn = get_db_connection()
    ctx = build_system_context(conn, request.scene_name, request.content)
    worldview, party_status, relevant_lore, session_memory, l1_context, world_entities_text, rag_context, map_context = ctx
    conn.close()

    system_prompt = f"""你是跑团GM。扩写时必须严格遵守以下已设定信息，不得编造世界观外的内容。

【全局世界观——扩写不得违反】
{worldview}
【角色当前状态】
{party_status}
{f"【已知世界实体】{chr(10)}{world_entities_text}" if world_entities_text else ""}
{f"【地图空间感知】{chr(10)}{map_context}" if map_context else ""}
{f"【近期推演快照（L1）】{chr(10)}{l1_context}" if l1_context else ""}
【剧情记忆流】
{session_memory}
【相关设定词条】
{relevant_lore}
{f"【知识库检索结果】{chr(10)}{rag_context}" if rag_context else ""}

请结合上述所有信息扩写当前场景的细节（150字左右）。扩写内容必须与世界观和角色状态一致。"""

    try:
        generated = _call_ai(system_prompt,
                             f"场景：{request.scene_name}\n描述：{request.content}",
                             temperature=0.7, max_tokens=1000, json_mode=False,
                             model_override=request.model)

        # 精炼记忆
        try:
            merged_note = _call_ai(
                "你是跑团记录员。将下方【原始描述】和【扩写补充】合并为一条不超过80字的精炼场景记忆，"
                "保留所有新增细节，去除重复内容，用第三人称叙述，不加任何前缀或解释。",
                f"【场景名】{request.scene_name}\n【原始描述】{request.content}\n【扩写补充】{generated}",
                temperature=0.2, max_tokens=120, json_mode=False,
                model_override=request.model
            ).replace('\n', ' ')
        except Exception as e:
            _log.debug("扩写记忆精炼 AI 调用失败: %s", e)
            merged_note = f"{request.scene_name}的场景细节已补充。"

        conn2 = get_db_connection()
        if _append_to_memory:
            _append_to_memory(conn2, f"【{request.scene_name}】{merged_note}")
        conn2.close()
        return {"status": "success", "generated_text": generated}
    except Exception as e:
        return {"status": "error", "message": str(e), "generated_text": ""}


# ---------------------------------------------------------
# 扩写流式端点
# ---------------------------------------------------------
@agent_router.post("/api/ai/expand-text/stream")
def expand_scene_text_stream(request: ExpandTextRequest):
    """流式扩写——前端可用打字机效果实时显示。"""
    conn = get_db_connection()
    ctx = build_system_context(conn, request.scene_name, request.content)
    worldview, party_status, relevant_lore, session_memory, l1_context, world_entities_text, rag_context, map_context = ctx
    conn.close()
# ==========================================
    # KV-Cache 拓扑结构设计 (Strict Prefix Isolation)
    # ==========================================

    # 【L0：绝对静态层】 
    # 包含系统指令、世界观、固定设定词条。
    # 特性：在整个 Session 中绝对不变，承载 90% 以上的 Token 负荷，享受最高 Cache Hit 命中率。
    static_prefix = (
        f"你是跑团GM。扩写时必须严格遵守以下已设定信息，不得编造世界观外的内容。\n"
        f"【全局世界观】\n{worldview}\n"
        f"【相关设定词条】\n{relevant_lore}\n"
    )

    # 【L1：准动态层】
    # RAG 检索结果。
    # 特性：随玩家的查询改变，放置于静态层之后，确保即使此处变动，L0 层的缓存依然有效。
    rag_infix = f"【知识库检索结果】\n{rag_context}\n" if rag_context else ""

    # 【L2：高频动态层】
    # 角色状态、地图感知、近期记忆流。
    # 特性：每轮对话必变，必须被强制放置于 Prompt 的绝对末端（Suffix）。
    dynamic_suffix = (
        f"【角色当前状态】\n{party_status}\n"
        f"{f'【地图空间感知】{chr(10)}{map_context}{chr(10)}' if map_context else ''}"
        f"【剧情记忆流】\n{session_memory}\n"
        f"请结合上述信息扩写当前场景的细节（150字左右），与世界观和角色状态一致。"
    )

    # 线性拼接，形成最终推演 Prompt
    system_prompt = f"{static_prefix}{rag_infix}{dynamic_suffix}"

    def generate():
        for sse_line in _stream_ai_sse(system_prompt,
                                       f"场景：{request.scene_name}\n玩家动作：{request.content}",
                                       temperature=0.7, max_tokens=1000):
            yield sse_line

    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")

##########################################################################
# 将此路由追加到 agent.py 末尾
@agent_router.post("/api/ai/npc-chat/stream")
def npc_chat_stream(request: NPCChatRequest):
    """NPC 专属短信/微信聊天流式端点"""
    conn = get_db_connection()
    
    # 获取 NPC 档案与情绪状态
    npc_row = conn.execute("SELECT state_desc FROM world_entities WHERE name=? AND entity_type='npc'", (request.npc_name,)).fetchone()
    
    npc_persona = ""
    if npc_row:
        sd = _parse_state_desc(npc_row["state_desc"])
        npc_persona = f"【当前状态与记忆】\n{sd.get('desc', '')}\n【情绪值】信任:{sd['emotion']['trust']} | 恐惧:{sd['emotion']['fear']} | 烦躁:{sd['emotion']['irritation']}\n"
        if sd.get("memory"):
            npc_persona += "【核心记忆】\n" + " / ".join(sd["memory"])
    else:
        npc_persona = "【系统提示】这是一个未知 NPC，请根据名字自行推断语气。"

    conn.close()

    # 微信聊天专属 Prompt 约束
    system_prompt = f"""你现在是跑团游戏中的角色：【{request.npc_name}】。
玩家正在通过类似微信的手机通讯软件和你进行文字聊天。

{npc_persona}

【严格扮演规则】：
1. 你正在用手机打字！绝对不要输出任何动作描写（例如不能出现 *皱眉*、(叹气)、【冷笑】 等）。
2. 保持日常口语化。依据你的情绪值决定语气（如果烦躁值高，字数要少、态度要冷漠或暴躁；如果信任高，可以多分享信息）。
3. 视你的性格决定是否使用 Emoji，但不要滥用。
4. 每次回复尽量简短（10-50字），符合现代人发消息的习惯。
5. 绝对不要打破第四面墙，你不知道自己是游戏角色。"""

    user_prompt = f"【近期聊天记录】\n{request.chat_history}\n\n玩家发来新消息：{request.player_message}\n请直接回复你的消息内容："

    def generate():
        for sse_line in _stream_ai_sse(system_prompt, user_prompt, temperature=0.7, max_tokens=150):
            yield sse_line

    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})