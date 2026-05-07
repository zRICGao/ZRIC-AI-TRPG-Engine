import fastapi
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import sqlite3
import json
import json_repair 
import os
import glob
from datetime import datetime
import urllib.parse
import random
import time
import webbrowser
import threading
import math



app = fastapi.FastAPI(title="RPG 桌游控制台 API - V5")

# CORS 仅允许本地访问，部署时通过 ALLOWED_ORIGINS 环境变量配置
# 示例：ALLOWED_ORIGINS=http://localhost:8000,http://192.168.1.100:8000
_default_origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "null",  # 支持 file:// 协议直接打开 player.html
]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _env_origins.split(",") if o.strip()] if _env_origins else _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------
# 【统一错误处理】：所有异常统一为 {"status":"error","message":"..."}
# 前端只需检查 response.status === "error" 即可
# ---------------------------------------------------------
@app.exception_handler(fastapi.HTTPException)
async def http_exception_handler(request, exc: fastapi.HTTPException):
    """将 HTTPException 统一为 JSON 格式，不再返回 FastAPI 默认的 {"detail":"..."}"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """将参数校验错误统一为 JSON 格式"""
    errors = exc.errors()
    msg = "; ".join(f"{e.get('loc',['?'])[-1]}: {e.get('msg','未知错误')}" for e in errors)
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": f"参数校验失败: {msg}"},
    )

# ---------------------------------------------------------
# 【模块化】：挂载独立地图模块
# ---------------------------------------------------------
from map import map_router, init_map_tables, set_db_file as map_set_db_file
from map import get_map_context as _get_map_context
from map import auto_place_room as _auto_place_room
from map import export_map_data, import_map_data, clear_map_data
app.include_router(map_router)

# ---------------------------------------------------------
# 【模块化】：挂载 RAG 知识库模块
# ---------------------------------------------------------
from rag import rag_router, configure_rag, init_rag_tables
from rag import chunk_text as _chunk_text, get_embeddings as _get_embeddings
from rag import rag_retrieve as _rag_retrieve, cosine_similarity as _cosine_similarity
from rag import refresh_vector_cache as _refresh_vector_cache
app.include_router(rag_router)

# ---------------------------------------------------------
# 【模块化】：挂载 AI Agent 模块（含 SSE 流式推演）
# ---------------------------------------------------------
from agent import agent_router, configure_agent
from agent import build_system_context as get_system_context
app.include_router(agent_router)

# ---------------------------------------------------------
# 【模块化】：挂载记忆系统模块
# ---------------------------------------------------------
from memory import memory_router, configure_memory
from memory import _l1_append
from memory import l1_get_working_context as _l1_get_working_context
from memory import append_to_memory, _tl_append_memory
from memory import fold_memory_with_ai as _fold_memory_with_ai
from memory import MEMORY_FOLD_THRESHOLD, MEMORY_SUMMARY_LIMIT
app.include_router(memory_router)

# ---------------------------------------------------------
# 【模块化】：挂载触发器系统模块
# ---------------------------------------------------------
from trigger import trigger_router, configure_trigger
app.include_router(trigger_router)

# ---------------------------------------------------------
# 【模块化】：挂载世界实体模块
# ---------------------------------------------------------
from entity import entity_router, configure_entity
from entity import get_world_entities_text as _get_world_entities_text
from entity import ai_extract_and_upsert_entities as _ai_extract_and_upsert_entities
app.include_router(entity_router)

# ---------------------------------------------------------
# 【模块化】：挂载时间线模块
# ---------------------------------------------------------
from timeline import timeline_router, configure_timeline
app.include_router(timeline_router)

# ---------------------------------------------------------
# 配置区与绝对路径锁定 (修复打包后变空白、找不到文件的问题)
# ---------------------------------------------------------
import sys
import os

# 【关键修复】：识别当前是源码运行，还是 exe 运行
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 【安全】：从 .env 文件或系统环境变量中加载 API Key，绝不硬编码
# 安装方式：pip install python-dotenv
# 使用方式：在 main.py 同级目录创建 .env 文件，写入：
#   DEEPSEEK_API_KEY=sk-xxx
#   SILICONFLOW_API_KEY=sk-xxx
#   ANTHROPIC_API_KEY=sk-xxx（可选）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass  # python-dotenv 未安装时，回退到纯系统环境变量

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DOUBAO_API_KEY = os.environ.get("DOUBAO_API_KEY", "")

if not DEEPSEEK_API_KEY:
    print("⚠️  警告：未检测到 DEEPSEEK_API_KEY 环境变量！AI 功能将不可用。", flush=True)
    print("   请在 .env 文件或系统环境变量中设置 DEEPSEEK_API_KEY=sk-xxx", flush=True)

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

DB_FILE = os.path.join(BASE_DIR, "rpg_game.db")
CAMPAIGNS_DIR = os.path.join(BASE_DIR, "campaigns")  # 模块化剧本文件夹
os.makedirs(CAMPAIGNS_DIR, exist_ok=True)

# 【模块化】：将 DB 路径注入地图模块
map_set_db_file(DB_FILE)

# 【模块化】：配置 RAG 模块
configure_rag(DB_FILE, SILICONFLOW_API_KEY)

# 【模块化】：配置记忆系统模块
configure_memory(DB_FILE, client, fn_get_embeddings=_get_embeddings)

# 【模块化】：配置世界实体模块
configure_entity(DB_FILE, client)

# 【模块化】：配置时间线模块（tl_append_memory 延迟到 startup 注入）
configure_timeline(DB_FILE, client, fn_tl_append_memory=_tl_append_memory,
                   memory_summary_limit=MEMORY_SUMMARY_LIMIT)

# ---------------------------------------------------------
# 【Persona 模式】：启动时加载 persona_mode.json（若存在）
# ---------------------------------------------------------
def _load_persona_config() -> dict:
    path = os.path.join(BASE_DIR, "persona_mode.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("enabled"):
            return cfg
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}

PERSONA_CONFIG = _load_persona_config()


# ---------------------------------------------------------
# 数据库初始化
# ---------------------------------------------------------
from contextlib import contextmanager
from logger import get_logger

_log = get_logger("main")

def get_db_connection():
    """
    创建 SQLite 连接。
    - WAL 模式：允许并发读不阻塞写，大幅减少 'database is locked' 错误
    - timeout=10：写锁等待最多 10 秒（默认 5 秒经常在后台线程并发时不够用）
    """
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


@contextmanager
def safe_db():
    """
    安全的数据库连接 context manager。
    用法：
        with safe_db() as conn:
            conn.execute(...)
    保证连接在正常退出和异常时都会关闭，且异常时自动回滚。
    """
    conn = get_db_connection()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, summary TEXT, content TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS options (id INTEGER PRIMARY KEY AUTOINCREMENT, node_id INTEGER, text TEXT, next_node_id INTEGER)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS characters (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, role TEXT, hp INTEGER, san INTEGER, inventory TEXT DEFAULT '', status TEXT DEFAULT 'active')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS system_state (key TEXT PRIMARY KEY, value TEXT)''')
    
    # 【新增】：百科库数据表
    cursor.execute('''CREATE TABLE IF NOT EXISTS lorebook (id INTEGER PRIMARY KEY AUTOINCREMENT, keywords TEXT, content TEXT)''')

    # 【关键节点触发器表】
    cursor.execute('''CREATE TABLE IF NOT EXISTS triggers (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        label       TEXT    NOT NULL DEFAULT '未命名触发器',
        target_node_id INTEGER NOT NULL,
        mode        TEXT    NOT NULL DEFAULT 'soft',
        cond_type   TEXT    NOT NULL DEFAULT '',
        cond_value  TEXT    NOT NULL DEFAULT '',
        conditions  TEXT    NOT NULL DEFAULT '[]',
        fired       INTEGER NOT NULL DEFAULT 0
    )''')
    # 兼容旧数据库：若 triggers 表缺少 conditions 列则自动添加
    try:
        cursor.execute("SELECT conditions FROM triggers LIMIT 1")
    except Exception:
        cursor.execute("ALTER TABLE triggers ADD COLUMN conditions TEXT NOT NULL DEFAULT '[]'")

    # 【多时间线并行】：时间线表
    cursor.execute('''CREATE TABLE IF NOT EXISTS timelines (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        label           TEXT    NOT NULL DEFAULT '时间线',
        color           TEXT    NOT NULL DEFAULT '#5b9cf5',
        current_node_id INTEGER,
        current_room_id INTEGER,
        memory          TEXT    NOT NULL DEFAULT '',
        char_ids        TEXT    NOT NULL DEFAULT '',
        status          TEXT    NOT NULL DEFAULT 'active',
        created_at      TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    )''')

    # 【世界实体注册表】：跨时间线共享的可变世界状态
    # 这是解决"同一NPC在不同时间线重复出现"问题的核心层
    # entity_type: npc | location | event
    # status:      active（正常）| dead（死亡）| moved（已离场）| resolved（事件结束）
    # last_seen_by: 最后接触该实体的时间线标签，供AI理解"谁知道这件事"
    cursor.execute('''CREATE TABLE IF NOT EXISTS world_entities (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type  TEXT NOT NULL DEFAULT 'npc',
        name         TEXT NOT NULL UNIQUE,
        location     TEXT NOT NULL DEFAULT '',
        status       TEXT NOT NULL DEFAULT 'active',
        last_seen_by TEXT NOT NULL DEFAULT '',
        state_desc   TEXT NOT NULL DEFAULT '',
        updated_at   TEXT NOT NULL DEFAULT (datetime('now','localtime')),
        room_id      INTEGER
    )''')

    # 【本地 RAG 知识库】：由独立模块 rag.py 管理
    init_rag_tables()

    # 【三级记忆系统 - L1 短期工作区】
    # 保存最近 N 次推演的完整上下文快照（场景名、玩家动作、AI结果摘要、思考过程）
    # 每条记录对应一次推演，按时间排序，超限时最旧的记录被淘汰到 L3
    cursor.execute('''CREATE TABLE IF NOT EXISTS memory_l1 (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_name  TEXT    NOT NULL DEFAULT '',
        player_action TEXT  NOT NULL DEFAULT '',
        ai_summary  TEXT    NOT NULL DEFAULT '',
        thought_process TEXT NOT NULL DEFAULT '',
        entity_updates TEXT NOT NULL DEFAULT '',
        created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    )''')
    # 迁移：为旧库补充 timeline_id 列（已有列时静默跳过）
    try:
        cursor.execute("ALTER TABLE memory_l1 ADD COLUMN timeline_id INTEGER DEFAULT NULL")
    except Exception:
        pass

    # 【待执行副作用表】：AI 推演 Phase 1 写入，玩家选择分支后 Phase 2 执行
    # 替代原来塞在 nodes.summary 中的 __FX__ 机制，避免 GM 手动编辑节点时损坏数据
    cursor.execute('''CREATE TABLE IF NOT EXISTS pending_effects (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id     INTEGER NOT NULL UNIQUE,
        payload     TEXT    NOT NULL DEFAULT '{}',
        created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
        FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
    )''')

    # 【地图系统】：由独立模块 map.py 管理
    init_map_tables()

    # 平滑升级：如果旧数据库没有 inventory 字段，尝试加上去
    try: cursor.execute("ALTER TABLE characters ADD COLUMN inventory TEXT DEFAULT ''")
    except sqlite3.OperationalError: pass
    # 平滑升级：characters 加 status 字段
    try: cursor.execute("ALTER TABLE characters ADD COLUMN status TEXT DEFAULT 'active'")
    except sqlite3.OperationalError: pass
    # 平滑升级：旧 map_rooms 表加 floor 字段
    try: cursor.execute("ALTER TABLE map_rooms ADD COLUMN floor INTEGER NOT NULL DEFAULT 1")
    except sqlite3.OperationalError: pass
    # 平滑升级：world_entities 加 room_id（实体坐标化）
    try: cursor.execute("ALTER TABLE world_entities ADD COLUMN room_id INTEGER")
    except sqlite3.OperationalError: pass
    # 平滑升级：world_entities 加 aliases（别名/描述词列表，用于同一实体多名称匹配）
    try: cursor.execute("ALTER TABLE world_entities ADD COLUMN aliases TEXT NOT NULL DEFAULT '[]'")
    except sqlite3.OperationalError: pass
    # 平滑升级：timelines 加 current_room_id（时间线坐标）
    try: cursor.execute("ALTER TABLE timelines ADD COLUMN current_room_id INTEGER")
    except sqlite3.OperationalError: pass
    # 平滑升级：nodes 加 expanded_content 字段（AI扩写叙事持久化）
    try: cursor.execute("ALTER TABLE nodes ADD COLUMN expanded_content TEXT DEFAULT ''")
    except sqlite3.OperationalError: pass

    # 【手机聊天记录】：NPC 私聊消息持久化
    cursor.execute('''CREATE TABLE IF NOT EXISTS npc_chat_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        npc_name   TEXT    NOT NULL,
        sender     TEXT    NOT NULL CHECK(sender IN ('player','npc')),
        message    TEXT    NOT NULL,
        created_at TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    )''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_npc_chat_npc ON npc_chat_logs(npc_name)")
    try: cursor.execute("ALTER TABLE npc_chat_logs ADD COLUMN created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))")
    except sqlite3.OperationalError: pass

    # 【时间回溯】：游戏状态快照表，保留最近10条
    cursor.execute('''CREATE TABLE IF NOT EXISTS game_checkpoints (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        from_node_id INTEGER NOT NULL,
        snapshot     TEXT    NOT NULL,
        created_at   TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
    )''')

    # 初始化本地剧情记忆流字段
    cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('session_memory', '【跑团记忆日志已初始化】\n')")
    # 投屏端同步状态
    cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_current_scene_id', '')")
    cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_scene_image', '')")
    cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_scene_prompt', '')")
    cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_scene_ai_text', '')")
    cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_bgm_url', '')")
    cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_bgm_name', '')")

    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# 【模块化】：配置 Agent 模块（注入所有依赖函数）
# 使用 startup 事件延迟执行，确保所有函数已定义
# ---------------------------------------------------------
@app.on_event("startup")
def _startup_wire_agent():
    configure_agent(
        db_file=DB_FILE,
        deepseek_client=client,
        anthropic_api_key=ANTHROPIC_API_KEY,
        persona_config=PERSONA_CONFIG,
        fn_get_map_context=_get_map_context,
        fn_auto_place_room=_auto_place_room,
        fn_process_map_actions=_process_map_actions,
        fn_get_current_room_id=_get_current_room_id,
        fn_ai_extract_and_upsert_entities=_ai_extract_and_upsert_entities,
        fn_build_persona_instruction=_build_persona_instruction,
        fn_l1_append=_l1_append,
        fn_l1_get_working_context=_l1_get_working_context,
        fn_append_to_memory=append_to_memory,
        fn_tl_append_memory=_tl_append_memory,
        fn_get_world_entities_text=_get_world_entities_text,
        fn_rag_retrieve=_rag_retrieve,
        fn_get_embeddings=_get_embeddings,
        fn_refresh_vector_cache=_refresh_vector_cache,
    )
    # 触发器模块需要 get_system_context（来自 agent），在 agent 配置完成后注入
    configure_trigger(
        db_file=DB_FILE,
        deepseek_client=client,
        fn_get_system_context=get_system_context,
        fn_append_to_memory=append_to_memory,
    )

# ---------------------------------------------------------
# 数据模型定义（供 API 请求体使用，CRUD 模型已迁移至各自模块）
# ---------------------------------------------------------
class AIContextRequest(BaseModel): scene_name: str = ""; content: str = ""
# DynamicActionRequest 已移至 agent.py
class CharUpdateRequest(BaseModel):
    name: str | None = None
    hp: int; san: int; inventory: str = ""
    status: str = "active"
class NodeCreateRequest(BaseModel): name: str; summary: str; content: str
class NodeUpdateRequest(BaseModel): name: str; summary: str; content: str
class OptionCreateRequest(BaseModel): node_id: int; text: str; next_node_id: int
class StringContentRequest(BaseModel): content: str
class LoadCampaignRequest(BaseModel): filename: str
class LorebookRequest(BaseModel): keywords: str; content: str
class CharCreateRequest(BaseModel):
    name: str
    role: str = "PC"
    hp: int = 100
    san: int = 80
    inventory: str = ""
    status: str = "active"
class AutoNPCRequest(BaseModel):
    scene_name: str
    scene_content: str
    player_action: str
class ImageGenRequest(BaseModel):
    description: str = ""       # GM 补充描述（可为空，系统会自动从场景提取）
    style: str = "fantasy"      # 风格：fantasy / horror / realistic / anime / sketch
    scene_id: int | None = None # 当前场景 ID（传入后自动提取场景名+描述+地图位置）
    scene_name: str = ""        # 场景名（scene_id 未传时的手动兜底）
    scene_content: str = ""     # 场景正文（scene_id 未传时的手动兜底）
    image_model: str = "free"   # 模型选择：free（Kolors 免费）或 doubao（豆包收费）

# 【多时间线推演】：数据模型（CRUD 模型已迁移至 timeline.py）
class TimelineDynamicRequest(BaseModel):
    timeline_id: int
    current_node_id: int
    scene_name: str = ""
    content: str = ""
    player_action: str = ""
    action_type: str = "mixed"

# 【地图系统】：数据模型已移至 map.py（通过 map_router 自动注册）

# ---------------------------------------------------------
# API 接口：剧本与存档管理（支持模块化文件夹结构）
# ---------------------------------------------------------
# 文件夹结构：
#   campaigns/
#     我的剧本/
#       campaign.json    ← 剧目（节点、角色、世界观、触发器等）
#       map.json         ← 地图（房间、通道）
#       knowledge/       ← 知识库
#         *.txt / *.md   ← RAG 文档
#   *.json               ← 兼容旧版单文件存档（平铺在 BASE_DIR 下）
# ---------------------------------------------------------

@app.get("/api/campaigns")
def list_campaigns():
    """列出所有可加载的剧本（兼容旧版单文件 + 新版文件夹结构）"""
    results = []

    # 旧版：BASE_DIR 下的 *.json（向后兼容）
    for f in glob.glob(os.path.join(BASE_DIR, "*.json")):
        fname = os.path.basename(f)
        if fname.startswith("persona_mode"): continue
        results.append({"name": fname, "type": "legacy", "path": fname})

    # 新版：campaigns/ 下的子文件夹（含 campaign.json）
    if os.path.isdir(CAMPAIGNS_DIR):
        for d in sorted(os.listdir(CAMPAIGNS_DIR)):
            folder = os.path.join(CAMPAIGNS_DIR, d)
            if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "campaign.json")):
                has_map = os.path.exists(os.path.join(folder, "map.json"))
                kb_dir = os.path.join(folder, "knowledge")
                kb_count = 0
                if os.path.isdir(kb_dir):
                    kb_count = len(
                        glob.glob(os.path.join(kb_dir, "*.txt"))
                        + glob.glob(os.path.join(kb_dir, "*.md"))
                    )
                results.append({
                    "name": d, "type": "folder",
                    "path": f"campaigns/{d}",
                    "has_map": has_map,
                    "kb_count": kb_count
                })

    # 向后兼容：同时返回旧版字符串数组格式（供未升级的前端使用）
    files_legacy = [r["path"] for r in results]
    return {"status": "success", "files": results, "files_legacy": files_legacy}

@app.post("/api/game/load")
def load_campaign(req: LoadCampaignRequest):
    """
    加载剧本。支持两种格式：
    1. 旧版单文件：req.filename = "save_xxx.json"
    2. 新版文件夹：req.filename = "campaigns/我的剧本"
       自动加载 campaign.json + map.json + knowledge/*.txt
    """
    # 判断是文件夹还是单文件
    target = os.path.join(BASE_DIR, req.filename)
    is_folder = os.path.isdir(target) and os.path.exists(os.path.join(target, "campaign.json"))

    if is_folder:
        campaign_path = os.path.join(target, "campaign.json")
        map_path = os.path.join(target, "map.json")
        kb_dir = os.path.join(target, "knowledge")
    else:
        campaign_path = target
        map_path = None
        kb_dir = None

    if not os.path.exists(campaign_path):
        raise fastapi.HTTPException(status_code=404, detail=f"文件不存在: {campaign_path}")

    try:
        with open(campaign_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        conn = get_db_connection()
        cursor = conn.cursor()

        # ── 清空所有业务数据表（含 RAG 知识库）──────────────────────
        for tbl in ("nodes", "options", "characters", "system_state",
                    "lorebook", "triggers", "timelines", "world_entities",
                    "map_rooms", "map_edges",
                    "rag_documents", "rag_chunks",
                    "memory_l1", "pending_effects", "npc_chat_logs"):
            try:
                cursor.execute(f"DELETE FROM {tbl}")
            except sqlite3.OperationalError:
                pass   # 表不存在时静默跳过（旧存档兼容）

        # 正确重置自增序列：只更新已存在行，不删整张表
        for tbl in ("nodes", "options", "characters", "lorebook",
                    "triggers", "timelines", "world_entities",
                    "map_rooms", "map_edges",
                    "rag_documents", "rag_chunks",
                    "memory_l1", "pending_effects", "npc_chat_logs"):
            try:
                cursor.execute(
                    "UPDATE sqlite_sequence SET seq=0 WHERE name=?", (tbl,)
                )
            except sqlite3.OperationalError:
                pass

        # ── 写入系统状态 ───────────────────────────────────────────
        worldview      = config.get("worldview", "【默认世界观】")
        session_memory = config.get("session_memory", "【跑团记忆日志已初始化】\n")
        cursor.execute("INSERT INTO system_state (key, value) VALUES ('worldview', ?)",       (worldview,))
        cursor.execute("INSERT INTO system_state (key, value) VALUES ('session_memory', ?)",  (session_memory,))
        cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_current_scene_id', '')")
        cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_scene_image', '')")
        cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_scene_prompt', '')")
        cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_scene_ai_text', '')")
        cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_bgm_url', '')")
        cursor.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES ('player_bgm_name', '')")

        # ── 还原业务数据 ───────────────────────────────────────────
        for char in config.get("characters", []):
            cursor.execute(
                "INSERT INTO characters (name, role, hp, san, inventory, status) VALUES (?,?,?,?,?,?)",
                (char.get("name"), char.get("role"),
                 char.get("hp"), char.get("san"), char.get("inventory", ""),
                 char.get("status", "active"))
            )
        for node in config.get("nodes", []):
            cursor.execute(
                "INSERT INTO nodes (id, name, summary, content, expanded_content) VALUES (?,?,?,?,?)",
                (node.get("id"), node.get("name"),
                 node.get("summary"), node.get("content"),
                 node.get("expanded_content", ""))
            )
        for opt in config.get("options", []):
            cursor.execute(
                "INSERT INTO options (node_id, text, next_node_id) VALUES (?,?,?)",
                (opt.get("node_id"), opt.get("text"), opt.get("next_node_id"))
            )
        for lore in config.get("lorebook", []):
            cursor.execute(
                "INSERT INTO lorebook (keywords, content) VALUES (?,?)",
                (lore.get("keywords"), lore.get("content"))
            )
        for t in config.get("triggers", []):
            # 兼容旧存档：可能只有 cond_type/cond_value，没有 conditions
            conditions_raw = t.get("conditions", [])
            # 导出时 conditions 列是 DB TEXT 字段，写入 JSON 后变成字符串 —— 需先反序列化
            if isinstance(conditions_raw, str):
                try:
                    conditions_raw = json.loads(conditions_raw)
                except (json.JSONDecodeError, TypeError):
                    conditions_raw = []
            if not conditions_raw:
                conditions_raw = []
            if not conditions_raw and t.get("cond_type") and t.get("cond_value"):
                conditions_raw = [{"type": t["cond_type"], "value": t["cond_value"]}]
            cond_type = t.get("cond_type", "")
            cond_value = t.get("cond_value", "")
            # 旧存档 cond_type 可能是 None；兼容 list 和 dict（树）两种格式
            if not cond_type:
                if isinstance(conditions_raw, list) and conditions_raw:
                    cond_type = conditions_raw[0].get("type", "")
                    cond_value = conditions_raw[0].get("value", "")
                elif isinstance(conditions_raw, dict):
                    first = (conditions_raw.get("children") or [{}])[0]
                    cond_type = first.get("type", "")
                    cond_value = first.get("value", "")
            cursor.execute(
                "INSERT INTO triggers (label, target_node_id, mode, cond_type, cond_value, conditions, fired) "
                "VALUES (?,?,?,?,?,?,0)",
                (t.get("label", ""), t.get("target_node_id", 0),
                 t.get("mode", "soft"), cond_type or "", cond_value or "",
                 json.dumps(conditions_raw, ensure_ascii=False))
            )
        for tl in config.get("timelines", []):
            cursor.execute(
                "INSERT INTO timelines (label, color, current_node_id, memory, char_ids, status, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (tl.get("label", "时间线"), tl.get("color", "#5b9cf5"),
                 tl.get("current_node_id"), tl.get("memory", ""),
                 tl.get("char_ids", ""), tl.get("status", "active"),
                 tl.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            )
        # 地图数据：优先从独立 map.json 加载，兼容旧版内嵌格式
        clear_map_data(conn)
        if is_folder and map_path and os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as mf:
                map_data = json.load(mf)
            import_map_data(conn, map_data)
        elif config.get("map_rooms"):
            import_map_data(conn, {
                "map_rooms": config.get("map_rooms", []),
                "map_edges": config.get("map_edges", [])
            })

        # ── 还原世界实体 ───────────────────────────────────────
        for we in config.get("world_entities", []):
            cursor.execute(
                "INSERT INTO world_entities "
                "(entity_type, name, location, status, last_seen_by, state_desc, updated_at, room_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (we.get("entity_type", "npc"), we.get("name", ""),
                 we.get("location", ""), we.get("status", "active"),
                 we.get("last_seen_by", ""), we.get("state_desc", ""),
                 we.get("updated_at", ""), we.get("room_id"))
            )

        # ── 还原 memory_l1 ─────────────────────────────────────────
        for ml in config.get("memory_l1", []):
            cursor.execute(
                "INSERT INTO memory_l1 (scene_name, player_action, ai_summary, "
                "thought_process, entity_updates, timeline_id) VALUES (?,?,?,?,?,?)",
                (ml.get("scene_name", ""), ml.get("player_action", ""),
                 ml.get("ai_summary", ""), ml.get("thought_process", ""),
                 ml.get("entity_updates", ""), ml.get("timeline_id"))
            )

        # ── 还原 pending_effects ────────────────────────────────────
        for pe in config.get("pending_effects", []):
            cursor.execute(
                "INSERT INTO pending_effects (node_id, payload) VALUES (?,?)",
                (pe.get("node_id"), pe.get("payload", "{}"))
            )

        # ── 还原 npc_chat_logs ──────────────────────────────────────
        for log in config.get("npc_chat_logs", []):
            cursor.execute(
                "INSERT INTO npc_chat_logs (npc_name, sender, message, created_at) VALUES (?,?,?,?)",
                (log.get("npc_name", ""), log.get("sender", "player"),
                 log.get("message", ""), log.get("created_at", ""))
            )

        # ── 还原 RAG 知识库 ────────────────────────────────────
        rag_library = config.get("rag_library", [])
        load_type = "文件夹" if is_folder else "单文件"

        # 判断是否为含预计算 embedding 的导出存档（items 有 chunks 字段而非 text 字段）
        has_precomputed = bool(rag_library) and all("chunks" in item for item in rag_library)

        if has_precomputed:
            # 直接恢复预计算数据，跳过 knowledge/ 目录和 embedding API，毫秒级完成
            for doc_item in rag_library:
                cur_doc = cursor.execute(
                    "INSERT INTO rag_documents (title, source, chunk_size) VALUES (?,?,?)",
                    (doc_item["title"][:100], doc_item.get("source", "")[:200],
                     len(doc_item.get("chunks", [])))
                )
                doc_id = cur_doc.lastrowid
                for chunk in doc_item.get("chunks", []):
                    cursor.execute(
                        "INSERT INTO rag_chunks (doc_id, chunk_index, chunk_text, embedding) "
                        "VALUES (?,?,?,?)",
                        (doc_id, chunk["index"], chunk["text"], chunk.get("embedding", "[]"))
                    )
            conn.commit()
            conn.close()
            _refresh_vector_cache()
            _log.info("RAG 知识库从存档直接恢复，共 %d 个文档，跳过 embedding 重建", len(rag_library))
            return {"status": "success", "message": f"成功加载 {req.filename}（{load_type}，含 {len(rag_library)} 个RAG文档）"}

        # 原始剧本（无预计算数据）：从 knowledge/ 目录完整导入管道
        raw_library = []
        if is_folder and kb_dir and os.path.isdir(kb_dir):
            for kb_file in sorted(glob.glob(os.path.join(kb_dir, "*.txt")) +
                                  glob.glob(os.path.join(kb_dir, "*.md"))):
                try:
                    with open(kb_file, "r", encoding="utf-8") as kf:
                        kb_text = kf.read().strip()
                    if kb_text:
                        raw_library.append({
                            "title": os.path.splitext(os.path.basename(kb_file))[0],
                            "source": f"knowledge/{os.path.basename(kb_file)}",
                            "text": kb_text
                        })
                except Exception as e:
                    _log.warning("知识库文件读取失败: %s — %s", kb_file, e)

        for rag_item in raw_library:
            title  = rag_item.get("title", "未命名")
            source = rag_item.get("source", "")
            text   = rag_item.get("text", "")
            if not text.strip():
                continue
            chunks = _chunk_text(text)
            cur_doc = cursor.execute(
                "INSERT INTO rag_documents (title, source, chunk_size) VALUES (?,?,?)",
                (title[:100], source[:200], len(chunks))
            )
            doc_id = cur_doc.lastrowid
            for idx, chunk in enumerate(chunks):
                cursor.execute(
                    "INSERT INTO rag_chunks (doc_id, chunk_index, chunk_text, embedding) "
                    "VALUES (?,?,?,?)",
                    (doc_id, idx, chunk, "[]")
                )

        conn.commit()

        # ── 异步重建 RAG embedding（不阻塞加载响应）───────────
        if raw_library:
            import threading
            def _rebuild_embeddings():
                try:
                    _conn = get_db_connection()
                    rows = _conn.execute(
                        "SELECT id, chunk_text FROM rag_chunks WHERE embedding='[]' ORDER BY id"
                    ).fetchall()
                    if not rows:
                        _conn.close()
                        return
                    _log.info("后台 RAG embedding 重建开始，共 %d 个 chunks", len(rows))
                    texts = [r["chunk_text"] for r in rows]
                    ids   = [r["id"] for r in rows]
                    BATCH = 16
                    embedded_count = 0
                    for i in range(0, len(texts), BATCH):
                        batch_texts = texts[i:i+BATCH]
                        batch_ids   = ids[i:i+BATCH]
                        try:
                            vecs = _get_embeddings(batch_texts)
                            for rid, vec in zip(batch_ids, vecs):
                                _conn.execute(
                                    "UPDATE rag_chunks SET embedding=? WHERE id=?",
                                    (json.dumps(vec), rid)
                                )
                            _conn.commit()
                            embedded_count += len(batch_texts)
                        except Exception as e:
                            _log.warning("RAG embedding 批次写入失败 (batch %d): %s", i // BATCH, e)
                    _conn.close()
                    _log.info("后台 RAG embedding 重建完成，成功 %d/%d", embedded_count, len(rows))
                    _refresh_vector_cache()  # 重建完成后刷新内存缓存
                except Exception as e:
                    _log.error("后台 RAG embedding 重建线程异常: %s", e, exc_info=True)
            threading.Thread(target=_rebuild_embeddings, daemon=True).start()

        conn.close()
        # 加载完成后立即刷新向量缓存（embedding 为空的 chunks 会在后台线程重建后再次刷新）
        _refresh_vector_cache()
        return {"status": "success", "message": f"成功加载 {req.filename}（{load_type}，含 {len(raw_library)} 个RAG文档）"}

    except Exception as e:
        _log.error("load_campaign 异常: %s", e, exc_info=True)
        raise fastapi.HTTPException(status_code=500, detail=f"加载失败: {str(e)}")

class ExportSaveRequest(BaseModel):
    save_name: str = ""

@app.post("/api/game/export")
def export_campaign(req: ExportSaveRequest = ExportSaveRequest()):
    """导出存档到 campaigns/ 文件夹，支持自定义名。"""
    conn = get_db_connection()
    nodes = [dict(row) for row in conn.execute("SELECT * FROM nodes").fetchall()]
    options = [dict(row) for row in conn.execute("SELECT * FROM options").fetchall()]
    characters = [dict(row) for row in conn.execute("SELECT * FROM characters").fetchall()]
    lorebook = [dict(row) for row in conn.execute("SELECT * FROM lorebook").fetchall()]
    wv_row = conn.execute("SELECT value FROM system_state WHERE key = 'worldview'").fetchone()
    mem_row = conn.execute("SELECT value FROM system_state WHERE key = 'session_memory'").fetchone()
    triggers = [dict(row) for row in conn.execute("SELECT * FROM triggers").fetchall()]
    timelines = [dict(row) for row in conn.execute("SELECT * FROM timelines").fetchall()]
    world_entities = [dict(row) for row in conn.execute("SELECT * FROM world_entities").fetchall()]
    map_data = export_map_data(conn)

    # RAG 知识库：导出含 embedding
    rag_export = []
    rag_docs = conn.execute("SELECT * FROM rag_documents ORDER BY id").fetchall()
    for doc in rag_docs:
        chunks = conn.execute(
            "SELECT chunk_index, chunk_text, embedding FROM rag_chunks WHERE doc_id=? ORDER BY chunk_index", (doc["id"],)
        ).fetchall()
        rag_export.append({
            "title": doc["title"], "source": doc["source"],
            "chunks": [{"index": c["chunk_index"], "text": c["chunk_text"], "embedding": c["embedding"]} for c in chunks]
        })
    memory_l1 = [dict(row) for row in conn.execute("SELECT * FROM memory_l1").fetchall()]
    pending_effects = [dict(row) for row in conn.execute("SELECT * FROM pending_effects").fetchall()]
    npc_chat_logs = [dict(row) for row in conn.execute("SELECT npc_name, sender, message, created_at FROM npc_chat_logs ORDER BY id").fetchall()]
    conn.close()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        if req.save_name.strip():
            safe_name = "".join(c for c in req.save_name.strip() if c.isalnum() or c in " _-（）()【】·")[:60]
            folder_name = safe_name or f"save_{timestamp}"
        else:
            folder_name = f"save_{timestamp}"
        folder_path = os.path.join(CAMPAIGNS_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        campaign_data = {
            "worldview": wv_row["value"] if wv_row else "",
            "session_memory": mem_row["value"] if mem_row else "",
            "characters": characters, "nodes": nodes, "options": options,
            "lorebook": lorebook, "triggers": triggers,
            "timelines": timelines, "world_entities": world_entities,
            "rag_library": rag_export,
            "memory_l1": memory_l1,
            "pending_effects": pending_effects,
            "npc_chat_logs": npc_chat_logs,
        }
        with open(os.path.join(folder_path, "campaign.json"), "w", encoding="utf-8") as f:
            json.dump(campaign_data, f, ensure_ascii=False, indent=4)
        with open(os.path.join(folder_path, "map.json"), "w", encoding="utf-8") as f:
            json.dump(map_data, f, ensure_ascii=False, indent=4)

        if rag_export:
            kb_dir = os.path.join(folder_path, "knowledge")
            os.makedirs(kb_dir, exist_ok=True)
            for i, rag_item in enumerate(rag_export):
                safe_title = "".join(c for c in rag_item["title"] if c.isalnum() or c in " _-")[:40] or f"doc_{i}"
                full_text = "\n".join(ch["text"] for ch in rag_item.get("chunks", []))
                with open(os.path.join(kb_dir, f"{safe_title}.txt"), "w", encoding="utf-8") as f:
                    f.write(full_text)

        return {
            "status": "success",
            "folder": folder_name,
            "message": f"已导出到 campaigns/{folder_name}/"
        }
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 【记忆系统】：已迁移至 memory.py（通过 app.include_router(memory_router) 自动注册）
# append_to_memory, _l1_append, _l1_get_working_context, _tl_append_memory 通过顶部 import 引入
# ---------------------------------------------------------

@app.get("/api/game/lorebook")
def get_lorebook():
    with safe_db() as conn:
        r = [dict(row) for row in conn.execute("SELECT * FROM lorebook").fetchall()]
    return {"status": "success", "lorebook": r}

@app.post("/api/game/lorebook")
def create_lore(req: LorebookRequest):
    with safe_db() as conn:
        conn.execute("INSERT INTO lorebook (keywords, content) VALUES (?, ?)", (req.keywords, req.content))
        conn.commit()
    return {"status": "success"}

@app.delete("/api/game/lorebook/{lore_id}")
def delete_lore(lore_id: int):
    with safe_db() as conn:
        conn.execute("DELETE FROM lorebook WHERE id = ?", (lore_id,))
        conn.commit()
    return {"status": "success"}

# ---------------------------------------------------------
# API 接口：词条库、世界观、节点 CRUD (原样保留)
# ---------------------------------------------------------
@app.get("/api/game/worldview")
def get_worldview():
    with safe_db() as conn:
        r = conn.execute("SELECT value FROM system_state WHERE key = 'worldview'").fetchone()
    return {"status": "success", "content": r["value"] if r else ""}

@app.put("/api/game/worldview")
def update_worldview(req: StringContentRequest):
    with safe_db() as conn:
        conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('worldview', ?)", (req.content,))
        conn.commit()
    return {"status": "success"}

@app.get("/api/game/stat-labels")
def get_stat_labels():
    """获取 HP/SAN 的自定义显示名。"""
    with safe_db() as conn:
        hp_row = conn.execute("SELECT value FROM system_state WHERE key='hp_label'").fetchone()
        san_row = conn.execute("SELECT value FROM system_state WHERE key='san_label'").fetchone()
    return {"hp_label": hp_row["value"] if hp_row else "HP", "san_label": san_row["value"] if san_row else "SAN"}

class StatLabelsRequest(BaseModel):
    hp_label: str = "HP"
    san_label: str = "SAN"

@app.put("/api/game/stat-labels")
def update_stat_labels(req: StatLabelsRequest):
    """更新 HP/SAN 显示名。"""
    with safe_db() as conn:
        conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('hp_label', ?)", (req.hp_label[:20],))
        conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('san_label', ?)", (req.san_label[:20],))
        conn.commit()
    return {"status": "success"}

@app.get("/api/game/state")
def get_game_state():
    with safe_db() as conn:
        n = [dict(row) for row in conn.execute("SELECT * FROM nodes").fetchall()]
        o = [dict(row) for row in conn.execute("SELECT * FROM options").fetchall()]
        c = [dict(row) for row in conn.execute("SELECT * FROM characters").fetchall()]
        wv_row = conn.execute("SELECT value FROM system_state WHERE key='worldview'").fetchone()
        worldview = wv_row["value"] if wv_row else ""
    for node in n:
        node["options"] = [opt for opt in o if opt["node_id"] == node["id"]]
    return {"status": "success", "nodes": n, "characters": c, "worldview": worldview}

@app.post("/api/game/character/{char_id}")
def update_character(char_id: int, req: CharUpdateRequest):
    with safe_db() as conn:
        if req.name is not None and req.name.strip():
            conn.execute("UPDATE characters SET name=?, hp=?, san=?, inventory=?, status=? WHERE id=?",
                         (req.name.strip()[:50], req.hp, req.san, req.inventory, req.status, char_id))
        else:
            conn.execute("UPDATE characters SET hp=?, san=?, inventory=?, status=? WHERE id=?",
                         (req.hp, req.san, req.inventory, req.status, char_id))
        conn.commit()
    return {"status": "success"}

@app.post("/api/game/node")
def create_node(req: NodeCreateRequest):
    with safe_db() as conn:
        c = conn.execute("INSERT INTO nodes (name, summary, content) VALUES (?,?,?)", (req.name, req.summary, req.content))
        i = c.lastrowid
        conn.commit()
    return {"status": "success", "id": i}

@app.put("/api/game/node/{node_id}")
def update_node(node_id: int, req: NodeUpdateRequest):
    with safe_db() as conn:
        conn.execute("UPDATE nodes SET name=?, summary=?, content=? WHERE id=?", (req.name, req.summary, req.content, node_id))
        conn.commit()
    return {"status": "success"}

@app.delete("/api/game/node/{node_id}")
def delete_node(node_id: int):
    with safe_db() as conn:
        conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        conn.execute("DELETE FROM options WHERE node_id=? OR next_node_id=?", (node_id, node_id))
        conn.commit()
    return {"status": "success"}

@app.post("/api/game/option")
def create_option(req: OptionCreateRequest):
    with safe_db() as conn:
        c = conn.execute("INSERT INTO options (node_id, text, next_node_id) VALUES (?,?,?)", (req.node_id, req.text, req.next_node_id))
        i = c.lastrowid
        conn.commit()
    return {"status": "success", "id": i}

@app.delete("/api/game/option/{option_id}")
def delete_option(option_id: int):
    with safe_db() as conn:
        conn.execute("DELETE FROM options WHERE id=?", (option_id,))
        conn.commit()
    return {"status": "success"}

# ---------------------------------------------------------
# 【新增核心】：打包提取四大维度的全知上下文
# ---------------------------------------------------------

def _build_persona_instruction() -> str:
    """
    将 persona_mode.json 的核心规则、MBTI 池、癖好池组装为可注入 system prompt 的文本块。
    若 persona_mode 未启用则返回空字符串，对已有逻辑零影响。
    """
    if not PERSONA_CONFIG:
        return ""
    lines = [PERSONA_CONFIG.get("core_rule", "")]
    npc_rule = PERSONA_CONFIG.get("npc_generation_rule", "")
    if npc_rule:
        lines.append(npc_rule)
    mbti_pool = PERSONA_CONFIG.get("mbti_pool", [])
    if mbti_pool:
        lines.append("【可用MBTI类型池】" + " | ".join(mbti_pool))
    quirk_pool = PERSONA_CONFIG.get("quirk_pool", [])
    if quirk_pool:
        lines.append("【可用行为癖好池】" + " | ".join(quirk_pool))
    return "\n".join(filter(None, lines))


# ---------------------------------------------------------
# 【世界实体】：已迁移至 entity.py
# _get_world_entities_text, _ai_extract_and_upsert_entities 通过顶部 import 引入
# ---------------------------------------------------------


# =============================================================
# 【RAG 知识库引擎】：已迁移至 rag.py
# =============================================================


# _get_map_context 和 _auto_place_room 已移至 map.py
# 通过顶部 from map import get_map_context as _get_map_context 引入
# ---------------------------------------------------------
# 【地图-AI 联动引擎】：处理 AI 返回的 map_actions
# ---------------------------------------------------------
def _process_map_actions(conn, parsed: dict, current_room_id: int | None,
                         timeline_id: int | None = None) -> dict:
    """
    处理 AI 推演返回的 map_actions 字段，执行地图状态变更。
    返回一个 map_result 字典，包含所有执行结果供前端渲染。

    处理顺序：unlock_edge → movement → new_room
    所有操作均有硬校验，AI 的幻觉不会破坏地图一致性。
    """
    result = {
        "moved_to": None,          # {"room_id": int, "label": str}
        "new_room": None,          # {"room_id": int, "label": str}
        "unlocked_edge": None,     # {"edge_id": int, "from": str, "to": str}
        "errors": [],              # 校验失败的信息（GM 可看到）
    }

    map_actions = parsed.get("map_actions") if isinstance(parsed, dict) else None
    if not map_actions or not isinstance(map_actions, dict):
        return result

    MAP_ID = 1  # 当前只支持单地图

    # ── 1. 解锁通道 ──────────────────────────────────────
    unlock = map_actions.get("unlock_edge")
    if unlock and isinstance(unlock, dict):
        from_label = str(unlock.get("from_label", "")).strip()
        to_label   = str(unlock.get("to_label", "")).strip()
        key_used   = str(unlock.get("key_used", "")).strip()

        if from_label and to_label:
            # 按标签模糊匹配房间对
            from_room = conn.execute(
                "SELECT id FROM map_rooms WHERE label LIKE ? LIMIT 1", (f"%{from_label[:20]}%",)
            ).fetchone()
            to_room = conn.execute(
                "SELECT id FROM map_rooms WHERE label LIKE ? LIMIT 1", (f"%{to_label[:20]}%",)
            ).fetchone()

            if from_room and to_room:
                edge = conn.execute(
                    "SELECT * FROM map_edges WHERE "
                    "((from_id=? AND to_id=?) OR (from_id=? AND to_id=?)) AND locked=1",
                    (from_room["id"], to_room["id"], to_room["id"], from_room["id"])
                ).fetchone()

                if edge:
                    # 硬校验：检查玩家背包是否有钥匙
                    if key_used:
                        all_inv = " ".join(
                            (c["inventory"] or "").lower()
                            for c in conn.execute("SELECT inventory FROM characters").fetchall()
                        )
                        if key_used.lower() in all_inv:
                            conn.execute("UPDATE map_edges SET locked=0 WHERE id=?", (edge["id"],))
                            result["unlocked_edge"] = {
                                "edge_id": edge["id"],
                                "from": from_label, "to": to_label,
                                "key_used": key_used
                            }
                            # 从背包移除钥匙
                            chars = conn.execute("SELECT * FROM characters").fetchall()
                            for c in chars:
                                inv = c["inventory"] or ""
                                if key_used.lower() in inv.lower():
                                    new_inv = ", ".join(
                                        p.strip() for p in inv.split(",")
                                        if key_used.lower() not in p.lower()
                                    ).strip(", ")
                                    conn.execute(
                                        "UPDATE characters SET inventory=? WHERE id=?",
                                        (new_inv, c["id"])
                                    )
                                    break
                        else:
                            result["errors"].append(f"解锁失败：背包中没有「{key_used}」")
                    else:
                        # 无需钥匙，直接解锁
                        conn.execute("UPDATE map_edges SET locked=0 WHERE id=?", (edge["id"],))
                        result["unlocked_edge"] = {
                            "edge_id": edge["id"],
                            "from": from_label, "to": to_label, "key_used": ""
                        }
                else:
                    result["errors"].append(f"解锁失败：{from_label}↔{to_label} 之间没有上锁的通道")

    # ── 2. 空间移动 ──────────────────────────────────────
    movement = map_actions.get("movement")
    if movement and isinstance(movement, dict) and current_room_id:
        target_label = str(movement.get("target_room_label", "")).strip()

        if target_label:
            # 按标签模糊匹配目标房间
            target_room = conn.execute(
                "SELECT id, label FROM map_rooms WHERE label LIKE ? LIMIT 1",
                (f"%{target_label[:20]}%",)
            ).fetchone()

            if target_room:
                # 硬校验：目标房间是否与当前房间相邻？
                is_adjacent = conn.execute(
                    "SELECT id, locked, key_item FROM map_edges WHERE "
                    "((from_id=? AND to_id=?) OR (from_id=? AND to_id=?))",
                    (current_room_id, target_room["id"],
                     target_room["id"], current_room_id)
                ).fetchone()

                if is_adjacent:
                    if is_adjacent["locked"]:
                        result["errors"].append(
                            f"移动失败：通往「{target_room['label']}」的通道被锁住"
                            f"（需要：{is_adjacent['key_item'] or '钥匙'}）"
                        )
                    else:
                        # 执行移动
                        conn.execute(
                            "UPDATE map_rooms SET state='explored' WHERE id=?",
                            (target_room["id"],)
                        )
                        if timeline_id:
                            conn.execute(
                                "UPDATE timelines SET current_room_id=? WHERE id=?",
                                (target_room["id"], timeline_id)
                            )
                        else:
                            conn.execute(
                                "INSERT OR REPLACE INTO system_state (key,value) "
                                "VALUES ('current_room_id',?)",
                                (str(target_room["id"]),)
                            )
                        result["moved_to"] = {
                            "room_id": target_room["id"],
                            "label": target_room["label"]
                        }
                else:
                    result["errors"].append(
                        f"移动失败：「{target_room['label']}」与当前房间不相邻"
                    )
            else:
                result["errors"].append(f"移动失败：找不到名为「{target_label}」的房间")

    # ── 3. 发现新房间（自动生长）──────────────────────────
    new_room_data = map_actions.get("new_room")
    if new_room_data and isinstance(new_room_data, dict) and current_room_id:
        nr_label = str(new_room_data.get("label", "")).strip()[:30]
        nr_desc  = str(new_room_data.get("description", "")).strip()[:200]

        if nr_label:
            # 检查是否已有同名房间
            existing = conn.execute(
                "SELECT id, label FROM map_rooms WHERE label LIKE ? LIMIT 1",
                (f"%{nr_label[:20]}%",)
            ).fetchone()

            if existing:
                result["errors"].append(f"新房间「{nr_label}」已存在（ID:{existing['id']}），跳过创建")
            else:
                # 调用 auto_place_room 在当前房间旁自动放置
                parent_room_id = current_room_id
                # 如果刚刚移动过，以移动后的房间为基点
                if result["moved_to"]:
                    parent_room_id = result["moved_to"]["room_id"]

                new_id = _auto_place_room(conn, MAP_ID, parent_room_id, nr_label, 0, nr_desc)
                if new_id:
                    result["new_room"] = {"room_id": new_id, "label": nr_label}
                else:
                    result["errors"].append(f"新房间「{nr_label}」生成失败：四周无空位")

    conn.commit()
    return result


def _get_current_room_id(conn, timeline_id: int | None = None) -> int | None:
    """获取当前房间 ID（支持时间线模式）"""
    if timeline_id:
        tl = conn.execute(
            "SELECT current_room_id FROM timelines WHERE id=?", (timeline_id,)
        ).fetchone()
        return tl["current_room_id"] if tl and tl["current_room_id"] else None
    else:
        row = conn.execute(
            "SELECT value FROM system_state WHERE key='current_room_id'"
        ).fetchone()
        return int(row["value"]) if row and row["value"] else None



# get_system_context, expand_scene_text, generate_dynamic_options
# 已迁移至 agent.py（通过 app.include_router(agent_router) 自动注册）

# ---------------------------------------------------------
# API 接口：角色管理 (新增/删除)
# ---------------------------------------------------------
@app.post("/api/game/character")
def create_character(req: CharCreateRequest):
    with safe_db() as conn:
        _existing_char = conn.execute(
            "SELECT id FROM characters WHERE name=?", (req.name[:50],)
        ).fetchone()
        if _existing_char:
            conn.execute(
                "UPDATE characters SET role=?, hp=?, san=?, inventory=?, status=? WHERE id=?",
                (req.role[:10], req.hp, req.san, req.inventory[:200], req.status, _existing_char["id"])
            )
            new_id = _existing_char["id"]
        else:
            new_id = conn.execute(
                "INSERT INTO characters (name, role, hp, san, inventory, status) VALUES (?, ?, ?, ?, ?, ?)",
                (req.name[:50], req.role[:10], req.hp, req.san, req.inventory[:200], req.status)
            ).lastrowid
        # NPC 自动创建世界实体条目（使情绪状态机可用）
        if req.role.upper() == 'NPC':
            existing = conn.execute("SELECT id FROM world_entities WHERE name=?", (req.name[:50],)).fetchone()
            if not existing:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                initial_sd = json.dumps({
                    "desc": req.inventory[:200] if req.inventory else "",
                    "emotion": {"trust": 0, "fear": 0, "irritation": 0},
                    "breakpoint": {"threshold": 70, "trigger_field": "irritation", "reaction": ""},
                    "memory": []
                }, ensure_ascii=False)
                conn.execute(
                    "INSERT INTO world_entities (entity_type, name, location, status, state_desc, updated_at) "
                    "VALUES (?,?,?,?,?,?)",
                    ("npc", req.name[:50], "", "active", initial_sd, now_str)
                )
        conn.commit()
    return {"status": "success", "id": new_id}

@app.delete("/api/game/character/{char_id}")
def delete_character(char_id: int):
    with safe_db() as conn:
        result = conn.execute("SELECT id FROM characters WHERE id = ?", (char_id,)).fetchone()
        if not result:
            raise fastapi.HTTPException(status_code=404, detail="角色不存在")
        conn.execute("DELETE FROM characters WHERE id = ?", (char_id,))
        conn.commit()
    return {"status": "success"}

@app.post("/api/ai/generate-npc")
def generate_npc(request: AutoNPCRequest):
    """根据当前剧情场景，AI自动生成一个合适的NPC并写入数据库"""
    conn = get_db_connection()
    worldview, party_status, relevant_lore, session_memory, l1_context, world_entities_text, rag_context, map_context = get_system_context(
        conn, request.scene_name, request.scene_content, request.player_action
    )
    conn.close()

    system_prompt = f"""你是一个跑团GM，需要根据当前剧情创建一个新NPC角色。
【全局世界观】\n{worldview}
【近期记忆】\n{session_memory}
【相关设定】\n{relevant_lore}
{f"【知识库检索结果（语义最相关的背景设定）】{chr(10)}{rag_context}" if rag_context else ""}
{f"【地图空间感知——AI必须遵守此空间结构推演】{chr(10)}{map_context}" if map_context else ""}
【当前队伍】\n{party_status}

请根据当前场景和玩家行动，生成一个符合剧情的NPC。
必须严格返回JSON格式：
{{"name": "NPC姓名（10字以内）", "role": "NPC", "hp": 50, "san": 60, "inventory": "持有物品或特征描述（30字以内）", "backstory": "简短背景描述（50字以内）"}}"""

    user_prompt = f"当前场景：{request.scene_name}\n场景内容：{request.scene_content}\n玩家行动：{request.player_action}\n请生成一个合适的NPC。"

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.9,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        ai_result = response.choices[0].message.content.strip()
        npc_data = json_repair.loads(ai_result)

        name = npc_data.get("name", "神秘人")[:50]
        hp   = int(npc_data.get("hp", 50))
        san  = int(npc_data.get("san", 60))
        inv  = npc_data.get("inventory", "")[:200]
        backstory = npc_data.get("backstory", "")

        conn2 = get_db_connection()
        _existing = conn2.execute("SELECT id FROM characters WHERE name=?", (name,)).fetchone()
        if _existing:
            conn2.execute(
                "UPDATE characters SET hp=?, san=?, inventory=? WHERE id=?",
                (hp, san, inv, _existing["id"]))
            new_id = _existing["id"]
        else:
            new_id = conn2.execute(
                "INSERT INTO characters (name, role, hp, san, inventory) VALUES (?, 'NPC', ?, ?, ?)",
                (name, hp, san, inv)).lastrowid
        append_to_memory(conn2, f"NPC [{name}] 登场于 [{request.scene_name}]。背景：{backstory}")
        conn2.commit()
        conn2.close()

        return {"status": "success", "npc": {"id": new_id, "name": name, "role": "NPC", "hp": hp, "san": san, "inventory": inv, "backstory": backstory}}
    except Exception as e:
        return {"status": "error", "message": f"NPC生成失败: {str(e)}"}

# ---------------------------------------------------------
# API 接口：跳转场景时自动检测是否需要生成 NPC
# ---------------------------------------------------------
class CheckNPCRequest(BaseModel):
    scene_name: str
    scene_content: str

@app.post("/api/ai/check-npc")
def check_npc_on_enter(request: CheckNPCRequest):
    """
    跳转到新场景时调用。
    AI 轻量判断：该场景是否暗示一个新NPC应当出现？
    为节省 token，使用较小的 max_tokens，且加入防刷限制（同名NPC不重复创建）。
    """
    conn = get_db_connection()
    worldview, party_status, relevant_lore, session_memory, l1_context, world_entities_text, rag_context, map_context = get_system_context(
        conn, request.scene_name, request.scene_content
    )

    # 防重复：若当前角色表中已有同名NPC则跳过
    existing_names = [r["name"] for r in conn.execute("SELECT name FROM characters WHERE role='NPC'").fetchall()]

    system_prompt = f"""你是跑团GM助手。根据玩家刚进入的场景，判断是否需要立即生成一个新NPC登场。
【世界观】{worldview}
【近期记忆】{session_memory}
【相关设定】{relevant_lore}
【已有角色】{', '.join(existing_names) or '无'}

判断规则：
- 场景描述中明确提到某个人物、守卫、商人、敌人等具体角色 → 生成
- 场景只是环境描写（森林、废墟等）→ 不生成，返回 null
- 已有同名角色 → 不生成，返回 null

必须返回 JSON：
{{"npc": {{"name": "姓名", "role": "NPC", "hp": 50, "san": 60, "inventory": "特征", "backstory": "背景"}} | null}}"""

    user_prompt = f"场景名：{request.scene_name}\n场景内容：{request.scene_content[:300]}"

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": user_prompt}],
            temperature=0.7,
            max_tokens=250,
            response_format={"type": "json_object"}
        )
        parsed = json_repair.loads(response.choices[0].message.content.strip())
        npc_data = parsed.get("npc")

        if not npc_data or not npc_data.get("name"):
            conn.close()
            return {"status": "success", "spawned_npc": None}

        npc_name = npc_data.get("name", "神秘人")[:50]

        # 二次防重复检查
        if npc_name in existing_names:
            conn.close()
            return {"status": "success", "spawned_npc": None}

        npc_hp   = int(npc_data.get("hp", 50))
        npc_san  = int(npc_data.get("san", 60))
        npc_inv  = npc_data.get("inventory", "")[:200]
        npc_back = npc_data.get("backstory", "")

        cur = conn.execute(
            "INSERT INTO characters (name, role, hp, san, inventory) VALUES (?, 'NPC', ?, ?, ?)",
            (npc_name, npc_hp, npc_san, npc_inv)
        )
        spawned_npc = {"id": cur.lastrowid, "name": npc_name, "role": "NPC",
                       "hp": npc_hp, "san": npc_san, "inventory": npc_inv, "backstory": npc_back}
        append_to_memory(conn, f"NPC [{npc_name}] 登场于新场景 [{request.scene_name}]。{npc_back}")
        conn.commit()
        conn.close()
        return {"status": "success", "spawned_npc": spawned_npc}

    except Exception as e:
        conn.close()
        return {"status": "success", "spawned_npc": None}  # 静默失败，不影响主流程


# ---------------------------------------------------------
# 【触发器系统】：已迁移至 trigger.py（通过 app.include_router(trigger_router) 自动注册）
# ---------------------------------------------------------

# ---------------------------------------------------------
# 【核心增强】：AI 图片生成（场景感知 + Kolors 中文直出）
# ---------------------------------------------------------
@app.post("/api/ai/generate-image")
def generate_image(request: ImageGenRequest):
    """
    图片生成（场景感知版）：
      Step 1 - 自动从当前场景提取上下文（场景名、正文、世界观、地图位置）
      Step 2 - DeepSeek 将上下文压缩为一段精炼的中文画面描述（含构图、光影、氛围）
      Step 3 - 中文 prompt + 风格锚点直接发给 Kolors
    GM 可以不填 description，系统自动从场景生图；也可以填补充描述来引导画面重点。
    """
    # ── 风格方向锚点（纯风格，画质控制词由 DeepSeek 在输出末尾自动追加） ──
    style_map = {
        "fantasy":   "奇幻RPG插画，数字绘画",
        "horror":    "黑暗恐怖风格，哥特式",
        "realistic": "照片级写实，电影画面",
        "anime":     "日系动漫风格，吉卜力风",
        "sketch":    "铅笔素描风格，黑白线稿",
        "none":      "",
    }
    style_anchor = style_map.get(request.style, style_map["fantasy"])

    # ── Step 1：收集场景上下文 ──
    scene_name = request.scene_name
    scene_content = request.scene_content
    worldview_snippet = ""
    map_location = ""

    if request.scene_id:
        try:
            with safe_db() as conn:
                node = conn.execute("SELECT name, content FROM nodes WHERE id=?", (request.scene_id,)).fetchone()
                if node:
                    scene_name = scene_name or node["name"] or ""
                    scene_content = scene_content or node["content"] or ""

                # 世界观摘要（取前 150 字）
                wv = conn.execute("SELECT value FROM system_state WHERE key='worldview'").fetchone()
                if wv and wv["value"]:
                    worldview_snippet = wv["value"][:150]

                # 地图位置
                room_row = conn.execute("SELECT value FROM system_state WHERE key='current_room_id'").fetchone()
                if room_row and room_row["value"]:
                    room = conn.execute("SELECT label, description FROM map_rooms WHERE id=?",
                                        (int(room_row["value"]),)).fetchone()
                    if room:
                        map_location = f"{room['label']}（{room['description'] or ''}）".strip("（）")
        except Exception as e:
            _log.debug("生图场景上下文提取失败: %s", e)

    use_doubao = (request.image_model == "doubao")

    if use_doubao:
        # ── 豆包分支：完整场景上下文直接拼接，跳过 DeepSeek 扩写 ──
        parts = []
        if scene_name:
            parts.append(scene_name)
        if scene_content:
            parts.append(scene_content.replace('\n', '，'))
        if worldview_snippet:
            parts.append(worldview_snippet)
        if map_location:
            parts.append(map_location)
        if request.description:
            parts.append(request.description)
        if not parts:
            parts.append("神秘场景")
        suffix = f"，{style_anchor}" if style_anchor else ""
        full_prompt = "，".join(parts) + suffix + "，电影级光影，高质量"
    else:
        # ── Kolors 分支：DeepSeek 扩写为细节丰富的画面描述 ──
        context_parts = []
        if scene_name:
            context_parts.append(f"场景名：{scene_name}")
        if scene_content:
            context_parts.append(f"场景描述：{scene_content[:300]}")
        if worldview_snippet:
            context_parts.append(f"世界观：{worldview_snippet}")
        if map_location:
            context_parts.append(f"地点：{map_location}")
        if request.description:
            context_parts.append(f"GM 补充指导：{request.description}")
        if not context_parts:
            context_parts.append(request.description or "一个神秘的奇幻场景")

        context_text = "\n".join(context_parts)

        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": (
                        "你是一个专业的 AI 视觉提示词工程师。\n"
                        "请将接收到的跑团场景描述，扩写为一段高质量、细节丰富的中文画面描述，字数控制在 150-200 字以内。\n"
                        "【扩写强制规则】：\n"
                        "1. 视觉具象化：将抽象的剧情转化为具体的画面要素，细化人物的衣着材质、神态、肢体动作"
                        "2. 环境与空间：补充明确的空间关系、背景陈设和光影氛围"
                        "3. 画质控制锚点：必须在段落末尾追加提升画质的中文限定词汇"
                        "（例如：杰作，最高画质，电影级光影，8k分辨率，极致细节，CG级渲染，景深）。\n"
                        "【禁止事项】：\n"
                        "- 不要出现对话、引号内的文字（模型会尝试渲染文字到画面上）\n"
                        "- 不要出现「这是」「画面中」等元描述，不要任何前缀或解释\n"
                        "请直接输出扩写后的中文段落。"
                    )},
                    {"role": "user", "content": context_text}
                ],
                temperature=0.7,
                max_tokens=350,
            )
            scene_prompt = resp.choices[0].message.content.strip().strip('"').replace('\n', '，')
        except Exception as e:
            _log.warning("生图 prompt 生成失败，使用场景名兜底: %s", e)
            scene_prompt = scene_name or request.description or "神秘的奇幻场景，戏剧性光影"

        full_prompt = f"{scene_prompt}，{style_anchor}" if style_anchor else scene_prompt

    import httpx

    if use_doubao:
        # ── 豆包 Doubao-Seedream-5.0-lite（收费，使用 OpenAI 兼容客户端） ──
        if not DOUBAO_API_KEY:
            return {
                "status":  "error",
                "message": "未配置 DOUBAO_API_KEY，请在 .env 中填写豆包 API Key",
                "image_url": "",
                "prompt_used": full_prompt,
            }
        try:
            from openai import OpenAI as _OpenAI
            _doubao_client = _OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=DOUBAO_API_KEY,
            )
            resp = _doubao_client.images.generate(
                model="doubao-seedream-5-0-260128",
                prompt=full_prompt,
                size="3136x1344",
                response_format="url",
                extra_body={"watermark": False},
            )
            image_url = resp.data[0].url
            return {
                "status":    "success",
                "image_url": image_url,
                "prompt_used": full_prompt,
            }
        except Exception as e:
            return {
                "status":  "error",
                "message": f"豆包 API 错误：{e}",
                "image_url": "",
                "prompt_used": full_prompt,
            }
    else:
        # ── 硅基流动 Kolors（免费测试） ──
        api_url = "https://api.siliconflow.cn/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":           "Kwai-Kolors/Kolors",
            "prompt":          full_prompt,
            "negative_prompt": "模糊，低质量，水印，文字，变形，多余肢体，丑陋，噪点",
            "image_size":      "1024x576",
            "num_inference_steps": 25,
            "guidance_scale":  7.5,
            "seed":            random.randint(1, 2147483647),
        }

    try:
        with httpx.Client(timeout=90.0) as http:
            r = http.post(api_url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

        img_item = data["images"][0] if "images" in data else data["data"][0]

        if "url" in img_item and img_item["url"]:
            image_url = img_item["url"]
        elif "b64_json" in img_item and img_item["b64_json"]:
            b64 = img_item["b64_json"]
            image_url = f"data:image/png;base64,{b64}"
        else:
            raise ValueError("响应中未找到图片数据")

        return {
            "status":    "success",
            "image_url": image_url,
            "prompt_used": full_prompt,
        }

    except httpx.HTTPStatusError as e:
        provider = "豆包" if use_doubao else "硅基流动"
        return {
            "status":  "error",
            "message": f"{provider} API 错误 {e.response.status_code}：{e.response.text[:200]}",
            "image_url": "",
            "prompt_used": full_prompt,
        }
    except Exception as e:
        return {
            "status":  "error",
            "message": str(e),
            "image_url": "",
            "prompt_used": full_prompt,
        }


# ---------------------------------------------------------
# API 接口：战报/小说导出
# ---------------------------------------------------------
@app.post("/api/ai/export-battle-report")
def export_battle_report():
    """将 session_memory 喂给 AI，润色为一篇奇幻小说/战报（Markdown）。"""
    conn = get_db_connection()
    mem_row = conn.execute("SELECT value FROM system_state WHERE key='session_memory'").fetchone()
    wv_row  = conn.execute("SELECT value FROM system_state WHERE key='worldview'").fetchone()
    chars   = conn.execute("SELECT name, role FROM characters").fetchall()
    conn.close()

    memory   = mem_row["value"] if mem_row else "无记录"
    worldview = wv_row["value"] if wv_row else ""
    char_list = "、".join([c["name"] for c in chars]) or "未知角色"

    system_prompt = (
        "你是一位才华横溢的奇幻小说作者。请将以下跑团流水账润色为一篇排版精美、文笔优美的奇幻战报/小说章节。"
        "要求："
        "- 使用 Markdown 格式，包含标题、段落分节"
        "- 保留所有关键事件、人名、地点，不得改变剧情走向"
        "- 文笔生动，有代入感，适当补充细节描写"
        "- 结尾加一句「今日战报完」"
        f"【世界观背景】{worldview}"
        f"【主要角色】{char_list}"
    )

    try:
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"以下是今晚的跑团记录，请润色：{memory}"}
            ],
            temperature=0.8,
            max_tokens=2000,
        )
        report = resp.choices[0].message.content.strip()
        # 同时写入文件
        report_dir = os.path.join(BASE_DIR, "battle_report")
        os.makedirs(report_dir, exist_ok=True)
        filename = f"battle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(report_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        return {"status": "success", "report": report, "filename": filename}
    except Exception as e:
        return {"status": "error", "message": str(e), "report": ""}




# =============================================================
# 【多时间线 CRUD】：已迁移至 timeline.py
# _tl_append_memory 通过顶部 import 引入
# =============================================================

@app.post("/api/timelines/{tl_id}/dynamic-options")
def timeline_dynamic_options(tl_id: int, req: TimelineDynamicRequest):
    """
    时间线分支推演——委托给 agent.py 的统一推演引擎。
    时间线专属上下文在此构建，AI 调用和后处理由 agent 模块完成。
    """
    from agent import _call_ai, _build_dynamic_system_prompt, _post_process_dynamic_result

    conn = get_db_connection()
    tl = conn.execute("SELECT * FROM timelines WHERE id=?", (tl_id,)).fetchone()
    if not tl:
        conn.close()
        return {"status": "error", "message": "时间线不存在"}

    tl_memory = tl["memory"] or ""
    tl_char_ids = [int(x) for x in tl["char_ids"].split(",") if x.strip().isdigit()]

    worldview_row = conn.execute("SELECT value FROM system_state WHERE key='worldview'").fetchone()
    worldview = worldview_row["value"] if worldview_row else ""

    combined = f"{req.scene_name} {req.content} {req.player_action}".lower()
    lores = conn.execute("SELECT keywords, content FROM lorebook").fetchall()
    injected = [f"[{l['keywords']}]: {l['content']}" for l in lores
                if any(k.strip().lower() in combined for k in l["keywords"].split(","))]
    relevant_lore = "\n".join(injected) or "无"

    tl_chars = conn.execute(
        f"SELECT * FROM characters WHERE id IN ({','.join('?'*len(tl_char_ids)) if tl_char_ids else '0'})",
        tl_char_ids if tl_char_ids else []
    ).fetchall()
    party_status = "\n".join(
        [f"- {c['name']} (HP:{c['hp']}, SAN:{c['san']}) | {c['inventory'] or '无'}" for c in tl_chars]
    ) or "（本时间线暂无绑定角色）"

    world_entities_text = _get_world_entities_text(conn, req.scene_name, req.content, req.player_action)
    rag_context = _rag_retrieve(conn, f"{req.scene_name} {req.content} {req.player_action}")
    l1_context = _l1_get_working_context(conn, tl_id)

    try:
        tl_room_id = tl["current_room_id"] if tl["current_room_id"] else None
        map_context = _get_map_context(conn, tl_room_id)
    except Exception:
        map_context = ""

    system_prompt = _build_dynamic_system_prompt(
        worldview, party_status, relevant_lore, tl_memory,
        l1_context, world_entities_text, rag_context, map_context,
        is_timeline=True, tl_label=tl["label"],
        action_type=req.action_type,
        gm_correction=""
    )
    user_prompt = f"场景：{req.scene_name}\n内容：{req.content}\n玩家动作：{req.player_action}\n先思考，再生成分支。"

    try:
        ai_result = _call_ai(system_prompt, user_prompt, temperature=0.8, max_tokens=2500, json_mode=True)
        if ai_result.startswith("```"):
            ai_result = ai_result.split("```")[1]
            if ai_result.startswith("json"): ai_result = ai_result[4:]
        parsed = json_repair.loads(ai_result.strip())

        new_options, spawned_npc, applied_changes, map_result, thought_process, entity_updates_text = \
            _post_process_dynamic_result(
                conn, parsed, req.scene_name, req.player_action,
                req.current_node_id, timeline_id=tl_id, tl_id_for_memory=tl_id,
                action_type=req.action_type
            )
        conn.close()
        return {"status": "success", "new_options": new_options,
                "spawned_npc": spawned_npc, "stat_changes": applied_changes,
                "thought_process": thought_process,
                "entity_updates": entity_updates_text,
                "map_result": map_result}
    except Exception as e:
        conn.close()
        return {"status": "error", "message": str(e), "new_options": []}


# =============================================================
# 【世界实体】：已迁移至 entity.py（通过 app.include_router(entity_router) 自动注册）
# =============================================================

# =============================================================
# 【时间线 CRUD/合并】：已迁移至 timeline.py（通过 app.include_router(timeline_router) 自动注册）
# =============================================================



# =============================================================
# 【RAG 知识库】：已迁移至 rag.py（通过 app.include_router(rag_router) 自动注册）
# =============================================================

# =============================================================
# 【地图系统】：已迁移至 map.py（通过 app.include_router(map_router) 自动注册）
# =============================================================

# ---------------------------------------------------------
# 【地图-AI 联动】：节点→房间绑定查询（供前端 jumpToNode 调用）
# ---------------------------------------------------------
@app.get("/api/map/room-by-node/{node_id}")
def get_room_by_node(node_id: int):
    """查询某个剧情节点绑定的地图房间。jumpToNode 时自动调用。"""
    with safe_db() as conn:
        room = conn.execute(
            "SELECT id, label FROM map_rooms WHERE node_id=? LIMIT 1", (node_id,)
        ).fetchone()
    if room:
        return {"status": "success", "room_id": room["id"], "label": room["label"]}
    return {"status": "success", "room_id": None}
# =============================================================

# ---------------------------------------------------------
# API 接口：投屏端同步（REST 轮询 + WebSocket 实时推送）
# ---------------------------------------------------------

# 【WebSocket 投屏】：连接管理 + 广播
import asyncio

_ws_clients: set[WebSocket] = set()

@app.websocket("/ws/player")
async def ws_player_endpoint(websocket: WebSocket):
    """
    投屏端 WebSocket 连接。
    连接建立后立即推送一次完整状态，之后 GM 每次 push_player_state 时自动广播。
    客户端只需监听 onmessage，不需要主动发消息。
    """
    await websocket.accept()
    _ws_clients.add(websocket)
    _log.info("投屏 WebSocket 连接建立（当前 %d 个客户端）", len(_ws_clients))
    try:
        # 连接建立后立即推送当前状态
        state = _build_player_state_snapshot()
        await websocket.send_json(state)
        # 保持连接，等待客户端断开（客户端不需要发数据，但 WebSocket 需要 recv 循环保持心跳）
        while True:
            # 等待客户端消息（通常不会收到，但需要这个循环来检测断开）
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # 心跳：每 30 秒发一个 ping（客户端无需处理）
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        _log.debug("投屏 WebSocket 异常断开: %s", e)
    finally:
        _ws_clients.discard(websocket)
        _log.info("投屏 WebSocket 连接断开（剩余 %d 个客户端）", len(_ws_clients))


def _build_player_state_snapshot() -> dict:
    """构建投屏端需要的完整状态快照（场景+角色+BGM）。"""
    with safe_db() as conn:
        def _get(k):
            r = conn.execute("SELECT value FROM system_state WHERE key=?", (k,)).fetchone()
            return r["value"] if r else ""

        scene_id_raw = _get("player_current_scene_id")
        scene_id = int(scene_id_raw) if scene_id_raw and scene_id_raw.isdigit() else None

        # 如果有当前场景，附带场景详情
        current_scene = None
        if scene_id:
            node = conn.execute("SELECT * FROM nodes WHERE id=?", (scene_id,)).fetchone()
            if node:
                options = [dict(o) for o in conn.execute(
                    "SELECT * FROM options WHERE node_id=?", (scene_id,)
                ).fetchall()]
                current_scene = {**dict(node), "options": options}

        characters = [dict(c) for c in conn.execute("SELECT * FROM characters").fetchall()]

    return {
        "type": "state_update",
        "current_scene_id": scene_id,
        "current_scene": current_scene,
        "scene_image":   _get("player_scene_image") if scene_id else "",
        "scene_prompt":  _get("player_scene_prompt") if scene_id else "",
        "scene_ai_text": _get("player_scene_ai_text") if scene_id else "",
        "bgm_url": _get("player_bgm_url") if scene_id else "",
        "bgm_name": _get("player_bgm_name") if scene_id else "",
        "characters": characters,
    }


async def _broadcast_player_state():
    """向所有已连接的投屏端 WebSocket 客户端广播当前状态。"""
    if not _ws_clients:
        return
    state = _build_player_state_snapshot()
    dead_clients = set()
    for ws in _ws_clients.copy():
        try:
            await ws.send_json(state)
        except Exception:
            dead_clients.add(ws)
    # 清理已断开的连接
    _ws_clients.difference_update(dead_clients)


# ---------------------------------------------------------
# 场景跳转日志：记录玩家通过预设选项进入节点的行为
# ---------------------------------------------------------
class SceneVisitRequest(BaseModel):
    node_id:      int
    node_name:    str
    option_text:  str = ""   # 玩家点击的选项文本，为空时表示直接跳转

@app.post("/api/game/log-scene-visit")
def log_scene_visit(req: SceneVisitRequest):
    # 只有通过预设选项跳转才记录，直接跳转不写记忆流
    if not req.option_text.strip():
        return {"status": "skipped"}
    conn = get_db_connection()
    log = f"玩家选择「{req.option_text}」→ 进入场景【{req.node_name}】"
    append_to_memory(conn, log)
    conn.close()
    return {"status": "success"}


class PlayerStateRequest(BaseModel):
    current_scene_id: int = 0
    scene_image:      str = ""
    scene_prompt:     str = ""
    scene_ai_text:    str = ""
    bgm_url:          str = ""
    bgm_name:         str = ""

class CheckpointRequest(BaseModel):
    from_node_id: int

@app.post("/api/game/checkpoint")
def create_checkpoint(req: CheckpointRequest):
    """选项跳转前保存当前游戏状态快照（仅快照跳转时会变动的9张表）。"""
    with safe_db() as conn:
        def rows(sql, *args):
            return [dict(r) for r in conn.execute(sql, args).fetchall()]

        snap = {
            "characters":     rows("SELECT * FROM characters"),
            "world_entities": rows("SELECT * FROM world_entities"),
            "memory_l1_max_id": (conn.execute("SELECT COALESCE(MAX(id),0) FROM memory_l1").fetchone()[0]),
            "timelines":      rows("SELECT * FROM timelines"),
            "node_expanded":  {str(r["id"]): r["expanded_content"]
                               for r in conn.execute("SELECT id,expanded_content FROM nodes").fetchall()},
            "triggers":       rows("SELECT id,fired,fire_count,last_fired_at FROM triggers"),
            "pending_effects": rows("SELECT * FROM pending_effects"),
            "session_memory": (conn.execute("SELECT value FROM system_state WHERE key='session_memory'").fetchone() or {"value":""})["value"],
            "map_rooms_state": {str(r["id"]): r["state"]
                                for r in conn.execute("SELECT id,state FROM map_rooms").fetchall()},
        }

        conn.execute(
            "INSERT INTO game_checkpoints (from_node_id, snapshot) VALUES (?,?)",
            (req.from_node_id, json.dumps(snap, ensure_ascii=False))
        )
        # 保留最近10条，自动淘汰最旧的
        conn.execute("""
            DELETE FROM game_checkpoints
            WHERE id NOT IN (SELECT id FROM game_checkpoints ORDER BY id DESC LIMIT 10)
        """)
        conn.commit()
    return {"status": "success"}


@app.post("/api/game/rollback")
def rollback_checkpoint():
    """恢复最近一次快照，返回应导航到的节点ID。快照消费后自动删除。"""
    with safe_db() as conn:
        row = conn.execute(
            "SELECT id, from_node_id, snapshot FROM game_checkpoints ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return {"status": "error", "message": "没有可用的快照"}

        cp_id       = row["id"]
        from_node   = row["from_node_id"]
        snap        = json.loads(row["snapshot"])

        # ── characters ──────────────────────────────────────────
        conn.execute("DELETE FROM characters")
        for c in snap["characters"]:
            conn.execute(
                "INSERT INTO characters (id,name,role,hp,san,inventory,status) VALUES (?,?,?,?,?,?,?)",
                (c["id"], c["name"], c["role"], c["hp"], c["san"],
                 c.get("inventory",""), c.get("status","active"))
            )

        # ── world_entities ───────────────────────────────────────
        conn.execute("DELETE FROM world_entities")
        for e in snap["world_entities"]:
            conn.execute(
                "INSERT INTO world_entities (id,entity_type,name,location,status,last_seen_by,state_desc,updated_at,room_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (e["id"], e["entity_type"], e["name"], e["location"],
                 e["status"], e["last_seen_by"], e["state_desc"],
                 e["updated_at"], e.get("room_id"))
            )

        # ── memory_l1（删除快照之后新增的条目）──────────────────
        conn.execute("DELETE FROM memory_l1 WHERE id > ?", (snap["memory_l1_max_id"],))

        # ── timelines ────────────────────────────────────────────
        conn.execute("DELETE FROM timelines")
        for tl in snap["timelines"]:
            conn.execute(
                "INSERT INTO timelines (id,label,color,current_node_id,current_room_id,memory,char_ids,status,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (tl["id"], tl["label"], tl["color"],
                 tl["current_node_id"], tl.get("current_room_id"),
                 tl["memory"], tl["char_ids"], tl["status"], tl["created_at"])
            )

        # ── nodes.expanded_content ───────────────────────────────
        for node_id_str, content in snap["node_expanded"].items():
            conn.execute("UPDATE nodes SET expanded_content=? WHERE id=?",
                         (content, int(node_id_str)))

        # ── triggers 状态字段 ─────────────────────────────────────
        for tr in snap["triggers"]:
            conn.execute(
                "UPDATE triggers SET fired=?,fire_count=?,last_fired_at=? WHERE id=?",
                (tr["fired"], tr.get("fire_count",0), tr.get("last_fired_at"), tr["id"])
            )

        # ── pending_effects ──────────────────────────────────────
        conn.execute("DELETE FROM pending_effects")
        for pe in snap["pending_effects"]:
            conn.execute(
                "INSERT INTO pending_effects (id,node_id,payload,created_at) VALUES (?,?,?,?)",
                (pe["id"], pe["node_id"], pe["payload"], pe["created_at"])
            )

        # ── session_memory ───────────────────────────────────────
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key,value) VALUES ('session_memory',?)",
            (snap["session_memory"],)
        )

        # ── map_rooms.state ──────────────────────────────────────
        for room_id_str, state in snap["map_rooms_state"].items():
            conn.execute("UPDATE map_rooms SET state=? WHERE id=?",
                         (state, int(room_id_str)))

        # 快照消费后删除
        conn.execute("DELETE FROM game_checkpoints WHERE id=?", (cp_id,))
        # 返回剩余快照数量，供前端更新按钮状态
        remaining = conn.execute("SELECT COUNT(*) FROM game_checkpoints").fetchone()[0]
        conn.commit()

    return {"status": "success", "restored_node_id": from_node, "remaining": remaining}

@app.get("/api/player/state")
def get_player_state():
    """REST 轮询接口（向后兼容，投屏端已升级为 WebSocket 则不再需要此接口）。"""
    with safe_db() as conn:
        def _get(k):
            r = conn.execute("SELECT value FROM system_state WHERE key=?", (k,)).fetchone()
            return r["value"] if r else ""
        scene_id_raw = _get("player_current_scene_id")
        result = {
            "current_scene_id": int(scene_id_raw) if scene_id_raw and scene_id_raw.isdigit() else None,
            "scene_image":      _get("player_scene_image"),
            "scene_prompt":     _get("player_scene_prompt"),
            "scene_ai_text":    _get("player_scene_ai_text"),
            "bgm_url":          _get("player_bgm_url"),
            "bgm_name":         _get("player_bgm_name"),
        }
    return result

@app.post("/api/player/state")
async def push_player_state(req: PlayerStateRequest):
    """GM 推送投屏状态。写入数据库后立即通过 WebSocket 广播给所有投屏端。"""
    with safe_db() as conn:
        conn.execute("INSERT OR REPLACE INTO system_state (key,value) VALUES ('player_current_scene_id',?)", (str(req.current_scene_id),))
        conn.execute("INSERT OR REPLACE INTO system_state (key,value) VALUES ('player_scene_image',?)",      (req.scene_image,))
        conn.execute("INSERT OR REPLACE INTO system_state (key,value) VALUES ('player_scene_prompt',?)",     (req.scene_prompt,))
        conn.execute("INSERT OR REPLACE INTO system_state (key,value) VALUES ('player_scene_ai_text',?)",    (req.scene_ai_text,))
        conn.execute("INSERT OR REPLACE INTO system_state (key,value) VALUES ('player_bgm_url',?)",          (req.bgm_url,))
        conn.execute("INSERT OR REPLACE INTO system_state (key,value) VALUES ('player_bgm_name',?)",         (req.bgm_name,))
        conn.commit()
    # WebSocket 广播（异步，不阻塞响应）
    await _broadcast_player_state()
    return {"status": "success"}

# ---------------------------------------------------------
# 【重点新增】：托管静态 HTML 文件，实现即插即用
# ---------------------------------------------------------
@app.get("/")
def serve_index():
    # 当访问 http://localhost:8000 时，直接返回 index.html
    path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(path): return FileResponse(path)
    return {"error": "未找到 index.html 文件，请确保它与 main.py 在同一目录下。"}

@app.get("/{filename}")
def serve_static(filename: str):
    # 【安全】：防止路径穿越攻击（如 ../../etc/passwd.html）
    # 1. 只允许纯文件名（不含目录分隔符）
    # 2. 只允许 .html 后缀
    # 3. 解析后的绝对路径必须仍在 BASE_DIR 内
    if "/" in filename or "\\" in filename or ".." in filename:
        raise fastapi.HTTPException(status_code=400, detail="非法文件名")
    if not filename.endswith(".html"):
        raise fastapi.HTTPException(status_code=404, detail="文件不存在")
    path = os.path.normpath(os.path.join(BASE_DIR, filename))
    if not path.startswith(os.path.normpath(BASE_DIR)):
        raise fastapi.HTTPException(status_code=403, detail="禁止访问")
    if not os.path.isfile(path):
        raise fastapi.HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path)

# ---------------------------------------------------------
# 程序入口
# ---------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    _log.info("=====================================================")
    _log.info("-Z.R.I.C 零界核心- 正在启动...")
    _log.info("当前工作目录: %s", BASE_DIR)
    _log.info("=====================================================")
    _log.info("请不要关闭此窗口！关闭窗口将停止游戏引擎。")
    
    def auto_open_browser():
        time.sleep(2)
        _log.info("正在自动为您打开浏览器...")
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=auto_open_browser, daemon=True).start()

    # 启动服务器 (使用 127.0.0.1 避免某些网络策略拦截)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")