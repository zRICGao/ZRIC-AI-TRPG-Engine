"""
Z.R.I.C 引擎 — 地图系统模块 (map.py)
独立的 FastAPI APIRouter，包含地图的全部数据模型、数据库操作、REST API 和辅助函数。
由 main.py 通过 app.include_router(map_router) 挂载。
"""

import fastapi
from fastapi import APIRouter
from pydantic import BaseModel
import sqlite3
import json
import os

# ---------------------------------------------------------
# Router 实例（main.py 中 include_router 时无需额外 prefix）
# ---------------------------------------------------------
map_router = APIRouter(tags=["地图系统"])


# ---------------------------------------------------------
# 数据库连接（由 main.py 在启动时注入）
# ---------------------------------------------------------
from contextlib import contextmanager
from logger import get_logger

_log = get_logger("map")

_db_file: str = ""

def set_db_file(path: str):
    """由 main.py 启动时调用，注入数据库路径。"""
    global _db_file
    _db_file = path

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
# 数据库表初始化
# ---------------------------------------------------------
def init_map_tables():
    """创建地图相关的数据库表（幂等）。由 main.py 的 init_db() 调用。"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS map_rooms (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        map_id      INTEGER NOT NULL DEFAULT 1,
        label       TEXT    NOT NULL DEFAULT '未命名',
        x           REAL    NOT NULL DEFAULT 0,
        y           REAL    NOT NULL DEFAULT 0,
        w           REAL    NOT NULL DEFAULT 120,
        h           REAL    NOT NULL DEFAULT 80,
        description TEXT    NOT NULL DEFAULT '',
        state       TEXT    NOT NULL DEFAULT 'unknown',
        color       TEXT    NOT NULL DEFAULT '#1e3a2f',
        node_id     INTEGER,
        floor       INTEGER NOT NULL DEFAULT 1
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS map_edges (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        map_id      INTEGER NOT NULL DEFAULT 1,
        from_id     INTEGER NOT NULL,
        to_id       INTEGER NOT NULL,
        label       TEXT    NOT NULL DEFAULT '',
        locked      INTEGER NOT NULL DEFAULT 0,
        key_item    TEXT    NOT NULL DEFAULT '',
        edge_type   TEXT    NOT NULL DEFAULT 'normal',
        FOREIGN KEY(from_id) REFERENCES map_rooms(id) ON DELETE CASCADE,
        FOREIGN KEY(to_id)   REFERENCES map_rooms(id) ON DELETE CASCADE
    )''')

    # 平滑升级
    try: cursor.execute("ALTER TABLE map_rooms ADD COLUMN floor INTEGER NOT NULL DEFAULT 1")
    except sqlite3.OperationalError: pass
    try: cursor.execute("ALTER TABLE map_edges ADD COLUMN edge_type TEXT NOT NULL DEFAULT 'normal'")
    except sqlite3.OperationalError: pass

    conn.commit()
    conn.close()


# ---------------------------------------------------------
# Pydantic 数据模型
# ---------------------------------------------------------
class MapRoomCreateRequest(BaseModel):
    map_id:      int    = 1
    label:       str    = "未命名"
    x:           float  = 0
    y:           float  = 0
    w:           float  = 120
    h:           float  = 80
    description: str    = ""
    state:       str    = "unknown"
    color:       str    = "#1e3a2f"
    node_id:     int | None = None
    floor:       int    = 1

class MapRoomUpdateRequest(BaseModel):
    label:       str
    x:           float
    y:           float
    w:           float
    h:           float
    description: str
    state:       str
    color:       str
    node_id:     int | None = None
    floor:       int = 1

class MapEdgeCreateRequest(BaseModel):
    map_id:    int   = 1
    from_id:   int
    to_id:     int
    label:     str   = ""
    locked:    bool  = False
    key_item:  str   = ""
    edge_type: str   = "normal"

class MapEdgeUpdateRequest(BaseModel):
    label:     str
    locked:    bool
    key_item:  str
    edge_type: str = "normal"

class MapAutoRoomRequest(BaseModel):
    """推演完成后自动生长地图用"""
    map_id:         int
    parent_room_id: int
    label:          str
    node_id:        int
    description:    str = ""


# ---------------------------------------------------------
# 辅助函数（供 main.py 的 AI 推演上下文调用）
# ---------------------------------------------------------
def get_map_context(conn, room_id: int | None) -> str:
    """
    读取当前房间及其相邻房间，生成供 AI 推演使用的空间感知文本。
    room_id 为 None 或房间不存在时返回空字符串（零影响）。
    支持多楼层：跨楼层连接标注楼层变化方向。
    """
    if not room_id:
        return ""
    room = conn.execute("SELECT * FROM map_rooms WHERE id=?", (room_id,)).fetchone()
    if not room:
        return ""

    floor = room["floor"] if "floor" in room.keys() else 1
    lines = [f"当前位置：{room['label']}（第{floor}层，{room['description'] or '无额外描述'}）"]

    # 相邻房间（双向边），含 edge_type
    edges = conn.execute(
        "SELECT me.*, "
        "  CASE WHEN me.from_id=? THEN me.to_id ELSE me.from_id END AS neighbor_id "
        "FROM map_edges me "
        "WHERE me.from_id=? OR me.to_id=?",
        (room_id, room_id, room_id)
    ).fetchall()

    neighbor_descs = []
    for e in edges:
        nb = conn.execute(
            "SELECT label, state, floor FROM map_rooms WHERE id=?",
            (e["neighbor_id"],)
        ).fetchone()
        if not nb:
            continue

        lock_note  = f"（上锁，需：{e['key_item']}）" if e["locked"] else ""
        dir_note   = f"{e['label']}→" if e["label"] else "→"
        state_map  = {"unknown": "未探索", "explored": "已探索",
                      "locked": "封锁", "active": "当前"}
        nb_state   = state_map.get(nb["state"], nb["state"])
        nb_floor   = nb["floor"] if "floor" in nb.keys() else 1

        etype = e["edge_type"] if "edge_type" in e.keys() else "normal"
        if nb_floor != floor:
            direction = "上" if nb_floor > floor else "下"
            type_label = {
                "stairs":   f"楼梯（{direction}至第{nb_floor}层）",
                "elevator": f"电梯（{direction}至第{nb_floor}层）",
                "portal":   f"传送门（→第{nb_floor}层）",
            }.get(etype, f"通道（{direction}至第{nb_floor}层）")
            neighbor_descs.append(
                f"{dir_note}{nb['label']}[{type_label}]（{nb_state}）{lock_note}"
            )
        else:
            neighbor_descs.append(
                f"{dir_note}{nb['label']}（{nb_state}）{lock_note}"
            )

    if neighbor_descs:
        lines.append("可通往：" + " / ".join(neighbor_descs))
    else:
        lines.append("此处是死胡同，没有其他出口。")

    # 同一地图同一楼层的已知状态
    known = conn.execute(
        "SELECT label, state, floor FROM map_rooms "
        "WHERE map_id=? AND id!=? AND state!='unknown' ORDER BY floor, id",
        (room["map_id"], room_id)
    ).fetchall()
    if known:
        grouped = {}
        for r in known[:12]:
            fl = r["floor"] if "floor" in r.keys() else 1
            grouped.setdefault(fl, []).append(f"{r['label']}[{r['state']}]")
        parts = []
        for fl in sorted(grouped):
            prefix = f"第{fl}层：" if len(grouped) > 1 else ""
            parts.append(prefix + " / ".join(grouped[fl]))
        lines.append("已知地图状态：" + " | ".join(parts))

    return "\n".join(lines)


def auto_place_room(conn, map_id: int, parent_room_id: int,
                    label: str, node_id: int, description: str = "") -> int | None:
    """
    推演时自动在父房间旁边生长一个新房间。
    按 右→下→左→上 四个方向依次尝试，找到空位后插入。
    返回新房间 id，无法放置时返回 None（静默失败）。
    """
    parent = conn.execute(
        "SELECT x, y, w, h FROM map_rooms WHERE id=?", (parent_room_id,)
    ).fetchone()
    if not parent:
        return None

    GRID = 40
    W, H = 120, 80
    GAP  = GRID

    offsets = [
        ( parent["w"] + GAP,  0            ),
        ( 0,                  parent["h"] + GAP),
        (-(W + GAP),          0            ),
        ( 0,                 -(H + GAP)    ),
    ]

    import math
    for dx, dy in offsets:
        nx = round((parent["x"] + dx) / GRID) * GRID
        ny = round((parent["y"] + dy) / GRID) * GRID
        clash = conn.execute(
            "SELECT id FROM map_rooms WHERE map_id=? "
            "AND x < ? AND x+w > ? AND y < ? AND y+h > ?",
            (map_id, nx + W, nx, ny + H, ny)
        ).fetchone()
        if not clash:
            cur = conn.execute(
                "INSERT INTO map_rooms "
                "(map_id, label, x, y, w, h, description, state, node_id) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (map_id, label[:30], nx, ny, W, H,
                 description[:200], "unknown", node_id)
            )
            new_id = cur.lastrowid
            conn.execute(
                "INSERT INTO map_edges (map_id, from_id, to_id) VALUES (?,?,?)",
                (map_id, parent_room_id, new_id)
            )
            conn.commit()
            return new_id

    return None


# ---------------------------------------------------------
# 地图数据导入 / 导出（独立文件 map.json）
# ---------------------------------------------------------
def export_map_data(conn) -> dict:
    """导出地图数据为字典，供独立 map.json 文件保存。"""
    rooms = [dict(r) for r in conn.execute(
        "SELECT * FROM map_rooms ORDER BY floor, id"
    ).fetchall()]
    edges = [dict(e) for e in conn.execute(
        "SELECT * FROM map_edges ORDER BY id"
    ).fetchall()]
    return {"map_rooms": rooms, "map_edges": edges}


def import_map_data(conn, data: dict):
    """从 map.json 字典导入地图数据。调用前应先清空旧数据。"""
    cursor = conn.cursor()
    for rm in data.get("map_rooms", []):
        cursor.execute(
            "INSERT INTO map_rooms (id, map_id, label, x, y, w, h, description, state, color, node_id, floor) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (rm.get("id"), rm.get("map_id", 1), rm.get("label", ""),
             rm.get("x", 0), rm.get("y", 0),
             rm.get("w", 120), rm.get("h", 80),
             rm.get("description", ""), rm.get("state", "unknown"),
             rm.get("color", "#1e3a2f"), rm.get("node_id"),
             rm.get("floor", 1))
        )
    for eg in data.get("map_edges", []):
        cursor.execute(
            "INSERT INTO map_edges (id, map_id, from_id, to_id, label, locked, key_item, edge_type) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (eg.get("id"), eg.get("map_id", 1),
             eg.get("from_id"), eg.get("to_id"),
             eg.get("label", ""), eg.get("locked", 0),
             eg.get("key_item", ""), eg.get("edge_type", "normal"))
        )
    conn.commit()


def clear_map_data(conn):
    """清空地图数据表。"""
    conn.execute("DELETE FROM map_rooms")
    conn.execute("DELETE FROM map_edges")
    try:
        conn.execute("UPDATE sqlite_sequence SET seq=0 WHERE name='map_rooms'")
        conn.execute("UPDATE sqlite_sequence SET seq=0 WHERE name='map_edges'")
    except sqlite3.OperationalError:
        pass
    conn.commit()


# ---------------------------------------------------------
# REST API 端点
# ---------------------------------------------------------
@map_router.get("/api/map/rooms")
def map_get_rooms(map_id: int = 1):
    conn = get_db_connection()
    rooms = [dict(r) for r in conn.execute(
        "SELECT * FROM map_rooms WHERE map_id=? ORDER BY floor, id", (map_id,)
    ).fetchall()]
    edges = [dict(e) for e in conn.execute(
        "SELECT * FROM map_edges WHERE map_id=? ORDER BY id", (map_id,)
    ).fetchall()]
    for r in rooms:
        r.setdefault("floor", 1)
    for e in edges:
        e.setdefault("edge_type", "normal")
    conn.close()
    floors = sorted(set(r["floor"] for r in rooms)) or [1]
    return {"status": "success", "rooms": rooms, "edges": edges, "floors": floors}


@map_router.post("/api/map/rooms")
def map_create_room(req: MapRoomCreateRequest):
    conn = get_db_connection()
    cur = conn.execute(
        "INSERT INTO map_rooms (map_id,label,x,y,w,h,description,state,color,node_id,floor) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (req.map_id, req.label[:40], req.x, req.y, req.w, req.h,
         req.description[:300], req.state, req.color[:20], req.node_id, req.floor)
    )
    new_id = cur.lastrowid
    conn.commit(); conn.close()
    return {"status": "success", "id": new_id}


@map_router.put("/api/map/rooms/{room_id}")
def map_update_room(room_id: int, req: MapRoomUpdateRequest):
    conn = get_db_connection()
    conn.execute(
        "UPDATE map_rooms SET label=?,x=?,y=?,w=?,h=?,description=?,state=?,color=?,node_id=?,floor=? "
        "WHERE id=?",
        (req.label[:40], req.x, req.y, req.w, req.h,
         req.description[:300], req.state, req.color[:20], req.node_id, req.floor, room_id)
    )
    conn.commit(); conn.close()
    return {"status": "success"}


@map_router.delete("/api/map/rooms/{room_id}")
def map_delete_room(room_id: int):
    conn = get_db_connection()
    conn.execute("DELETE FROM map_rooms WHERE id=?", (room_id,))
    conn.execute("DELETE FROM map_edges WHERE from_id=? OR to_id=?", (room_id, room_id))
    conn.commit(); conn.close()
    return {"status": "success"}


@map_router.post("/api/map/edges")
def map_create_edge(req: MapEdgeCreateRequest):
    conn = get_db_connection()
    exists = conn.execute(
        "SELECT id FROM map_edges WHERE map_id=? AND "
        "((from_id=? AND to_id=?) OR (from_id=? AND to_id=?))",
        (req.map_id, req.from_id, req.to_id, req.to_id, req.from_id)
    ).fetchone()
    if exists:
        conn.close()
        return {"status": "exists", "id": exists["id"]}
    cur = conn.execute(
        "INSERT INTO map_edges (map_id,from_id,to_id,label,locked,key_item,edge_type) "
        "VALUES (?,?,?,?,?,?,?)",
        (req.map_id, req.from_id, req.to_id,
         req.label[:40], int(req.locked), req.key_item[:60], req.edge_type)
    )
    new_id = cur.lastrowid
    conn.commit(); conn.close()
    return {"status": "success", "id": new_id}


@map_router.put("/api/map/edges/{edge_id}")
def map_update_edge(edge_id: int, req: MapEdgeUpdateRequest):
    conn = get_db_connection()
    conn.execute(
        "UPDATE map_edges SET label=?,locked=?,key_item=?,edge_type=? WHERE id=?",
        (req.label[:40], int(req.locked), req.key_item[:60], req.edge_type, edge_id)
    )
    conn.commit(); conn.close()
    return {"status": "success"}


@map_router.post("/api/map/move")
def map_move_to_room(room_id: int, timeline_id: int | None = None):
    """
    玩家移动到指定房间，更新当前位置并标记为已探索。
    timeline_id 不为空时只更新该时间线的坐标（分头行动模式）；
    为空时更新全局 current_room_id（主线模式）。
    """
    conn = get_db_connection()
    room = conn.execute("SELECT * FROM map_rooms WHERE id=?", (room_id,)).fetchone()
    if not room:
        conn.close()
        raise fastapi.HTTPException(status_code=404, detail="房间不存在")
    conn.execute("UPDATE map_rooms SET state='explored' WHERE id=?", (room_id,))
    if timeline_id:
        conn.execute(
            "UPDATE timelines SET current_room_id=? WHERE id=?", (room_id, timeline_id)
        )
    else:
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key,value) VALUES ('current_room_id',?)",
            (str(room_id),)
        )
    conn.commit(); conn.close()
    return {"status": "success", "room_id": room_id, "label": room["label"]}


@map_router.post("/api/map/auto-room")
def map_auto_room(req: MapAutoRoomRequest):
    """推演时自动在父房间旁边生长新房间。"""
    conn = get_db_connection()
    new_id = auto_place_room(
        conn, req.map_id, req.parent_room_id,
        req.label, req.node_id, req.description
    )
    conn.close()
    if new_id:
        return {"status": "success", "id": new_id}
    return {"status": "skipped", "message": "四个方向均被占用"}


@map_router.put("/api/map/rooms/{room_id}/state")
def map_set_room_state(room_id: int, state: str):
    """快速更新房间状态（unknown/explored/locked/active）。"""
    if state not in ("unknown", "explored", "locked", "active"):
        raise fastapi.HTTPException(status_code=400, detail="非法状态值")
    conn = get_db_connection()
    conn.execute("UPDATE map_rooms SET state=? WHERE id=?", (state, room_id))
    conn.commit(); conn.close()
    return {"status": "success"}


# ---------------------------------------------------------
# 独立地图文件 I/O API（前端可直接调用）
# ---------------------------------------------------------
@map_router.post("/api/map/export-file")
def map_export_file():
    """将当前地图导出为独立 JSON 文件。"""
    conn = get_db_connection()
    data = export_map_data(conn)
    conn.close()
    return {"status": "success", "data": data}


@map_router.post("/api/map/import-file")
def map_import_file(data: dict):
    """从前端上传的 JSON 导入地图数据（先清空旧地图）。"""
    conn = get_db_connection()
    clear_map_data(conn)
    import_map_data(conn, data)
    conn.close()
    return {"status": "success"}
