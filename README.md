> ⚠️ **项目状态：开发早期阶段 (Early Development)**
> 
> 本项目目前处于早期开发阶段，功能尚不完善，API 可能会发生重大变化。
> **请勿在生产环境中使用！** 欢迎提交 Issue 或 Pull Request 参与贡献。

# 🌌Z.RIC — AI 驱动的 TRPG 引擎

DM在环的半自动文字冒险推演模拟器。AI 实时生成多分支剧情、自动管理 NPC 情绪、语义检索知识库裁判规则、场景感知生图、投屏同步给玩家。
只需要加载合适的剧本，无论是恋爱生活模拟和规则怪谈剧本均能恰当推演。

## 🧠Why Z.R.I.C

全称为 **Z**ero-boundary **R**oleplay **I**ntelligence **C**ore, 中文名零界核心。发音大概是Z-ric.
（其实是ZURICH的变体。

## 💡它能做什么

- **AI 推演**：输入玩家行动，AI 生成 2-4 个分支结果（含 HP/SAN 变化、物品获取、地图移动），支持流式输出
- **多模型切换**：DeepSeek / Claude 一键切换，请求级回退
- **三级记忆**：L1 短期工作区（最近 8 次推演）→ L2 实体档案 → L3 向量长期记忆，自动折叠不丢失
- **RAG 知识库**：上传世界观/规则文档，AI 推演时自动语义检索相关内容注入 prompt
- **NPC 情绪状态机**：信任/恐惧/烦躁三轴及断点反应，NPC 会根据剧情走向产生记忆并改变态度。
- **空间感知地图**：房间/通道拓扑，AI 推演时知道"你在哪、旁边有什么"，自动生长新地点
- **触发器系统**：场景/物品/属性/AI 四种条件，hard 强制跳转 + soft 提示
- **多时间线**：分支平行推演，独立记忆，可合并
- **场景生图**：Kolors 中文直出，自动从当前场景提取上下文生成插图
- **投屏系统**：WebSocket 实时推送场景/图片/BGM 到玩家屏幕
- **剧本导入导出**：完整存档含向量索引，加载即用无需重建

## 🚀快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key（必需）
- 硅基流动 API Key（必需，用于 embedding 和生图）

### 安装

```bash
git clone https://github.com/你的用户名/nexus-rpg-engine.git
cd nexus-rpg-engine
pip install -r requirements.txt
```

### 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```
DEEPSEEK_API_KEY=sk-你的key
SILICONFLOW_API_KEY=sk-你的key
```

API Key 获取：
- DeepSeek：https://platform.deepseek.com/
- 硅基流动：https://cloud.siliconflow.cn/
* DEEPSEEK_API_KEY: 用于场景推演与逻辑判断。
* SILICONFLOW_API_KEY: 用于 RAG 向量化（默认使用 BAAI/bge-m3 模型）。

### 启动

```bash
python main.py
```

浏览器会自动打开 `http://127.0.0.1:8000`。如果没有自动打开，手动访问这个地址或者双击根目录下的`index.html`。

投屏端（给玩家看的界面）：`http://127.0.0.1:8000/player.html`

## 🛡️项目结构 （3/19版，后续有更新）

```
nexus-rpg-engine/
├── main.py          # 编排层：App 实例、剧本管理、模块挂载（1786 行）
├── agent.py         # AI 推演核心：多模型、prompt 构建、流式输出（1186 行）
├── memory.py        # 三级记忆：L1 工作区、折叠、L3 淘汰（299 行）
├── trigger.py       # 触发器：scene/item/stat/ai 四种条件（265 行）
├── entity.py        # 世界实体：NPC 情绪状态机、AI 提取（385 行）
├── timeline.py      # 多时间线：CRUD、合并、独立记忆（213 行）
├── rag.py           # 向量知识库：切片、embedding、语义检索（521 行）
├── map.py           # 空间地图：房间/通道、移动、自动生长（500 行）
├── logger.py        # 统一日志：控制台 + 文件双输出（56 行）
├── index.html       # GM 控制台：Vue 3 单页应用（4241 行）
├── player.html      # 投屏端：WebSocket 实时同步（474 行）
├── .env.example     # API Key 配置模板
├── requirements.txt # Python 依赖
└── campaigns/       # 剧本文件夹
    └── 盘步山规则怪谈/  # 示例剧本
        ├── campaign.json
        ├── map.json
        └── knowledge/
```

## ✨使用流程

### 1. 加载剧本

启动后在左侧「选择剧本/存档」下拉框中选择剧本，点击「加载」。项目附带了两个**不**完整的剧本：「苏黎世恋歌」剧本和规则怪谈示例剧本「盘步山规则怪谈」。

### 2. 开始推演

选中一个场景节点，在右侧「AI 推演」面板输入玩家的行动描述，点击「推演」。AI 会生成 2-3 个分支结果，每个分支包含：
- 剧情文本
- HP/SAN 变化
- 物品获取/失去
- 地图移动指令
- 新 NPC 生成

点击你觉得合适的分支，副作用会自动执行。

### 3. 投屏给玩家

在场景面板点击「投屏」按钮推送当前场景。玩家在 `player.html` 页面实时看到场景文本、插图和 BGM。

### 4. 存档

点击顶部「存档」按钮，输入自定义名字，存档会保存到 `campaigns/` 文件夹（含完整的向量索引，下次加载无需重建）。

## 🏗️技术架构

- **后端**：FastAPI + SQLite（WAL 模式）+ 9 个业务模块
- **前端**：Vue 3 + Tailwind（CDN，零构建）
- **AI**：DeepSeek / Claude（OpenAI 兼容协议）
- **Embedding**：硅基流动 BGE-M3（1024 维）
- **生图**：硅基流动 Kwai-Kolors（中文原生理解）
- **通信**：REST + SSE 流式 + WebSocket 投屏
- **总计**：76 个 API 端点，约 10000 行代码

## 🌍局域网联机

如果想让同一局域网内的玩家访问投屏端：

1. 查看你的局域网 IP（Windows: `ipconfig`，Mac/Linux: `ifconfig`）
2. 在 `.env` 中添加：
   ```
   ALLOWED_ORIGINS=http://你的IP:8000
   ```
3. 将 `main.py` 最后一行的 `host="127.0.0.1"` 改为 `host="0.0.0.0"`
4. 玩家访问 `http://你的IP:8000/player.html`

## 🎮怎么玩
目前`（2026/3/20）`由于精力有限，没有时间再做完整的剧本和边际测试，匆匆发布。这里说一下我后续的方向。  
该引擎主要的运作模式是人负责基本规则和主世界观的构造（包括背景故事，主要人物，主线，地图，usw.），拜托传统电脑游戏中有限分支玩法，玩家自由决定行动并由AI推演剧情走向。  
如果需要体验完整剧本需要依赖大量**触发器**驱动。触发器是收束时间线的关键。触发器在这个引擎中承担包括场景转换，软提示，结局触发等关键作用。  
在关键节点设置触发器可以帮助AI不要过于发散，引导玩家走向‘正确’的推进。  
本项目依然处于极度前期的Demo阶段，仍然存在大量不足和各种性能限制。欢迎提供各种意义上的宝贵建议和帮助。  

## 📝许可证

- 使用GPL v3协议，详见 [LICENSE](LICENSE) 文件。

## ⚡致谢

- CTO: Claude 项目经理: Gemini 档案: Deepseek 志愿者: 硅基流动 
- 部分BGM 资源来自 [Soundimage.org](https://soundimage.org/)（Eric Matyas），使用时请注明出处
- 盘步山剧本来自 抖音：汤姆要哈气了


## “世界已就绪，你准备好投掷骰子了吗？” 🎲