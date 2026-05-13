> ⚠️ **项目状态：开发早期阶段 (Early Development)**
>
> 本项目目前处于早期开发阶段，功能尚不完善，API 可能会发生重大变化。
> **请勿在生产环境中使用！** 欢迎提交 Issue 或 Pull Request 参与贡献。

# 🌌Z.RIC — AI 驱动的 TRPG 引擎

DM在环的半自动文字冒险推演模拟器。AI 实时生成多分支剧情、自动管理 NPC 情绪、语义检索知识库裁判规则、场景感知生图、投屏同步给玩家。
只需要加载合适的剧本，无论是恋爱生活模拟还是规则怪谈剧本均能恰当推演。

## 🧠 Why Z.R.I.C

全称为 **Z**ero-boundary **R**oleplay **I**ntelligence **C**ore，中文名零界核心。发音大概是 Z-ric。

## 💡 它能做什么

- **AI 推演**：输入玩家行动，AI 生成 2-4 个分支结果（含 HP/SAN 变化、物品获取、地图移动），支持流式输出
- **多模型切换**：DeepSeek / Claude Opus 4.6 一键切换，请求级回退
- **三级记忆**：L1 短期工作区（最近 8 次推演）→ L2 实体档案 → L3 向量长期记忆，自动折叠不丢失
- **RAG 知识库**：上传世界观/规则文档，AI 推演时自动语义检索相关内容注入 prompt
- **NPC 情绪状态机**：信任/恐惧/烦躁三轴及断点反应，NPC 会根据剧情走向产生记忆并改变态度
- **NPC Persona 模式**：为 NPC 配置 MBTI 性格、怪癖与独立记忆，由 Claude 扮演角色实时对话
- **空间感知地图**：房间/通道拓扑，AI 推演时知道"你在哪、旁边有什么"，自动生长新地点
- **触发器系统**：scene / item / stat / ai 四种条件，hard 强制跳转 + soft 提示
- **多时间线**：分支平行推演，独立记忆，可合并
- **深入调查**：对当前场景一键 AI 扩写细节，结果实时推送至投屏端紫色框
- **场景生图**：双图源支持——Kolors（硅基流动，免费）和 Doubao-Seedream-5.0-lite（豆包，收费）；自动从场景扩写内容提取上下文生成插图
- **投屏系统**：WebSocket 实时推送场景文本/深入调查内容/图片/BGM 到玩家屏幕
- **手机通讯录界面**：`phone.html`，模拟 NPC 联系人与角色短信对话
- **剧本导入导出**：完整存档含向量索引，加载即用无需重建

## 🚀 快速开始

### 方法一：直接下载运行（推荐）

> 适合不熟悉命令行的用户，5 分钟内即可启动。

**前置要求**：安装 [Python 3.10+](https://www.python.org/downloads/)（安装时勾选 **Add Python to PATH**）

1. 点击本页右上角 **Code → Download ZIP**，解压到任意文件夹
2. 在解压后的文件夹中，复制 `.env.example`，重命名为 `.env`，用记事本打开，粘贴你的 API Key：
   ```
   DEEPSEEK_API_KEY=sk-你的key
   SILICONFLOW_API_KEY=sk-你的key
   ```
3. 双击 `启动.bat`，浏览器会自动打开引擎界面

API Key 获取：
- DeepSeek（必填）：https://platform.deepseek.com/
- 硅基流动（必填）：https://cloud.siliconflow.cn/
- Anthropic（可选，用于 Claude 推演）：https://console.anthropic.com/
- 豆包（可选，用于收费生图）：https://console.volcengine.com/ark

---

### 方法二：命令行安装

**环境要求**

- Python 3.10+
- DeepSeek API Key（必需，AI 推演核心）
- 硅基流动 API Key（必需，用于 embedding 和免费生图）

**安装**

```bash
git clone https://github.com/zRICGao/ZRIC-AI-TRPG-Engine.git
cd ZRIC-AI-TRPG-Engine
pip install -r requirements.txt
```

**配置 API Key**

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```
# 必填
DEEPSEEK_API_KEY=sk-你的key
SILICONFLOW_API_KEY=sk-你的key

# 可选：填写后可切换为 Claude Opus 4.6 推演 / NPC Persona 对话
ANTHROPIC_API_KEY=sk-ant-你的key

# 可选：填写后可使用豆包 Seedream 收费生图
DOUBAO_API_KEY=你的豆包key
```

**启动**

```bash
python main.py
```

---

浏览器会自动打开 `http://127.0.0.1:8000`。如果没有自动打开，手动访问该地址。

| 页面 | 地址 | 用途 |
|---|---|---|
| GM 控制台 | `http://127.0.0.1:8000` | 主操作界面 |
| 投屏端 | `http://127.0.0.1:8000/player.html` | 玩家大屏 |
| 通讯录 | `http://127.0.0.1:8000/phone.html` | NPC 短信模拟 |

## 🛡️ 项目结构

```
nexus-rpg-engine/
├── main.py          # 编排层：App 实例、剧本管理、模块挂载（1729 行）
├── agent.py         # AI 推演核心：多模型、prompt 构建、流式输出（1509 行）
├── trigger.py       # 触发器：scene/item/stat/ai 四种条件（480 行）
├── rag.py           # 向量知识库：切片、embedding、语义检索（547 行）
├── entity.py        # 世界实体：NPC 情绪状态机、AI 提取（435 行）
├── memory.py        # 三级记忆：L1 工作区、折叠、L3 淘汰（302 行）
├── map.py           # 空间地图：房间/通道、移动、自动生长（418 行）
├── timeline.py      # 多时间线：CRUD、合并、独立记忆（173 行）
├── logger.py        # 统一日志：控制台 + 文件双输出（41 行）
├── index.html       # GM 控制台：Vue 3 单页应用（4250 行）
├── player.html      # 投屏端：WebSocket 实时同步（448 行）
├── phone.html       # 通讯录：NPC 短信模拟界面（555 行）
├── .env.example     # API Key 配置模板
├── requirements.txt # Python 依赖
└── campaigns/       # 剧本文件夹（含示例及存档）
    ├── 盘步山规则怪谈/
    ├── 苏黎世恋曲/
    ├── 空白剧本/
    └── ...
```

## ✨ 使用流程

### 1. 加载剧本

启动后在左侧「选择剧本/存档」下拉框中选择剧本，点击「加载」。项目附带了多个示例剧本，包括规则怪谈「盘步山规则怪谈」和恋爱模拟「苏黎世恋曲」等（内容均不完整，仅作演示）。

### 2. 开始推演

选中一个场景节点，在右侧「AI 推演」面板输入玩家的行动描述，点击「推演」。AI 会生成 2-3 个分支结果，每个分支包含：

- 剧情文本（流式输出）
- HP / SAN 变化
- 物品获取/失去
- 地图移动指令
- 新 NPC 生成

点击合适的分支，副作用会自动执行，场景叙事自动扩写。

### 3. 深入调查与生图

在场景面板点击「深入调查」，AI 对当前场景骨架扩写细节，结果会实时推送到投屏端的紫色信息框。点击「生图」可根据扩写内容自动生成场景插图（免费用 Kolors，收费用豆包 Seedream）。

### 4. 投屏给玩家

推演或生图完成后，场景文本、深入调查内容、插图和 BGM 通过 WebSocket 实时同步到 `player.html`，玩家无需刷新页面。

### 5. 存档

点击顶部「存档」按钮，输入自定义名字，存档保存到 `campaigns/` 文件夹（含完整向量索引，下次加载无需重建）。

## 🏗️ 技术架构

- **后端**：FastAPI + SQLite（WAL 模式）+ 8 个业务模块
- **前端**：Vue 3 + Tailwind CSS（CDN，零构建）
- **AI 推演**：DeepSeek Chat / Claude Opus 4.6（OpenAI 兼容协议，一键切换）
- **Embedding**：硅基流动 BGE-M3（1024 维）
- **生图**：硅基流动 Kwai-Kolors（免费）/ 火山方舟 Doubao-Seedream-5.0-lite（收费）
- **通信**：REST + SSE 流式 + WebSocket 投屏

## 🌍 局域网联机

如果想让同一局域网内的玩家访问投屏端：

1. 查看你的局域网 IP（Windows: `ipconfig`，Mac/Linux: `ifconfig`）
2. 在 `.env` 中添加：
   ```
   ALLOWED_ORIGINS=http://你的IP:8000
   ```
3. 将 `main.py` 最后一行的 `host="127.0.0.1"` 改为 `host="0.0.0.0"`
4. 玩家访问 `http://你的IP:8000/player.html`

## 🎮 怎么玩

该引擎的主要运作模式：人负责构造基本规则和主世界观（背景故事、主要人物、主线、地图等），然后由 AI 承接玩家的自由行动推演剧情走向，摆脱传统电脑游戏的有限分支。

**触发器是收束时间线的关键**——在关键节点设置触发器，可以引导 AI 不过于发散，推动玩家走向「正确」的叙事节点（场景转换、软提示、结局触发等）。

本项目仍处于早期 Demo 阶段，存在大量不足和各种性能限制。欢迎提供各种形式的宝贵建议和帮助。

## 📝 许可证

使用 GPL v3 协议，详见 [LICENSE](LICENSE) 文件。

## ⚡ 致谢

- CTO: Claude　项目经理: Gemini　档案: DeepSeek　志愿者: 硅基流动
- 部分 BGM 资源来自 [Soundimage.org](https://soundimage.org/)（Eric Matyas），使用时请注明出处
- 盘步山剧本来自 抖音：汤姆要哈气了

---

## "世界已就绪，你准备好投掷骰子了吗？" 🎲
