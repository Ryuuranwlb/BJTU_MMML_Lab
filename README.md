# Pangu Agent

<div align="center">

**基于 LLM 的智能文献管理 Agent 系统**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 目录

- [项目概览](#项目概览)
  - [项目简介](#项目简介)
  - [核心特性](#核心特性)
  - [适用场景](#适用场景)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
  - [配置 Azure OpenAI](#配置-azure-openai)
  - [基础使用示例](#基础使用示例)
  - [首次运行检查清单](#首次运行检查清单)
- [系统架构与设计哲学](#系统架构与设计哲学)
- [详细使用指南](#详细使用指南)
  - [CLI 命令详解](#cli-命令详解)
    - [pangu run - 固定任务模式](#pangu-run---固定任务模式)
    - [pangu interactive - 交互模式](#pangu-interactive---交互模式)
  - [高级使用场景](#高级使用场景)
  - [目录结构规范](#目录结构规范)
  - [工作流最佳实践](#工作流最佳实践)
- [代码示例与 API 参考](#代码示例与-api-参考)
- [系统组件详解](#系统组件详解)
- [测试与开发](#测试与开发)
- [常见问题与故障排除](#常见问题与故障排除)


---

## 项目概览

### 项目简介

**Pangu Agent** 是一个现代化的智能文献管理系统，采用大语言模型（LLM）驱动的 Agent 架构，能够通过自然语言交互来自动组织、检索和管理学术文献（PDF 和图像）。

本项目受 **ByteDance TraeAgent** 启发，采用了最先进的 LLM Agent 工作流设计：

- 🤖 **自主决策**：LLM 自主决定执行流程，而非硬编码的规则引擎
- 🧠 **多模态理解**：基于 GPT-4o 等多模态模型，理解 PDF 文本和图像内容
- 🔧 **工具调用**：通过 Function Calling 机制动态调用工具完成任务
- 💭 **序列思考**：具备规划（Planning）和记忆（Memory）能力
- 💬 **自然交互**：支持多轮对话，像与 AI 助手聊天一样管理文献库

与传统的文献管理工具不同，Pangu Agent 不依赖预设的分类规则或关键词匹配，而是通过理解文献内容的语义来进行智能组织和检索。

### 核心特性

#### 🧠 TraeAgent 启发的自主决策架构

- **LLM 控制执行流程**：Agent 自主分析任务、规划步骤、调用工具，无需硬编码工作流
- **Tool-Calling Loop**：基于 OpenAI Function Calling 的工具调用循环
- **Sequential Thinking**：具备推理、规划和反思能力
- **Memory Management**：维护完整的对话历史和上下文

#### 🎨 多模态理解（PDF + 图像）

- 支持 **PDF 文献**：提取文本内容并进行语义理解
- 支持 **学术图像**：理解图表、架构图、实验结果等视觉内容
- **统一嵌入空间**：使用 OpenCLIP 将文本和图像映射到同一向量空间
- **LLM 生成摘要**：自动为每个文件生成简洁的自然语言描述

#### 🔍 语义检索（OpenCLIP + ChromaDB）

- **向量检索**：基于 OpenCLIP ViT-B-32 模型生成 512 维嵌入向量
- **语义相似度**：使用 cosine 相似度进行排序，而非关键词匹配
- **多模态查询**：自然语言查询可以检索 PDF 和图像
- **过滤和排序**：支持按文件类型过滤、Top-K 结果返回

#### 📚 智能文献组织（LLM 决定目录结构）

- **自主决策位置**：Agent 分析文献内容后自主决定存放目录
- **语义化分类**：基于研究主题创建有意义的目录结构（如 `nlp/transformers/`）
- **Inbox 暂存机制**：新文件先暂存到 `.inbox`，由 Agent 决定最终位置
- **元数据富化**：每个文件配备 Sidecar 元数据文件（`.{filename}.meta.json`）

#### 🛠️ 灵活工具系统（5 种核心工具）

| 工具名称 | 功能描述 | 典型用例 |
|---------|---------|---------|
| `ExploreLibrary` | 浏览文献库的树状目录结构 | "查看现有的文献分类" |
| `SearchLibrary` | 基于语义的文献检索 | "找所有关于 Transformer 的论文" |
| `ViewFile` | 查看文件内容和元数据 | "预览这篇论文的摘要" |
| `MoveFile` | 移动文件并更新元数据 | "将论文移动到 nlp/bert/ 目录" |
| `Finish` | 返回最终结果并终止任务 | "完成检索，返回文件列表" |

#### 💬 自然语言交互

- **交互模式**：支持多轮对话式文献管理
- **意图理解**：Agent 自动理解用户需求并执行相应操作
- **上下文保持**：记住对话历史，支持追问和细化需求
- **丰富的工具集成**：可在对话中调用搜索、浏览、添加、移动等所有工具

### 适用场景

- **🎓 学术研究者**：管理大量论文，通过语义检索快速找到相关文献
- **👥 研究团队**：协同维护文献库，自动化分类减少人工整理工作
- **📖 知识工作者**：整理多模态文档（报告、演示文稿、图表等）
- **🏫 课程学习**：组织课程资料，按主题自动归档

**示例工作流**：

**方式一：使用 CLI 命令**

1. 下载一批最新的 arXiv 论文
2. 使用 `pangu run --action add --path papers/` 批量添加
3. Agent 自动分析每篇论文的主题（如 "Self-Attention 机制"）
4. 自动创建目录结构（如 `nlp/attention_mechanisms/`）并移动文件
5. 需要时通过自然语言检索：`pangu run --action search --prompt "注意力机制的改进方法"`

**方式二：使用交互模式**（推荐）

1. 启动交互模式：`pangu interactive`
2. 对话式管理："帮我添加 Downloads 目录下的所有论文"
3. Agent 自动处理："已添加 5 篇论文并自动分类"
4. 继续追问："刚才添加的论文中有哪些是关于 Transformer 的？"
5. Agent 智能响应："其中 3 篇与 Transformer 相关..."
6. 进一步操作："把它们移到 nlp/transformers/ 目录"

---

## 快速开始

### 环境要求

- **Python**：≥ 3.11
- **操作系统**：macOS / Linux / Windows
- **必需服务**：Azure OpenAI API 访问权限（需要支持 GPT-4o 或类似多模态模型）
- **可选硬件**：GPU（用于加速 OpenCLIP 编码，但 CPU 也可运行）

### 安装步骤

#### 方法一：使用 uv（推荐）

[uv](https://github.com/astral-sh/uv) 是一个快速的 Python 包管理器，推荐用于开发环境。

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/pangu-agent.git
cd pangu-agent

# 2. 安装 uv（如果尚未安装）
pip install uv

# 3. 同步依赖并创建虚拟环境
uv sync

# 4. 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows
```

#### 方法二：使用 pip

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/pangu-agent.git
cd pangu-agent

# 2. 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -e .
```

### 配置 Azure OpenAI

Pangu Agent 需要 Azure OpenAI API 来驱动 Agent 的决策和元数据生成。以下是三种配置方式：

#### 方法一：环境变量（推荐）

```bash
# 设置环境变量
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key-here"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"  # 你的部署名称
```

**永久保存**（添加到 `~/.bashrc` 或 `~/.zshrc`）：

```bash
echo 'export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"' >> ~/.bashrc
echo 'export AZURE_OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export AZURE_OPENAI_API_VERSION="2024-02-15-preview"' >> ~/.bashrc
echo 'export AZURE_OPENAI_DEPLOYMENT="gpt-4o"' >> ~/.bashrc
source ~/.bashrc
```

#### 方法二：配置文件

创建 `config.yaml` 文件：

```yaml
deployment_name: "gpt-4o"
azure_endpoint: "https://your-resource.openai.azure.com/"
azure_api_key: "your-api-key-here"
azure_api_version: "2024-02-15-preview"
temperature: 0.3
max_tokens: 2000
```

使用时指定配置文件（在代码中）：

```python
from pangu_agent.llm_client import LLMClient

client = LLMClient.from_config("config.yaml")
```

#### 方法三：代码传参

```python
from pangu_agent.llm_client import LLMClient, LLMConfig

config = LLMConfig(
    deployment_name="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_api_key="your-api-key-here",
    azure_api_version="2024-02-15-preview",
    temperature=0.3,
    max_tokens=2000
)

client = LLMClient(config)
```

### 基础使用示例

#### 示例 1：添加文献

将文献添加到库中，Agent 会自动分析内容并决定存放位置。

```bash
# 添加单个 PDF 文件
pangu run --action add --path ~/Downloads/attention_is_all_you_need.pdf

# 添加整个目录（递归扫描所有 PDF 和图像）
pangu run --action add --path ~/Documents/research_papers/

# 添加时提供上下文提示（帮助 Agent 更好地理解）
pangu run --action add \
  --path paper.pdf \
  --prompt "这是关于 Transformer 架构的开创性论文"
```

**执行流程说明**：

1. **暂存到 Inbox**：文件被复制到 `.inbox/` 暂存区
2. **生成元数据**：
   - 提取 PDF 文本或图像内容
   - LLM 生成 2-3 句的简洁摘要
   - 计算文件哈希值并分配唯一 ID
3. **Agent 分析**：Agent 读取文件内容和元数据
4. **探索库结构**：Agent 使用 `ExploreLibrary` 工具浏览现有目录
5. **决策并移动**：Agent 决定目标位置（如 `nlp/transformers/`）并使用 `MoveFile` 工具移动文件

**输出示例**：

```
✓ Processing: attention_is_all_you_need.pdf
  Staged to inbox: .inbox/a3f5b8c9_attention_is_all_you_need.pdf
  Generated metadata with description
  Agent exploring library structure...

✓ File organized successfully!
  From: .inbox/a3f5b8c9_attention_is_all_you_need.pdf
  To:   nlp/transformers/attention_is_all_you_need.pdf

Summary: Added 1 file(s) in 12.3 seconds
```

#### 示例 2：语义检索

使用自然语言查询来检索相关文献。

```bash
# 自然语言查询
pangu run --action search --prompt "找出所有关于注意力机制的论文"

# 更具体的查询
pangu run --action search --prompt "BERT 模型的预训练方法"

# 多模态查询（可以找到相关图像）
pangu run --action search --prompt "Transformer 架构图"
```

**输出示例**：

```
✓ Search completed in 3 iteration(s)

Found 5 relevant file(s):

PDF Files:
  1. nlp/transformers/attention_is_all_you_need.pdf (similarity: 0.89)
     "Introduces the Transformer architecture based entirely on attention mechanisms..."

  2. nlp/bert/bert_pretraining.pdf (similarity: 0.85)
     "BERT uses masked language modeling and next sentence prediction for pre-training..."

  3. vision/vit/vision_transformer.pdf (similarity: 0.78)
     "Applies pure transformer architecture to image classification tasks..."

Image Files:
  1. diagrams/transformer_architecture.png (similarity: 0.82)
     "Diagram showing the encoder-decoder structure of the Transformer model..."

Observation:
找到 5 篇相关文献，涵盖 Transformer 原始论文、BERT 预训练方法、
视觉 Transformer 以及架构图。建议从第 1、2 篇论文开始阅读。
```

#### 示例 3：交互模式使用

启动交互式对话界面，与 PangGu🍄 助手自然交流。

```bash
# 启动交互模式
pangu interactive

# 或指定文献库路径
pangu interactive --library ~/my_library
```

**交互示例**：

```
╔══════════════════════════════════════════════╗
║    🍄 PangGu Literature Assistant 🍄      ║
╚══════════════════════════════════════════════╝

│ You › 帮我搜索关于 BERT 的论文

│ PangGu🍄 › 我找到了 3 篇关于 BERT 的论文：
1. nlp/language_models/bert/bert_paper.pdf
2. nlp/language_models/bert/roberta_paper.pdf
3. nlp/text_classification/bert_for_classification.pdf

│ You › 第一篇讲什么？

│ PangGu🍄 › BERT（Bidirectional Encoder Representations from
Transformers）使用 Transformer 编码器进行双向预训练。
主要创新是 Masked Language Model (MLM) 和 Next Sentence
Prediction (NSP) 两个预训练任务...

│ You › 添加我桌面上的新论文 ~/Desktop/new_paper.pdf

│ PangGu🍄 › 已分析并添加到 nlp/sentiment_analysis/new_paper.pdf

│ You › exit

Thanks for using PangGu! Goodbye! 👋
```

**执行流程说明**：

- **意图理解**：Agent 自动识别用户需求（搜索、查看、添加等）
- **工具调用**：Agent 自主决定调用哪些工具
- **上下文保持**：记住对话历史，支持连续追问
- **自然响应**：以对话方式返回结果

#### 示例 4：重置文献库

清空整个文献库（谨慎使用）。

```bash
pangu run --action reset --library ./my_library
```

系统会提示确认：

```
WARNING: This will delete all files in the library!
Are you sure you want to reset the library at './my_library'? (yes/no): yes

✓ Library reset successfully
  - Deleted all files and metadata
  - Reinitialized .inbox and .vector_store
```

### 首次运行检查清单

在开始使用前，请确认以下项目：

- [ ] **Azure OpenAI API 配置正确**
  - 测试命令：`python -c "from pangu_agent.llm_client import LLMClient; LLMClient()"`
  - 应该不报错且能成功初始化

- [ ] **PyTorch 和 OpenCLIP 模型已下载**
  - 首次运行会自动下载 ViT-B-32 模型（约 300MB）
  - 确保网络连接稳定

- [ ] **文献库目录有写入权限**
  - 默认路径：`./library`
  - 测试命令：`mkdir -p library && touch library/test.txt && rm library/test.txt`

- [ ] **至少 2GB 可用磁盘空间**
  - 用于存储文件、元数据和 ChromaDB 向量数据库

- [ ] **Python 版本正确**
  - 检查命令：`python --version`（应显示 3.11 或更高版本）

**快速测试**：

```bash
# 下载测试数据（包含几篇 arXiv 论文）
python tests/download_test_data.py

# 运行测试添加功能
pangu run --action add --path tests/data/

# 运行测试检索功能
pangu run --action search --prompt "deep learning"
```

---

## 系统架构与设计哲学

### 设计哲学

#### 核心理念：Agent 自主决策，而非规则引擎

Pangu Agent 的核心设计理念来自于 **ByteDance TraeAgent**，强调 **LLM 自主控制执行流程**，而非传统的硬编码规则引擎。

**传统规则引擎 vs Pangu Agent**：

| 维度 | 传统规则引擎 | Pangu Agent（TraeAgent 启发） |
|------|------------|----------------------------|
| **执行流程** | if-else 硬编码逻辑 | LLM 动态决策 |
| **灵活性** | 固定流程，难以适应新场景 | 自适应，理解上下文后规划 |
| **扩展性** | 每个新功能需修改代码 | 添加工具即可，LLM 自动学会使用 |
| **用户交互** | 命令式（明确指定步骤） | 声明式（描述目标即可） |
| **决策依据** | 预设规则（如关键词匹配） | 语义理解（内容深度分析） |

**示例对比**：

```
【传统方式】添加文献
1. 用户必须手动指定目标目录
2. 系统按关键词或文件名规则匹配分类
3. 分类规则需要预先定义和维护

【Pangu Agent 方式】添加文献
1. 用户：「添加这篇关于 BERT 的论文」
2. Agent：分析论文内容 → 理解是预训练语言模型
3. Agent：探索现有目录 → 发现 nlp/language_models/
4. Agent：决策 → 创建 nlp/language_models/bert/
5. Agent：移动文件并生成描述性元数据
```

#### TraeAgent 启发的关键特性

1. **Tool-Calling Loop（工具调用循环）**
   - LLM 在每次迭代中决定：调用工具 or 返回结果
   - 工具执行结果反馈给 LLM，形成闭环
   - 直到任务完成或达到最大迭代次数

2. **Sequential Thinking（序列思考）**
   - LLM 具备推理能力，可以规划多步操作
   - 示例：「先搜索现有文献 → 查看内容 → 决定分类 → 移动文件」

3. **Planning（规划能力）**
   - Agent 可以分解复杂任务为子任务
   - 动态调整计划应对意外情况

4. **Memory（记忆机制）**
   - 维护完整对话历史（system/user/assistant/tool）
   - 上下文理解：记住之前的操作和结果

#### 元数据富化策略

Pangu Agent 不仅存储文件，还为每个文件生成 **LLM 摘要**，这是一个关键的设计决策：

**为什么需要 LLM 生成的元数据？**

- **快速预览**：2-3 句摘要让用户无需打开文件即可了解内容
- **语义检索增强**：摘要可以被向量化，提升检索准确度
- **Agent 决策依据**：Agent 可以通过摘要快速理解文件，而非读取完整内容
- **多模态统一**：无论 PDF 还是图像，都有自然语言描述

**元数据生成流程**：

```
PDF 文件 → 提取文本（pypdf）→ 截取前 10000 字符 → LLM 总结
图像文件 → 编码为 base64 → 多模态 LLM（GPT-4o）理解 → 生成描述
```

---

### 系统架构

#### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        应用层（Application Layer）              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐              ┌──────────────────┐        │
│  │   CLI 模式    │              │  交互模式        │        │
│  │  pangu run   │              │ pangu interactive│        │
│  │  - add       │              │ - 多轮对话       │        │
│  │  - search    │              │ - 上下文保持     │        │
│  │  - reset     │              │ - 意图理解       │        │
│  └──────────────┘              └──────────────────┘        │
│         │                              │                   │
└─────────┼──────────────────────────────┼───────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        逻辑层（Logic Layer）                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────┐        │
│  │            Agent 控制器（Agent）                 │        │
│  │  • Tool-Calling Loop（工具调用循环）            │        │
│  │  • Memory（对话历史管理）                       │        │
│  │  • Stop Condition（停止条件判定）               │        │
│  └────────────────────────────────────────────────┘        │
│                         │                                   │
│                         │ 调用                              │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────┐        │
│  │         服务层（Services）                       │        │
│  │  • AddLiteratureService（文献添加服务）         │        │
│  │  • SearchFilesService（检索服务）               │        │
│  └────────────────────────────────────────────────┘        │
│                         │                                   │
│                         │ 使用                              │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────┐        │
│  │          工具层（Tools）                         │        │
│  │  • ExploreLibrary  • SearchLibrary              │        │
│  │  • ViewFile        • MoveFile                   │        │
│  │  • Finish                                       │        │
│  └────────────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ 操作
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                        数据层（Data Layer）                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────┐        │
│  │        LibraryManager（文献库管理器）            │        │
│  │  • 文件系统操作                                  │        │
│  │  • 元数据管理（Sidecar JSON）                   │        │
│  │  • 向量数据库交互                                │        │
│  └────────────────────────────────────────────────┘        │
│                         │                                   │
│           ┌─────────────┼─────────────┐                     │
│           ▼             ▼             ▼                     │
│  ┌──────────────┐ ┌──────────┐ ┌─────────────┐            │
│  │  文件系统     │ │ 元数据层  │ │  向量数据库  │            │
│  │              │ │          │ │             │            │
│  │ library/     │ │ .meta    │ │ ChromaDB    │            │
│  │ ├─ .inbox/   │ │ .json    │ │ (OpenCLIP   │            │
│  │ ├─ nlp/      │ │ files    │ │  嵌入)      │            │
│  │ └─ vision/   │ │          │ │             │            │
│  └──────────────┘ └──────────┘ └─────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 三层架构说明

##### 1. 应用层（Application Layer）

**职责**：提供用户交互接口

- **CLI 模式**（已实现）：
  - `pangu run --action add`：批量添加文献
  - `pangu run --action search`：语义检索
  - `pangu run --action reset`：重置文献库

- **交互模式**（已实现）：
  - 多轮对话式文献管理
  - 自然语言意图理解
  - 上下文保持和追问
  - 支持所有核心功能的集成调用

##### 2. 逻辑层（Logic Layer）

**核心组件**：

**① Agent 控制器**

负责执行 Tool-Calling Loop，是整个系统的"大脑"。

关键代码逻辑（伪代码简化版）：

```python
class Agent:
    def run(self, stop_condition=None):
        for iteration in range(self._max_iterations):
            # 1. LLM 根据记忆和工具模式生成响应
            response = self._llm_client.completion(
                self._memory, tools=tool_schemas
            )

            # 2. 如果 LLM 返回工具调用
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    # 3. 执行工具
                    result = self._tool_executor.execute(tool_call)
                    # 4. 将结果加入记忆
                    self._memory.add("tool", result)

                    # 5. 检查停止条件
                    if stop_condition and stop_condition(tool_call, result):
                        return result

            # 6. 如果 LLM 返回文本（主动结束）
            elif response.content:
                return {"final_message": response.content}

        # 7. 达到最大迭代次数
        return {"stop_reason": "max_iterations"}
```

**② 服务层（Services）**

封装特定任务的高级工作流：

- **AddLiteratureService**：
  ```
  扫描文件 → 暂存到 inbox → 为每个文件创建 Agent
  → Agent 探索库 → Agent 决定位置 → Agent 移动文件
  ```

- **SearchFilesService**：
  ```
  创建 Agent → Agent 语义检索 → Agent 查看候选文件
  → Agent 编译结果 → Agent 调用 Finish 工具返回
  ```

**③ 工具层（Tools）**

提供 Agent 可调用的原子操作，每个工具都符合 OpenAI Function Calling 规范。

##### 3. 数据层（Data Layer）

**LibraryManager** 是数据层的核心，管理三种数据：

**① 文件层**：树状目录结构

```
library/
├── .inbox/                    # 暂存区（对 ExploreLibrary 不可见）
│   └── {uuid}_{filename}.pdf
├── .vector_store/             # ChromaDB 持久化数据（自动管理）
├── nlp/
│   ├── transformers/
│   │   ├── attention_is_all_you_need.pdf
│   │   └── .attention_is_all_you_need.pdf.meta.json  # Sidecar 元数据
│   └── language_models/
│       └── bert/
└── computer_vision/
    └── object_detection/
```

**② 元数据层**：Sidecar JSON 文件

每个文件都有一个配套的 `.{filename}.meta.json`：

```json
{
  "id": "a3f5b8c9-1234-5678-9abc-def012345678",
  "path": "nlp/transformers/attention_is_all_you_need.pdf",
  "hash": "sha256:a3f5b8c9...",
  "added_at": "2024-01-05T10:30:00Z",
  "updated_at": "2024-01-05T10:30:00Z",
  "description": "This paper introduces the Transformer, a novel architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
}
```

**③ 向量层**：ChromaDB 向量存储

- **嵌入模型**：OpenCLIP ViT-B-32（512 维）
- **距离度量**：Cosine 相似度
- **索引**：HNSW（分层小世界导航图）
- **持久化**：自动保存到 `.vector_store/`

---

### 核心组件概览

详细的组件说明请参见 [系统组件详解](#系统组件详解) 章节。这里仅列出关键要点：

#### 1. Agent 控制器

- **Tool-Calling Loop**：迭代执行 LLM → 工具 → LLM 循环
- **Memory 管理**：维护完整对话历史
- **Stop Condition**：灵活的任务结束判定

#### 2. Library Manager

- **三层数据模型**：文件 + 元数据 + 向量
- **Inbox 暂存机制**：新文件先进暂存区，由 Agent 决定最终位置
- **Sidecar 元数据**：`.{filename}.meta.json` 模式

#### 3. 嵌入式检索系统

- **OpenCLIP ViT-B-32**：多模态嵌入模型（512 维）
- **ChromaDB**：向量数据库，Cosine 相似度检索
- **HNSW 索引**：高效近似最近邻搜索

#### 4. 五大核心工具

| 工具名称 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| `ExploreLibrary` | 浏览目录 | path, depth | 树状结构文本 |
| `SearchLibrary` | 语义检索 | query, top_k | 排序文件列表 |
| `ViewFile` | 查看文件 | file_path, info_type | 内容+元数据 |
| `MoveFile` | 移动文件 | source, dest | 成功/失败消息 |
| `Finish` | 完成任务 | result, status | 最终结果 |

---

## 详细使用指南

### CLI 命令详解

#### `pangu run` - 固定任务模式

**基本语法**：

```bash
pangu run --action {search|add|reset} [OPTIONS]
```

**通用参数**：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--library` | PATH | 否 | `./library` | 文献库根目录 |
| `--action` | CHOICE | 是 | - | 任务类型：`add`/`search`/`reset` |

**action=add 专用参数**：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--path` | PATH | 是 | - | 要添加的文件或目录路径 |
| `--prompt` | TEXT | 否 | - | 可选的上下文提示，帮助 Agent 更好地理解 |

**action=search 专用参数**：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--prompt` | TEXT | 是 | - | 自然语言搜索查询 |

**命令示例**：

```bash
# 使用默认文献库（./library）添加文件
pangu run --action add --path ~/Downloads/paper.pdf

# 指定自定义文献库
pangu run --action add \
  --library ~/my_research/literature \
  --path paper.pdf

# 添加时提供上下文
pangu run --action add \
  --path paper.pdf \
  --prompt "这是我导师推荐的关于强化学习的综述论文"

# 搜索
pangu run --action search --prompt "深度学习优化算法"

# 重置（需确认）
pangu run --action reset --library ./library
```

#### `pangu interactive` - 交互模式

**基本语法**：

```bash
pangu interactive [OPTIONS]
```

**参数**：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--library` | PATH | 否 | `./library` | 文献库根目录 |
| `--verbose` / `-v` | FLAG | 否 | False | 启用详细日志输出 |

**功能特性**：

- 🤖 **多轮对话**：支持连续对话，Agent 记住上下文
- 🧠 **意图理解**：自动理解并执行复杂的文献管理任务
- 🔧 **工具调用**：Agent 根据需求自主调用搜索、添加、移动等工具
- 💬 **自然交互**：像与 AI 助手聊天一样管理文献库
- 🔄 **会话管理**：支持 `reset` 命令清除对话历史

**启动交互模式**：

```bash
# 使用默认文献库
pangu interactive

# 指定自定义文献库
pangu interactive --library ~/my_research/literature

# 启用详细日志（查看 Agent 工具调用过程）
pangu interactive --verbose
```

**交互示例**：

```
$ pangu interactive

╔══════════════════════════════════════════════╗
║    🍄 PangGu Literature Assistant 🍄      ║
╚══════════════════════════════════════════════╝

┌─ Ready to assist! ───────────────────────────┐
│ Library: ./library                           │
│                                              │
│ I can help you with:                         │
│   • Search for papers and files              │
│   • Add and organize new literature          │
│   • Explore the library structure            │
│   • View and analyze file contents           │
│   • Move and reorganize files                │
│                                              │
│ Commands: exit to quit, reset to clear       │
│ history                                      │
└──────────────────────────────────────────────┘

│ You › 帮我找所有关于 Transformer 的论文

│ PangGu🍄 › 我找到了 5 篇关于 Transformer 的论文：

1. nlp/transformers/attention_is_all_you_need.pdf
   - 这是原始 Transformer 论文，介绍了基于注意力机制的序列到序列架构

2. nlp/language_models/bert/bert_paper.pdf
   - BERT 使用 Transformer 编码器进行预训练

3. vision/vit/vision_transformer.pdf
   - 将 Transformer 架构应用于图像分类

4. nlp/transformers/transformer_xl.pdf
   - 改进的 Transformer，支持更长的上下文

5. multimodal/vision_language/clip_paper.pdf
   - CLIP 使用 Transformer 进行多模态学习

这些论文涵盖了 Transformer 在 NLP、CV 和多模态领域的应用。

│ You › 其中哪些是原始论文？

│ PangGu🍄 › 根据内容分析，第 1 篇 "Attention Is All You Need"
(nlp/transformers/attention_is_all_you_need.pdf) 是 2017 年发表
的原始 Transformer 论文，由 Vaswani 等人提出。这是整个
Transformer 架构的开创性工作。

│ You › 帮我把它移动到 classics/ 目录

│ PangGu🍄 › 已将文件移动到 classics/attention_is_all_you_need.pdf。
这篇开创性论文现在归类在经典论文目录中。

│ You › 添加我桌面上的新论文 ~/Desktop/new_paper.pdf

│ PangGu🍄 › 我已经分析了这篇论文的内容，它是关于自监督学习的
综述。我将其整理到了 machine_learning/self_supervised/ 目录下。

文件已添加：machine_learning/self_supervised/new_paper.pdf

│ You › exit

Thanks for using PangGu! Goodbye! 👋
```

**内置命令**：

| 命令 | 功能 |
|------|------|
| `exit` / `quit` / `q` | 退出交互模式 |
| `reset` | 清除对话历史（保留系统提示词） |

**使用技巧**：

1. **自然语言查询**：无需记住命令格式，直接用自然语言表达需求
   ```
   "找出所有关于深度学习的论文"
   "帮我整理桌面上的 PDF 文件"
   "这个目录下有什么内容？"
   ```

2. **追问和细化**：Agent 记住对话历史，支持连续追问
   ```
   User: "搜索注意力机制相关的论文"
   Agent: [返回结果]
   User: "其中哪些是关于自注意力的？"  ← 基于上一轮结果
   ```

3. **复杂任务分解**：Agent 会自动规划并执行多步骤任务
   ```
   User: "找到所有 CV 相关的论文，然后整理到 computer_vision/ 目录"
   Agent: [自动调用搜索工具 → 识别文件 → 调用移动工具]
   ```

4. **查看 Agent 工作过程**：使用 `--verbose` 查看详细日志
   ```bash
   pangu interactive --verbose
   # 显示每次工具调用、迭代次数等调试信息
   ```

---

### 高级使用场景

#### 场景 0：交互模式的实际应用

交互模式是使用 Pangu Agent 的**推荐方式**，它提供了更自然、更高效的文献管理体验。

**场景 0.1：初次整理文献库**

```
$ pangu interactive

│ You › 我的 Downloads 文件夹里有很多 PDF，帮我整理一下

│ PangGu🍄 › 我来帮你扫描 Downloads 文件夹。请问具体路径是什么？

│ You › ~/Downloads

│ PangGu🍄 › 我找到了 15 个 PDF 文件。我现在开始分析每个文件的内容
并为它们分类...

[Agent 自动调用 AddLiteratureTool 批量处理]

│ PangGu🍄 › 整理完成！我已经将 15 篇论文分类到以下目录：
- nlp/transformers/ (5 篇)
- computer_vision/object_detection/ (4 篇)
- reinforcement_learning/policy_gradient/ (3 篇)
- machine_learning/optimization/ (2 篇)
- multimodal/vision_language/ (1 篇)
```

**场景 0.2：探索和重组文献库**

```
│ You › 我的文献库现在是什么样子的？

│ PangGu🍄 › 让我为你探索一下文献库的结构...

[Agent 调用 ExploreLibraryTool]

│ PangGu🍄 › 你的文献库包含以下主要分类：

1. nlp/ (12 篇论文)
   - transformers/ (5 篇)
   - language_models/bert/ (4 篇)
   - language_models/gpt/ (3 篇)

2. computer_vision/ (8 篇论文)
   - object_detection/ (4 篇)
   - segmentation/ (2 篇)
   - image_generation/ (2 篇)

3. reinforcement_learning/ (5 篇论文)
...

│ You › nlp 下面的内容有点乱，帮我重新整理一下

│ PangGu🍄 › 好的，我来分析 nlp 目录下的论文，看看如何更好地组织...

[Agent 调用 ViewFile 查看每篇论文，然后调用 MoveFile 重组]

│ PangGu🍄 › 重组完成！我将 nlp/ 目录重新整理为：
- nlp/architectures/transformers/ (原 transformers/ 内容)
- nlp/pretraining/masked_lm/ (BERT 相关论文)
- nlp/pretraining/autoregressive/ (GPT 相关论文)

这样分类更清晰，按照模型的训练方式划分。
```

**场景 0.3：深度研究特定主题**

```
│ You › 我想研究 Transformer 在计算机视觉中的应用

│ PangGu🍄 › 让我为你搜索相关论文...

[Agent 调用 SearchLibraryTool]

│ PangGu🍄 › 找到 4 篇相关论文：

1. vision/vit/vision_transformer.pdf (相似度: 0.91)
2. vision/detection/detr.pdf (相似度: 0.87)
3. multimodal/vision_language/clip_paper.pdf (相似度: 0.84)
4. vision/segmentation/segformer.pdf (相似度: 0.79)

│ You › 第一篇讲的是什么？

│ PangGu🍄 › 让我查看一下这篇论文...

[Agent 调用 ViewFile]

│ PangGu🍄 › 《Vision Transformer》这篇论文提出将标准的 Transformer
直接应用于图像分类任务。主要创新点是：

1. 将图像分割成固定大小的 patch (16×16)
2. 将每个 patch 线性投影为 embedding
3. 使用标准 Transformer encoder 处理 patch 序列
4. 在大规模数据集上预训练后，在下游任务上 fine-tune

在 ImageNet 上达到了 88.55% 的准确率，性能媲美 SOTA 的卷积网络。

│ You › 这几篇论文我想单独放一个目录，叫 vision_transformers

│ PangGu🍄 › 好的，我来帮你创建这个目录并移动这些论文...

[Agent 调用 MoveFile 工具]

│ PangGu🍄 › 完成！已创建 vision/vision_transformers/ 目录，
并将 4 篇论文移动到这个目录下。
```

#### 场景 1：批量导入研究项目文献

**需求**：将一个课题的所有文献（来自不同子目录）批量导入并自动分类。

**方案**：

```bash
#!/bin/bash
# batch_add_literature.sh

PROJECT_DIR="~/research/multimodal_learning"
LIBRARY_PATH="~/my_library"

# 1. 遍历所有子目录
for subdir in "$PROJECT_DIR"/*; do
    if [ -d "$subdir" ]; then
        echo "Processing directory: $subdir"

        # 2. 为每个子目录添加描述性提示
        dirname=$(basename "$subdir")
        prompt="Papers related to $dirname"

        # 3. 批量添加
        pangu run --action add \
          --library "$LIBRARY_PATH" \
          --path "$subdir" \
          --prompt "$prompt"
    fi
done

echo "✓ Batch import completed!"
```

**执行**：

```bash
chmod +x batch_add_literature.sh
./batch_add_literature.sh
```

#### 场景 2：精准检索策略

**需求**：在大型文献库中找到特定主题的相关论文。

**策略：从宽泛到精确的漏斗式检索**

```bash
# Step 1: 宽泛检索，了解文献库概况
pangu run --action search --prompt "机器学习"

# 输出：找到 50 篇论文，涵盖监督学习、强化学习、深度学习等

# Step 2: 细化查询，缩小范围
pangu run --action search --prompt "深度强化学习"

# 输出：找到 15 篇论文，包括 DQN、A3C、PPO 等

# Step 3: 精确查询，锁定目标
pangu run --action search --prompt "PPO 算法在机器人控制中的应用"

# 输出：找到 3 篇高度相关的论文
```

**多模态查询**：

```bash
# 查找图表和论文
pangu run --action search --prompt "神经网络架构可视化图表"

# 可以同时找到:
# - PDF 中的架构说明
# - 独立的架构图图片
```

#### 场景 3：多项目文献库隔离

**需求**：为不同研究项目维护独立的文献库。

**方案**：

```bash
# 项目 A：自然语言处理
pangu run --action add \
  --library ~/projects/nlp_project/literature \
  --path ~/Downloads/bert_paper.pdf

# 项目 B：计算机视觉
pangu run --action add \
  --library ~/projects/cv_project/literature \
  --path ~/Downloads/resnet_paper.pdf

# 项目 C：多模态学习（共享库）
pangu run --action add \
  --library ~/shared_literature \
  --path ~/Downloads/clip_paper.pdf
```

**使用别名简化操作**（添加到 `~/.bashrc`）：

```bash
alias pangu-nlp="pangu run --library ~/projects/nlp_project/literature"
alias pangu-cv="pangu run --library ~/projects/cv_project/literature"
alias pangu-mm="pangu run --library ~/shared_literature"

# 使用
pangu-nlp --action add --path paper.pdf
pangu-cv --action search --prompt "object detection"
```

#### 场景 4：与现有工具集成

**需求**：将 Pangu Agent 集成到论文下载工作流中。

**示例：从 arXiv 下载并自动添加**

```bash
#!/bin/bash
# arxiv_auto_add.sh

ARXIV_ID=$1
LIBRARY_PATH="~/my_library"

# 1. 下载 arXiv 论文
echo "Downloading arXiv:$ARXIV_ID..."
wget -O "/tmp/${ARXIV_ID}.pdf" "https://arxiv.org/pdf/${ARXIV_ID}.pdf"

# 2. 获取论文元信息（使用 arXiv API）
TITLE=$(curl -s "http://export.arxiv.org/api/query?id_list=$ARXIV_ID" | \
        grep -oP '(?<=<title>).*?(?=</title>)' | tail -1)

# 3. 使用 Pangu Agent 添加
pangu run --action add \
  --library "$LIBRARY_PATH" \
  --path "/tmp/${ARXIV_ID}.pdf" \
  --prompt "arXiv paper: $TITLE"

# 4. 清理临时文件
rm "/tmp/${ARXIV_ID}.pdf"

echo "✓ Paper $ARXIV_ID added to library!"
```

**使用**：

```bash
# 添加单篇论文
./arxiv_auto_add.sh 1706.03762  # Attention Is All You Need

# 批量添加
cat arxiv_ids.txt | while read id; do
    ./arxiv_auto_add.sh "$id"
done
```

---

### 目录结构规范

#### 推荐的文献库结构

```
library/
├── .inbox/                          # 暂存区（系统自动管理）
│   ├── uuid1_paper1.pdf
│   ├── .uuid1_paper1.pdf.meta.json
│   └── ...
│
├── .vector_store/                   # ChromaDB 数据（系统自动管理）
│   └── chroma.sqlite3
│
├── computer_vision/                 # 计算机视觉
│   ├── object_detection/
│   │   ├── yolo_paper.pdf
│   │   ├── .yolo_paper.pdf.meta.json
│   │   ├── faster_rcnn.pdf
│   │   └── .faster_rcnn.pdf.meta.json
│   ├── segmentation/
│   │   └── unet_paper.pdf
│   └── image_generation/
│       ├── gan_paper.pdf
│       └── diffusion_models.pdf
│
├── nlp/                             # 自然语言处理
│   ├── transformers/
│   │   ├── attention_is_all_you_need.pdf
│   │   └── transformer_xl.pdf
│   ├── language_models/
│   │   ├── bert/
│   │   │   ├── bert_paper.pdf
│   │   │   └── roberta_paper.pdf
│   │   └── gpt/
│   │       ├── gpt2_paper.pdf
│   │       └── gpt3_paper.pdf
│   └── text_generation/
│
├── multimodal/                      # 多模态学习
│   ├── vision_language/
│   │   ├── clip_paper.pdf
│   │   └── flamingo_paper.pdf
│   └── video_understanding/
│
├── reinforcement_learning/          # 强化学习
│   ├── policy_gradient/
│   ├── q_learning/
│   └── model_based/
│
├── diagrams/                        # 图表和可视化
│   ├── transformer_architecture.png
│   ├── resnet_structure.jpg
│   └── attention_visualization.png
│
└── classics/                        # 经典论文
    ├── alexnet_paper.pdf
    ├── resnet_paper.pdf
    └── attention_is_all_you_need.pdf
```

#### 元数据文件示例

**完整的 `.meta.json` 文件**（`nlp/transformers/.attention_is_all_you_need.pdf.meta.json`）：

```json
{
  "id": "a3f5b8c9-1234-5678-9abc-def012345678",
  "path": "nlp/transformers/attention_is_all_you_need.pdf",
  "hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "added_at": "2024-01-05T10:30:00.000Z",
  "updated_at": "2024-01-05T10:30:00.000Z",
  "description": "This paper introduces the Transformer, a novel neural network architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions. The model achieves state-of-the-art results on machine translation tasks while being more parallelizable and requiring significantly less time to train."
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID | 文件的唯一标识符 |
| `path` | String | 文件在库中的相对路径 |
| `hash` | String | SHA-256 哈希值，用于去重和完整性检查 |
| `added_at` | ISO 8601 | 文件首次添加时间 |
| `updated_at` | ISO 8601 | 文件最后移动或修改时间 |
| `description` | String | LLM 生成的 2-3 句摘要 |

**未来可能扩展的字段**（🚧 开发中）：

```json
{
  ...
  "title": "Attention Is All You Need",
  "authors": ["Vaswani, Ashish", "Shazeer, Noam", ...],
  "year": "2017",
  "venue": "NIPS",
  "keywords": ["transformer", "attention", "neural machine translation"],
  "citation_count": 50000,
  "doi": "10.5555/3295222.3295349"
}
```

---

### 工作流最佳实践

#### 文献添加流程

**推荐流程**：

```
1. 下载/收集文献
   ↓
2. 整理到临时目录（如 ~/Downloads/new_papers/）
   ↓
3. 添加前检查（避免重复）
   pangu run --action search --prompt "论文标题或关键内容"
   ↓
4. 批量添加
   pangu run --action add --path ~/Downloads/new_papers/ \
     --prompt "简短描述这批论文的共同主题"
   ↓
5. 验证结果
   - 检查文件是否正确分类
   - 查看元数据是否准确
   ↓
6. 清理临时目录
```

**避免重复的技巧**：

```bash
# 在添加前，先搜索标题
pangu run --action search --prompt "Attention Is All You Need"

# 如果找到了，可能已经存在
# 如果没找到，可以放心添加
```

#### 检索优化技巧

**1. 使用具体的描述性查询**

```bash
# ❌ 不推荐：过于宽泛
pangu run --action search --prompt "deep learning"

# ✅ 推荐：具体且描述性
pangu run --action search --prompt "deep learning for image classification using convolutional networks"
```

**2. 利用多模态能力**

```bash
# 查找图表
pangu run --action search --prompt "neural network architecture diagram"

# 查找包含特定实验结果的论文
pangu run --action search --prompt "ImageNet top-1 accuracy comparison table"
```

**3. 迭代细化查询**

```bash
# 第一次：宽泛查询，了解领域
pangu run --action search --prompt "推荐系统"

# 第二次：根据结果细化
pangu run --action search --prompt "基于协同过滤的推荐系统"

# 第三次：锁定具体技术
pangu run --action search --prompt "矩阵分解在协同过滤中的应用"
```

**4. 组合关键词和上下文**

```bash
# 包含多个关键概念
pangu run --action search --prompt "Transformer 在计算机视觉中的应用，特别是目标检测任务"
```

#### 元数据维护建议

**1. 定期验证元数据质量**

```python
# 简单的验证脚本（Python）
from pathlib import Path
import json

library_path = Path("./library")

for meta_file in library_path.rglob("*.meta.json"):
    with open(meta_file) as f:
        metadata = json.load(f)

    # 检查必需字段
    required_fields = ["id", "path", "hash", "added_at", "description"]
    missing = [f for f in required_fields if f not in metadata]

    if missing:
        print(f"⚠️  {meta_file}: Missing fields {missing}")

    # 检查描述长度（应该是 2-3 句话）
    desc = metadata.get("description", "")
    if len(desc) < 50:
        print(f"⚠️  {meta_file}: Description too short")
```

**2. 手动修正元数据**

```bash
# 1. 找到元数据文件
library/nlp/transformers/.attention_is_all_you_need.pdf.meta.json

# 2. 编辑（使用任何文本编辑器）
vi library/nlp/transformers/.attention_is_all_you_need.pdf.meta.json

# 3. 修改 description 字段
{
  ...
  "description": "更准确的描述..."
}

# 注意：修改 path 字段不会移动文件，仅修改元数据
# 如需移动文件，应使用 MoveFile 工具或手动移动+更新元数据
```

**3. 定期清理和重建索引**（未来功能 🚧）

```bash
# 计划中的维护命令
pangu run --action rebuild-index --library ./library
pangu run --action verify-integrity --library ./library
pangu run --action deduplicate --library ./library
```

---

## 代码示例与 API 参考

本章节提供可直接运行的 Python 代码示例，以及核心 API 的详细参考。

### Python API 使用

#### 示例 1：编程式添加文献

```python
from pathlib import Path
from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client import LLMClient
from pangu_agent.services import AddLiteratureService
from pangu_agent.tools import (
    ExploreLibraryTool,
    ViewFileTool,
    MoveFileTool,
    SearchLibraryTool,
)

# 1. 初始化组件
library_root = Path("./my_library")
manager = LibraryManager(library_root)
llm_client = LLMClient()  # 自动从环境变量读取配置

# 2. 设置工具
tools = [
    ExploreLibraryTool(manager),
    ViewFileTool(manager),
    MoveFileTool(manager),
    SearchLibraryTool(manager),
]

# 3. 创建服务
service = AddLiteratureService(manager, llm_client, tools)

# 4. 添加单个文件
result = service.add_file(
    "~/Downloads/attention_paper.pdf",
    user_prompt="This is the original Transformer paper"
)

if result["success"]:
    print(f"✓ {result['source']} → {result['destination']}")
else:
    print(f"✗ Error: {result['error']}")

# 5. 批量添加目录
results = service.add_path(
    "~/Documents/papers/",
    user_prompt="Deep learning papers from my course"
)

# 处理结果
successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

print(f"\n✓ Successfully added {len(successful)} file(s)")
print(f"✗ Failed to add {len(failed)} file(s)")

for r in failed:
    print(f"  - {r['source']}: {r['error']}")
```

#### 示例 2：编程式检索

```python
from pangu_agent.services import SearchFilesService
from pangu_agent.tools import FinishTool
import json

# 1. 初始化（复用上面的 manager 和 llm_client）
tools = [
    SearchLibraryTool(manager),
    ViewFileTool(manager),
    ExploreLibraryTool(manager),
    FinishTool(),
]

service = SearchFilesService(manager, llm_client, tools)

# 2. 执行检索
result = service.search("找出所有关于 GAN 生成对抗网络的论文")

# 3. 处理结果
if result["success"]:
    print(f"Found {len(result['files'])} file(s):\n")
    for file_path in result['files']:
        print(f"  - {file_path}")

    print(f"\nObservation: {result['observation']}")
else:
    print(f"Search failed: {result.get('error', 'Unknown error')}")
```

#### 示例 3：直接使用 Agent

```python
from pangu_agent.agent import Agent
from pangu_agent.prompts import FILE_SEARCHER_SYSTEM_PROMPT

# 1. 创建自定义 Agent
agent = Agent(
    llm_client=llm_client,
    tools=tools,
    max_iterations=10
)

# 2. 设置提示词
agent.add_system_prompt(FILE_SEARCHER_SYSTEM_PROMPT)
agent.add_user_message("帮我找关于强化学习中策略梯度方法的综述论文")

# 3. 运行 Agent
result = agent.run()

# 4. 检查结果
print(f"Success: {result['success']}")
print(f"Iterations: {result['iterations']}")
print(f"Stop reason: {result['stop_reason']}")

if result.get('result'):
    print(f"Result: {result['result']}")
```

#### 示例 4：使用交互服务

```python
from pangu_agent.services import InteractiveService
from pangu_agent.tools import AddLiteratureTool

# 1. 初始化（复用上面的 manager 和 llm_client）

# 创建 AddLiteratureService（用于 AddLiteratureTool）
add_lit_service = AddLiteratureService(
    manager=manager,
    llm_client=llm_client,
    tools=[
        ExploreLibraryTool(manager),
        ViewFileTool(manager),
        MoveFileTool(manager),
        SearchLibraryTool(manager),
    ]
)

# 创建交互服务所需的所有工具
tools = [
    SearchLibraryTool(manager),
    ExploreLibraryTool(manager),
    ViewFileTool(manager),
    AddLiteratureTool(add_lit_service),
    MoveFileTool(manager),
]

# 创建交互服务
service = InteractiveService(
    manager=manager,
    llm_client=llm_client,
    tools=tools,
    max_iterations=15,
)

# 2. 进行多轮对话
print("=== Conversation 1 ===")
result = service.chat("帮我搜索关于 Transformer 的论文")
print(f"Response: {result['response']}")
print(f"Iterations: {result['iterations']}\n")

print("=== Conversation 2 (with context) ===")
result = service.chat("第一篇讲的是什么？")  # 基于上一轮对话
print(f"Response: {result['response']}")
print(f"Iterations: {result['iterations']}\n")

print("=== Conversation 3 ===")
result = service.chat("添加 ~/Desktop/new_paper.pdf")
print(f"Response: {result['response']}")
print(f"Iterations: {result['iterations']}\n")

# 3. 重置对话历史
service.reset()
print("Conversation history cleared!\n")

# 4. 开始新对话（没有之前的上下文）
result = service.chat("探索文献库的结构")
print(f"Response: {result['response']}")
```

**输出示例**：

```
=== Conversation 1 ===
Response: 我找到了 5 篇关于 Transformer 的论文：...
Iterations: 3

=== Conversation 2 (with context) ===
Response: 第一篇是 "Attention Is All You Need"，这是 2017 年提出的...
Iterations: 2

=== Conversation 3 ===
Response: 已分析并添加到 machine_learning/optimization/new_paper.pdf
Iterations: 5

Conversation history cleared!

Response: 你的文献库包含以下主要分类：...
```

#### 示例 5：自定义工具开发

```python
from pangu_agent.tools.base import Tool, ToolParameter, ToolCallArguments

class ExportToBibTexTool(Tool):
    """导出文件元数据为 BibTeX 格式"""

    def __init__(self, manager: LibraryManager):
        super().__init__(
            name="export_bibtex",
            description="Export file metadata to BibTeX citation format",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to export",
                    required=True
                ),
                ToolParameter(
                    name="citation_key",
                    type="string",
                    description="BibTeX citation key (optional, defaults to file ID)",
                    required=False
                )
            ]
        )
        self._manager = manager

    def run(self, arguments: ToolCallArguments) -> str:
        file_path = arguments["file_path"]
        citation_key = arguments.get("citation_key")

        # 读取元数据
        file_info = self._manager.read_file(
            file_path,
            include_content=False,
            include_meta=True
        )

        metadata = file_info["meta_data"]

        # 使用 ID 作为默认 citation key
        if not citation_key:
            citation_key = metadata.get("id", "unknown")[:8]

        # 生成 BibTeX
        bibtex = f"""@article{{{citation_key},
    title = {{{metadata.get('title', 'Unknown')}}},
    author = {{{metadata.get('authors', 'Unknown')}}},
    year = {{{metadata.get('year', 'Unknown')}}},
    note = {{{metadata.get('description', '')}}}
}}"""

        return f"BibTeX entry:\n\n{bibtex}"

# 使用自定义工具
tools = [
    ExploreLibraryTool(manager),
    SearchLibraryTool(manager),
    ExportToBibTexTool(manager),  # 新工具
]

agent = Agent(llm_client, tools)
agent.add_system_prompt("You can export papers to BibTeX format.")
agent.add_user_message("Export nlp/transformers/attention.pdf to BibTeX")
result = agent.run()
```

---

### 核心 API 参考

#### LibraryManager

**初始化**：

```python
from pangu_agent.library.manager import LibraryManager

manager = LibraryManager(
    root_path: str | Path,           # 文献库根目录
    inbox_name: str = ".inbox",       # inbox 子目录名（默认 .inbox）
    llm_client: Optional[LLMClient] = None  # LLM 客户端（用于生成描述）
)
```

**关键方法**：

```python
# 1. 暂存文件到 inbox
staged_path = manager.stage_copy(source_path: str) -> Path
# 返回：inbox 中的新路径

# 2. 移动文件
new_path = manager.move_file(
    source_path: str,     # 源路径（可以在 inbox）
    dest_path: str        # 目标路径（不能在 inbox）
) -> Path
# 返回：新的文件路径

# 3. 语义检索
results = manager.search_library(
    query: str,                      # 自然语言查询
    top_k: int = 5,                  # 返回结果数
    file_types: List[str] = ["pdf", "image"]  # 文件类型过滤
) -> List[Dict[str, Any]]
# 返回：[{path, score, metadata}, ...]

# 4. 读取文件
file_info = manager.read_file(
    path: str,
    include_content: bool = True,    # 是否包含内容
    include_meta: bool = True        # 是否包含元数据
) -> Dict[str, Any]
# 返回：{"kind": "text"|"image", "text": "...", "meta_data": {...}}

# 5. 列出目录内容
children = manager.list_children(path: Path | str) -> List[Path]
# 返回：目录下的文件和子目录列表

# 6. 重置文献库
manager.reset() -> None
# 删除所有文件和元数据，重新初始化
```

---

#### Agent

**初始化**：

```python
from pangu_agent.agent import Agent

agent = Agent(
    llm_client: LLMClient,    # LLM 客户端
    tools: List[Tool],        # 可用工具列表
    max_iterations: int = 10  # 最大迭代次数
)
```

**方法**：

```python
# 1. 添加系统提示词
agent.add_system_prompt(prompt: str) -> None

# 2. 添加用户消息
agent.add_user_message(message: str) -> None

# 3. 运行 Agent
result = agent.run(
    stop_condition: Optional[Callable[[ToolCall, ToolResult], bool]] = None
) -> Dict[str, Any]

# 返回格式：
{
    "success": bool,           # 是否成功完成
    "iterations": int,         # 实际迭代次数
    "stop_reason": str,        # "stop_condition_met" | "llm_finished" | "max_iterations"
    "result": Dict | None,     # 如果调用了 Finish 工具，包含结果
    "final_message": str | None  # 如果 LLM 返回文本
}
```

---

#### InteractiveService

**初始化**：

```python
from pangu_agent.services import InteractiveService

service = InteractiveService(
    manager: LibraryManager,      # 文献库管理器
    llm_client: LLMClient,         # LLM 客户端
    tools: List[Tool],             # 可用工具列表
    max_iterations: int = 15       # 每次对话的最大迭代次数
)
```

**方法**：

```python
# 1. 处理用户消息
result = service.chat(user_message: str) -> Dict[str, Any]

# 返回格式：
{
    "success": bool,            # 是否成功处理
    "response": str,            # Agent 的响应文本
    "iterations": int,          # 使用的迭代次数
    "stop_reason": str          # 停止原因
}

# 2. 重置对话历史
service.reset() -> None
# 清除所有对话历史，但保留系统提示词
```

**使用示例**：

```python
# 创建服务
service = InteractiveService(manager, llm_client, tools)

# 多轮对话
r1 = service.chat("搜索 Transformer 论文")
print(r1['response'])

r2 = service.chat("第一篇讲什么？")  # 记住上下文
print(r2['response'])

# 重置历史
service.reset()

# 新对话（没有之前的上下文）
r3 = service.chat("探索库结构")
print(r3['response'])
```

**特性**：

- ✅ 自动维护对话历史（上下文保持）
- ✅ 支持自然语言意图理解
- ✅ 可调用所有已注册的工具
- ✅ 自动处理工具调用循环
- ✅ 提供友好的错误处理

---

#### OpenCLIPEncoder

**初始化**：

```python
from pangu_agent.library.embeddings.encoder import OpenCLIPEncoder

encoder = OpenCLIPEncoder(
    model_name: str = "ViT-B-32",     # CLIP 模型名称
    pretrained: str = "openai",        # 预训练权重
    device: Optional[str] = None       # "cuda"|"cpu"（None 自动检测）
)
```

**方法**：

```python
# 1. 编码文本
embedding = encoder.encode_text(text: str) -> np.ndarray  # shape: (512,)

# 2. 编码图像
embedding = encoder.encode_image(image_path: Path) -> np.ndarray

# 3. 编码 PDF（提取文本后编码）
embedding = encoder.encode_pdf(pdf_path: Path) -> np.ndarray

# 4. 自动检测文件类型并编码
embedding = encoder.encode_file(file_path: Path) -> np.ndarray
```

---

## 系统组件详解

本章节提供各核心组件的深度技术剖析，适合想要理解实现细节或扩展系统的开发者。

### Agent 执行流程深度剖析

#### Tool-Calling Loop 完整流程

位于 [`pangu_agent/agent/agent.py`](pangu_agent/agent/agent.py)：

```python
def run(self, stop_condition=None):
    tool_schemas = self._tool_executor.schema()  # 生成 OpenAI function calling 格式

    for iteration in range(self._max_iterations):
        # 步骤 1：LLM 生成响应
        response = self._llm_client.completion(
            self._memory,           # 对话历史
            tools=tool_schemas,     # 可用工具列表
            raw=True                # 返回原始响应对象
        )

        if not response:
            return {"success": False, "stop_reason": "llm_no_response"}

        message = response.choices[0].message

        # 步骤 2：将 LLM 响应加入记忆
        self._memory.add_raw(message.model_dump(exclude_unset=True))

        # 步骤 3：处理工具调用
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # 3.1 执行工具
                tc = ToolCall(name=tool_name, arguments=tool_args, call_id=tool_call.id)
                result = self._tool_executor.execute(tc)

                # 3.2 将工具结果加入记忆
                self._memory.add(
                    "tool",
                    content=str(result.output) if result.success else result.error,
                    tool_call_id=tool_call.id
                )

                # 3.3 检查停止条件
                if stop_condition and stop_condition(tc, result):
                    return {
                        "success": True,
                        "iterations": iteration + 1,
                        "stop_reason": "stop_condition_met",
                        "result": result.output
                    }

        # 步骤 4：LLM 返回文本（主动结束）
        elif message.content:
            return {
                "success": True,
                "iterations": iteration + 1,
                "stop_reason": "llm_finished",
                "final_message": message.content
            }

    # 步骤 5：达到最大迭代次数
    return {
        "success": False,
        "iterations": self._max_iterations,
        "stop_reason": "max_iterations"
    }
```

#### 两类服务对比

| 特性 | AddLiteratureService | SearchFilesService |
|------|---------------------|-------------------|
| **目标** | 文件整理和分类 | 文献检索 |
| **可用工具** | ExploreLibrary, ViewFile, MoveFile, SearchLibrary | SearchLibrary, ViewFile, ExploreLibrary, Finish |
| **系统提示词** | `LITERATURE_ORGANIZER_SYSTEM_PROMPT` | `FILE_SEARCHER_SYSTEM_PROMPT` |
| **停止条件** | `move_file` 成功执行 | `finish` 工具调用 |
| **最大迭代** | 5 | 10 |
| **典型流程** | 暂存→分析→探索→决定→移动 | 检索→查看→过滤→总结→返回 |
| **返回结果** | 移动后的文件路径 | 文件列表 + 观察说明 |

---

### 嵌入式检索系统实现

#### OpenCLIP 编码流程详解

位于 [`pangu_agent/library/embeddings/encoder.py`](pangu_agent/library/embeddings/encoder.py)：

**PDF 编码**：

```
PDF 文件
  ↓ pypdf.PdfReader
提取文本
  ↓ 截取前 10000 字符（避免超过 CLIP token 限制）
CLIP Tokenizer
  ↓ clip.tokenize([text])
Token Tensor [1, 77]  # CLIP 最大 77 tokens
  ↓ model.encode_text()
Feature Tensor [1, 512]
  ↓ F.normalize()  # L2 归一化
归一化嵌入 [1, 512]
  ↓ .cpu().numpy()
NumPy Array (512,)
```

**图像编码**：

```
图像文件
  ↓ PIL.Image.open()
PIL Image (RGB)
  ↓ preprocess (Resize 224x224, Center Crop, Normalize)
Tensor [1, 3, 224, 224]
  ↓ model.encode_image()
Feature Tensor [1, 512]
  ↓ F.normalize()
归一化嵌入 [1, 512]
  ↓ .cpu().numpy()
NumPy Array (512,)
```

**关键参数**：

| 参数 | 值 | 说明 |
|------|---|------|
| 模型 | ViT-B-32 | Vision Transformer Base, 32x32 patch |
| 预训练权重 | OpenAI | 在 4 亿图文对上训练 |
| 嵌入维度 | 512 | 输出向量维度 |
| 最大文本长度 | 77 tokens | CLIP 的上下文长度 |
| 图像尺寸 | 224×224 | 输入图像分辨率 |

---

#### ChromaDB 检索流程

位于 [`pangu_agent/library/embeddings/vector_store.py`](pangu_agent/library/embeddings/vector_store.py)：

**添加嵌入**：

```python
def add(self, file_id: str, embedding: np.ndarray, metadata: Dict):
    self._collection.upsert(
        ids=[file_id],
        embeddings=[embedding.tolist()],
        metadatas=[metadata]
    )
```

**语义检索**：

```python
def search(
    self,
    query_embedding: np.ndarray,
    top_k: int = 5,
    where: Optional[Dict] = None  # 元数据过滤条件
) -> List[SearchResult]:
    results = self._collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where,  # 例如: {"file_type": {"$in": ["pdf"]}}
        include=["metadatas", "distances"]
    )

    # ChromaDB 返回的是 distance（越小越好）
    # 转换为 similarity（越大越好）
    search_results = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance  # Cosine 距离转相似度

        search_results.append(SearchResult(
            file_id=results["ids"][0][i],
            score=similarity,
            metadata=results["metadatas"][0][i]
        ))

    return search_results
```

**HNSW 索引参数**（ChromaDB 默认）：

- **M**: 16（每个节点的邻居数）
- **ef_construction**: 200（构建时的搜索宽度）
- **ef_search**: 10（查询时的搜索宽度）

---

## 测试与开发

### 运行测试

项目包含全面的测试套件，覆盖所有核心组件。

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_add_literature.py -v

# 运行带覆盖率报告
pytest tests/ --cov=pangu_agent --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### 测试覆盖

| 测试文件 | 测试内容 | 行数 |
|---------|---------|------|
| `test_agent.py` | Agent 工具调用循环、记忆管理 | 185 |
| `test_tools.py` | 5 种工具的功能测试 | 351 |
| `test_add_literature.py` | 端到端文献添加流程 | ~270 |
| `test_search_files.py` | 端到端检索流程 | ~190 |
| `test_vector_store.py` | ChromaDB 操作 | ~110 |
| `test_encoder.py` | OpenCLIP 编码 | ~50 |

**总覆盖率**：约 80%

### 测试数据准备

```bash
# 下载真实 arXiv 论文作为测试数据
python tests/download_test_data.py

# 将下载的论文到 tests/data/
# 包括：Attention Is All You Need, BERT, ResNet 等经典论文
```

### 开发建议

#### 代码风格

- **类型注解**：所有函数参数和返回值必须有类型提示
- **Docstring**：使用 Google 风格文档字符串
- **格式化**：使用 `black` 和 `isort`

```bash
# 格式化代码
black pangu_agent/
isort pangu_agent/
```

#### 添加新工具

1. 在 `pangu_agent/tools/` 下创建新文件
2. 继承 `Tool` 基类
3. 实现 `run()` 方法
4. 添加到工具列表

```python
# pangu_agent/tools/my_tool.py
from pangu_agent.tools.base import Tool, ToolParameter

class MyTool(Tool):
    def __init__(self, manager):
        super().__init__(
            name="my_tool",
            description="Description visible to LLM",
            parameters=[...]
        )
        self._manager = manager

    def run(self, arguments):
        # 实现逻辑
        return "Result string"
```

#### 添加新服务

1. 在 `pangu_agent/services/` 下创建新文件
2. 定义服务类
3. 使用 Agent 编排工具调用

---

## 常见问题与故障排除

### 安装问题

#### Q: PyTorch 安装失败

**症状**：

```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

**解决方案**：

```bash
# 方案 1：使用 PyTorch 官方镜像
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 方案 2：使用清华镜像
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方案 3：手动下载 whl 文件
# 访问 https://download.pytorch.org/whl/torch_stable.html
```

#### Q: ChromaDB 权限错误

**症状**：

```
PermissionError: [Errno 13] Permission denied: './library/.vector_store/'
```

**解决方案**：

```bash
# 修复权限
chmod -R u+w ./library/.vector_store/

# 或删除并重建
rm -rf ./library/.vector_store/
pangu run --action reset --library ./library
```

#### Q: uv sync 报错

**症状**：

```
error: Failed to download distributions
```

**解决方案**：

```bash
# 清理缓存
uv cache clean

# 重新同步
uv sync

# 或回退到 pip
pip install -e .
```

---

### 使用问题

#### Q: 检索结果不准确

**可能原因和解决方案**：

1. **嵌入模型未正确加载**

```python
# 验证模型
from pangu_agent.library.embeddings.encoder import OpenCLIPEncoder

encoder = OpenCLIPEncoder()
test_embedding = encoder.encode_text("test")
print(test_embedding.shape)  # 应该是 (512,)
```

2. **查询过于宽泛**

```bash
# ❌ 不推荐
pangu run --action search --prompt "AI"

# ✅ 推荐
pangu run --action search --prompt "Transformer 模型在机器翻译中的应用"
```

3. **文献库中文件太少**

```bash
# 至少添加 10+ 篇论文以获得有意义的检索结果
```

#### Q: LLM 没有调用工具

**症状**：

Agent 运行后直接返回文本，未调用任何工具。

**可能原因**：

1. **系统提示词不清晰**
2. **LLM 认为任务已完成**
3. **max_iterations 太小**

**解决方案**：

```python
# 1. 检查系统提示词
agent.add_system_prompt(
    "You MUST use tools to complete the task. "
    "Do not return text without using tools first."
)

# 2. 增加最大迭代次数
agent = Agent(llm_client, tools, max_iterations=15)

# 3. 查看日志
import logging
logging.basicConfig(level=logging.INFO)
```

#### Q: 添加文件后找不到

**排查步骤**：

```bash
# 1. 检查文件是否在 inbox
ls -la library/.inbox/

# 2. 检查元数据是否生成
ls -la library/.inbox/.*.meta.json

# 3. 检查向量库
ls -la library/.vector_store/

# 4. 搜索文件名
pangu run --action search --prompt "文件名的关键词"
```

#### Q: 元数据未生成或为空

**可能原因**：

1. LLM API 配置错误
2. PDF 提取失败
3. 图像格式不支持

**验证 LLM 配置**：

```python
from pangu_agent.llm_client import LLMClient

client = LLMClient()
response = client.completion_text("Hello, are you working?")
print(response)  # 应该返回 LLM 的回复
```

**验证 PDF 提取**：

```python
from pypdf import PdfReader

reader = PdfReader("your_paper.pdf")
text = "".join([page.extract_text() for page in reader.pages])
print(len(text))  # 应该 > 0
```

---

### 性能优化

#### 慢速检索优化

**问题**：检索耗时超过 5 秒。

**优化方案**：

1. **减少 top_k**

```bash
# 默认每种类型返回 3 个，总共可能 6 个
# 可以通过修改代码减少

# 或使用 SearchLibraryTool 时指定
# （需要在代码中使用，CLI 不支持）
```

2. **限制文件类型**

```python
# 只搜索 PDF
results = manager.search_library(
    query="transformer",
    file_types=["pdf"]  # 不搜索图像
)
```

3. **使用 GPU 加速**

```bash
# 确保 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
```

#### 内存占用优化

**问题**：运行时内存占用超过 4GB。

**优化方案**：

```python
# 1. 使用更小的嵌入模型
encoder = OpenCLIPEncoder(model_name="ViT-B-16")  # 而非 ViT-L-14

# 2. 批量处理时控制并发
# 不要一次添加超过 100 个文件

# 3. 定期清理 ChromaDB
manager.reset()  # 谨慎！会删除所有数据
```

---

## 路线图与贡献

### 当前状态

#### ✅ 已实现功能

- **固定任务模式**（CLI）
  - ✅ 添加文献（`pangu run --action add`）
  - ✅ 语义检索（`pangu run --action search`）
  - ✅ 重置文献库（`pangu run --action reset`）

- **交互模式**（CLI）
  - ✅ 多轮对话接口（`pangu interactive`）
  - ✅ 上下文保持和追问
  - ✅ 意图理解和任务分解
  - ✅ 集成所有核心工具（搜索、添加、浏览、查看、移动）

- **核心组件**
  - ✅ Agent 控制器（Tool-Calling Loop）
  - ✅ LibraryManager（文件+元数据+向量）
  - ✅ 5 种核心工具 + AddLiteratureTool
  - ✅ OpenCLIP + ChromaDB 检索系统
  - ✅ InteractiveService（对话服务）

- **测试覆盖**
  - ✅ 8 个测试文件
  - ✅ ~80% 代码覆盖率

#### 🚧 开发中功能

- **元数据字段扩展**
  - title, authors, year, venue
  - keywords, citation_count, doi