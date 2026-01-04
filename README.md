# 本地 AI 智能文献与图像管理助手 (Local Multimodal AI Agent)

## 1. 项目简介 (Project Introduction)
本项目是一个基于 Python 的本地多模态 AI 智能助手，旨在解决本地大量文献和图像素材管理困难的问题。不同于传统的文件名搜索，本项目利用多模态神经网络技术，实现对内容的**语义搜索**和**自动分类**。本项目完全本地化运行，确保数据隐私，同时利用高性能的嵌入模型提供精准的检索服务。

## 2. 核心功能 (Core Features)

### 2.1 智能文献管理
*   **语义搜索**: 支持使用自然语言提问（如“Transformer 的核心架构是什么？”）。系统基于语义理解返回最相关的论文文件。
*   **自动分类**: 添加新论文时，根据指定的主题（如 "CV, NLP, RL"）自动分析内容，将其归类并移动到对应的子文件夹中。
*   **文件索引**: 自动建立本地向量索引，支持快速检索。

### 2.2 智能图像管理
*   **以文搜图**: 利用多模态图文匹配技术，支持通过自然语言描述（如“海边的日落”）来查找本地图片库中最匹配的图像。
*   **批量处理**: 支持一键导入整个文件夹的图片素材。

### 2.3 数据库管理
*   **状态监控**: 实时查看当前数据库中的图像和论文数量及最近添加记录。
*   **一键重置**: 支持快速清空数据库，方便重新构建知识库。

## 3. 环境配置 (Environment)

### 3.1 系统要求
*   **操作系统**: Windows / macOS / Linux
*   **Python 版本**: 建议 Python 3.8 及以上
*   **内存**: 建议 8GB 及以上（用于加载 Embedding 模型）

### 3.2 依赖安装
请确保已安装 Python 环境，然后通过以下命令安装所需的第三方库：

```bash
pip install numpy pillow torch chromadb pypdf2 sentence-transformers open-clip-torch
```

> **注意**: 如果您的机器支持 GPU (NVIDIA CUDA)，建议安装对应的 PyTorch 版本以获得更快的处理速度。

## 4. 使用说明 (Usage Instructions)

本项目提供统一的命令行入口 `main.py`，支持以下功能：

### 4.1 论文管理

**1. 添加并分类论文**
将 PDF 论文添加到数据库，并根据提供的主题自动分类移动到 `organized_papers_v2` 目录下。
```bash
python main.py add_paper ./papers/03.pdf --topics "机器学习,大数据,多模态"
```

**2. 搜索论文**
使用自然语言描述搜索相关论文。
```bash
python main.py search_paper  “机器学习是什么” --top_k 2
```

### 4.2 图像管理

**1. 添加单张图片**
```bash
python main.py add_image ./images/dog.jpg"
```

**2. 批量添加图片**
扫描指定文件夹下的所有支持格式图片（.jpg, .png 等）并导入数据库。
```bash
python main.py batch_add_images ./images"
```

**3. 以文搜图**
通过文本描述搜索图片。
```bash
python main.py search_image  “小猫” --top_k  2"
```

### 4.3 系统维护

**1. 查看数据库状态**
显示当前存储的论文和图片数量。
```bash
python main.py status
```

**2. 重置数据库**
清空所有数据（**慎用**）。
```bash
python main.py reset
```

## 5. 技术选型 (Technical Stack)

本项目采用轻量级且高效的本地化模型方案：

*   **文本嵌入 (Text Embedding)**: `SentenceTransformers` (模型: `all-MiniLM-L12-v2`)
    *   用于将论文文本转换为向量，实现语义搜索。
*   **图像嵌入 (Image Embedding)**: `OpenCLIP` (模型: `ViT-B-32`, 预训练: `openai`)
    *   用于将图像和文本映射到同一向量空间，实现以文搜图。
*   **向量数据库 (Vector Database)**: `ChromaDB`
    *   本地轻量级向量数据库，用于存储和检索高维向量数据。
*   **PDF 处理**: `PyPDF2`
    *   用于提取 PDF 文档中的文本内容。

