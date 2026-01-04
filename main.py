import os
import argparse
import shutil
import numpy as np
import uuid
from PIL import Image
import torch
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import open_clip as clip
from datetime import datetime
import warnings

# 忽略无关警告，保持输出整洁
warnings.filterwarnings("ignore")

# 全局配置常量，便于后续修改维护
CONFIG = {
    "TEXT_EMBED_MODEL": "all-MiniLM-L12-v2",  # 微调文本模型，保持同量级性能
    "CLIP_MODEL_NAME": "ViT-B-32",
    "CLIP_PRETRAINED": "openai",
    "CHROMA_DB_PATH": os.path.abspath("./chroma_db_v2"),
    "ORGANIZED_PAPER_DIR": "organized_papers_v2",
    "MAX_PDF_TEXT_LENGTH": 10000,
    "MAX_STORED_DOC_LENGTH": 1000,
    "MAX_STORED_FULL_TEXT": 2000,
    "SUPPORTED_IMAGE_EXTENSIONS": ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
}


class LocalMultiModalAgent:
    """本地多模态AI助手，优化结构与兼容性"""

    def __init__(self):
        """初始化模型、数据库连接与目录结构"""
        print("初始化本地多模态AI助手...")

        # 初始化核心目录
        self._init_directories()

        # 加载文本嵌入模型（论文处理专用）
        self.text_embedding_model = self._load_text_model()
        print("文本嵌入模型加载完成")

        # 加载CLIP模型与预处理工具（图像处理专用）
        self.clip_device, self.clip_model, self.clip_preprocess = self._load_clip_model()
        print(f"图像模型加载完成（使用设备: {self.clip_device}）")

        # 初始化ChromaDB客户端与数据集合
        self.chroma_client = self._init_chroma_client()
        self.image_collection, self.paper_collection = self._init_chroma_collections()

        print("多模态AI助手初始化完成\n")

    def _init_directories(self):
        """初始化必要的本地目录，避免文件操作异常"""
        os.makedirs(CONFIG["CHROMA_DB_PATH"], exist_ok=True)
        os.makedirs(CONFIG["ORGANIZED_PAPER_DIR"], exist_ok=True)
        print(f"核心目录初始化完成")
        print(f"- 数据库路径: {CONFIG['CHROMA_DB_PATH']}")
        print(f"- 论文分类路径: {CONFIG['ORGANIZED_PAPER_DIR']}")

    def _load_text_model(self):
        """加载文本嵌入模型，增加异常重试逻辑"""
        try:
            return SentenceTransformer(CONFIG["TEXT_EMBED_MODEL"])
        except Exception as e:
            print(f"默认文本模型加载失败: {e}")
            print("尝试回退到基础文本模型...")
            return SentenceTransformer('all-MiniLM-L6-v2')

    def _load_clip_model(self):
        """加载CLIP模型，优化设备兼容性（支持CUDA/MPS/CPU）"""
        # 设备优先级：CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        try:
            model, _, preprocess = clip.create_model_and_transforms(
                CONFIG["CLIP_MODEL_NAME"],
                pretrained=CONFIG["CLIP_PRETRAINED"],
                device=device
            )
            return device, model, preprocess
        except Exception as e:
            print(f"目标设备{device}加载CLIP失败: {e}")
            print("回退到CPU加载CLIP模型...")
            model, _, preprocess = clip.create_model_and_transforms(
                CONFIG["CLIP_MODEL_NAME"],
                pretrained=CONFIG["CLIP_PRETRAINED"],
                device="cpu"
            )
            return "cpu", model, preprocess

    def _init_chroma_client(self):
        """初始化ChromaDB客户端，兼容新旧版API"""
        print(f"初始化ChromaDB数据库（路径: {CONFIG['CHROMA_DB_PATH']}）")
        try:
            # 优先使用新版PersistentClient
            return chromadb.PersistentClient(path=CONFIG["CHROMA_DB_PATH"])
        except Exception as e1:
            print(f"新版ChromaDB API失败: {e1}")
            try:
                # 回退到旧版Client API
                return chromadb.Client(
                    Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=CONFIG["CHROMA_DB_PATH"],
                        anonymized_telemetry=False
                    )
                )
            except Exception as e2:
                print(f"旧版ChromaDB API失败: {e2}")
                print("回退到内存模式（数据不会持久化）")
                return chromadb.Client()

    def _init_chroma_collections(self):
        """初始化图像与论文数据集合，确保集合创建成功"""
        print("初始化ChromaDB数据集合...")

        # 初始化图像集合
        try:
            image_collection = self.chroma_client.get_or_create_collection(
                name="images_v2",
                metadata={"description": "图像向量数据库（重构版）", "version": "2.0"}
            )
        except Exception as e:
            print(f"图像集合创建失败: {e}")
            image_collection = self.chroma_client.create_collection(name="images_v2")

        # 初始化论文集合
        try:
            paper_collection = self.chroma_client.get_or_create_collection(
                name="papers_v2",
                metadata={"description": "论文向量数据库", "version": "2.0"}
            )
        except Exception as e:
            print(f"论文集合创建失败: {e}")
            paper_collection = self.chroma_client.create_collection(name="papers_v2")

        # 打印集合状态
        img_count = image_collection.count()
        paper_count = paper_collection.count()
        print(f"图像集合就绪（当前总数: {img_count}）")
        print(f"论文集合就绪（当前总数: {paper_count}）")

        return image_collection, paper_collection

    def extract_text_from_pdf(self, pdf_file_path):
        """从PDF文件中提取文本内容，增加格式优化与长度限制"""
        if not os.path.exists(pdf_file_path) or not pdf_file_path.endswith(".pdf"):
            print(f"无效的PDF文件路径: {pdf_file_path}")
            return ""

        try:
            print(f"正在提取PDF文本: {os.path.basename(pdf_file_path)}")
            pdf_reader = PdfReader(pdf_file_path)
            extracted_text = []

            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    extracted_text.append(f"[第{page_num}页]\n{page_text.strip()}\n")

            full_text = "\n".join(extracted_text)
            # 限制文本长度，避免嵌入模型过载
            return full_text[:CONFIG["MAX_PDF_TEXT_LENGTH"]]
        except Exception as e:
            print(f"PDF文本提取失败: {str(e)}")
            return ""

    def add_paper(self, paper_pdf_path, topics_str):
        """添加论文到数据库并根据主题自动分类，优化嵌入与存储逻辑"""
        # 1. 提取PDF文本
        pdf_text = self.extract_text_from_pdf(paper_pdf_path)
        if not pdf_text:
            print("论文添加失败：无法提取有效文本")
            return False

        # 2. 生成文本嵌入向量
        print("正在生成论文文本嵌入向量...")
        paper_embedding = self.text_embedding_model.encode(pdf_text)

        # 3. 构造论文元数据与唯一ID
        paper_filename = os.path.basename(paper_pdf_path)
        paper_unique_id = f"paper_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        paper_metadata = {
            "file_path": os.path.abspath(paper_pdf_path),
            "file_name": paper_filename,
            "short_content": pdf_text[:CONFIG["MAX_STORED_DOC_LENGTH"]],
            "full_content_preview": pdf_text[:CONFIG["MAX_STORED_FULL_TEXT"]],
            "added_timestamp": datetime.now().isoformat(),
            "file_type": "pdf"
        }

        # 4. 存入ChromaDB论文集合
        try:
            self.paper_collection.add(
                documents=[paper_metadata["short_content"]],
                embeddings=[paper_embedding.tolist()],
                metadatas=[paper_metadata],
                ids=[paper_unique_id]
            )

            # 5. 论文主题分类
            self._classify_paper_by_topic(paper_pdf_path, paper_filename, pdf_text, topics_str)

            # 6. 打印成功信息
            current_paper_count = self.paper_collection.count()
            print(f"论文添加成功（ID: {paper_unique_id}，当前总论文数：{current_paper_count}）")
            return True
        except Exception as e:
            print(f"论文存入数据库失败: {str(e)}")
            return False

    def _classify_paper_by_topic(self, paper_path, paper_filename, paper_text, topics_str):
        """根据主题相似度为论文分类，优化相似度计算逻辑"""
        topics_list = [topic.strip() for topic in topics_str.split(",") if topic.strip()]
        if not topics_list:
            print("未提供有效主题，跳过论文分类")
            return

        # 计算论文与各主题的相似度
        print(f"正在根据 {len(topics_list)} 个主题进行分类...")
        topic_embeddings = self.text_embedding_model.encode(topics_list)
        paper_embedding = self.text_embedding_model.encode(paper_text[:1000])  # 取文本前1000字提升效率
        similarity_scores = np.dot(paper_embedding, topic_embeddings.T)

        # 选择相似度最高的主题进行分类
        best_topic_idx = np.argmax(similarity_scores)
        best_topic = topics_list[best_topic_idx]
        best_similarity = round(similarity_scores[best_topic_idx], 4)

        # 创建主题目录并复制论文
        topic_directory = os.path.join(CONFIG["ORGANIZED_PAPER_DIR"], best_topic)
        os.makedirs(topic_directory, exist_ok=True)
        dest_paper_path = os.path.join(topic_directory, paper_filename)

        shutil.copy2(paper_path, dest_paper_path)
        print(f"论文已分类至：{topic_directory}（相似度：{best_similarity}）")

    def search_papers(self, search_query, top_k=5):
        """搜索论文数据库，返回相似度最高的结果，优化结果解析逻辑"""
        if not search_query or not isinstance(search_query, str):
            print("无效的搜索查询：必须为非空字符串")
            return []

        print(f"正在搜索论文：'{search_query}'（返回前{top_k}条结果）")

        # 验证论文集合是否有数据
        if self.paper_collection.count() == 0:
            print("论文集合为空，无搜索结果")
            return []

        try:
            # 生成查询嵌入向量
            query_embedding = self.text_embedding_model.encode(search_query)

            # 执行向量搜索
            search_results = self.paper_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # 解析并格式化搜索结果
            formatted_results = []
            if search_results and search_results['ids'][0]:
                for idx, paper_id in enumerate(search_results['ids'][0]):
                    metadata = search_results['metadatas'][0][idx]
                    distance = search_results['distances'][0][idx]
                    document = search_results['documents'][0][idx]

                    result_item = {
                        "paper_id": paper_id,
                        "file_name": metadata.get("file_name", "未知文件名"),
                        "file_path": metadata.get("file_path", "未知路径"),
                        "similarity": round(1 - distance, 4),
                        "content_snippet": f"{document[:200]}..." if document else "",
                        "add_time": metadata.get("added_timestamp", "未知时间")
                    }
                    formatted_results.append(result_item)

            print(f"搜索完成，找到 {len(formatted_results)} 条相关论文")
            return formatted_results
        except Exception as e:
            print(f"论文搜索失败: {str(e)}")
            return []

    def add_image(self, image_file_path):
        """添加单张图片到数据库，优化图像预处理与向量标准化"""
        if not os.path.exists(image_file_path):
            print(f"图片文件不存在: {image_file_path}")
            return False

        if not image_file_path.lower().endswith(CONFIG["SUPPORTED_IMAGE_EXTENSIONS"]):
            print(f"不支持的图片格式：{os.path.splitext(image_file_path)[1]}")
            return False

        try:
            print(f"正在处理图片: {os.path.basename(image_file_path)}")
            # 1. 图像加载与预处理
            with Image.open(image_file_path).convert("RGB") as img:
                # 缩略图处理，提升效率
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                img_width, img_height = img.size

                # 2. CLIP模型预处理与向量生成
                img_input = self.clip_preprocess(img).unsqueeze(0).to(self.clip_device)
                with torch.no_grad():
                    img_embedding = self.clip_model.encode_image(img_input)
                    # 向量标准化，提升搜索精度
                    img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)

            # 3. 构造图片元数据与唯一ID
            img_filename = os.path.basename(image_file_path)
            img_unique_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            img_metadata = {
                "image_path": os.path.abspath(image_file_path),
                "file_name": img_filename,
                "image_size": f"{img_width}x{img_height}",
                "added_timestamp": datetime.now().isoformat(),
                "file_format": os.path.splitext(img_filename)[1][1:]
            }

            # 4. 存入ChromaDB图像集合
            self.image_collection.add(
                embeddings=[img_embedding.squeeze().cpu().numpy().tolist()],
                metadatas=[img_metadata],
                ids=[img_unique_id]
            )

            # 5. 打印成功信息
            current_img_count = self.image_collection.count()
            print(f"图片添加成功（ID: {img_unique_id}，当前总图片数：{current_img_count}）")
            return True
        except Exception as e:
            print(f"图片添加失败: {str(e)}")
            return False

    def batch_add_images(self, image_folder_path):
        """批量添加文件夹中的所有图片，优化批量处理效率"""
        if not os.path.isdir(image_folder_path):
            print(f"无效的文件夹路径: {image_folder_path}")
            return

        print(f"开始批量处理图片文件夹: {image_folder_path}")

        # 遍历文件夹获取所有支持的图片文件
        image_file_list = []
        for root, _, files in os.walk(image_folder_path):
            for file in files:
                if file.lower().endswith(CONFIG["SUPPORTED_IMAGE_EXTENSIONS"]):
                    image_file_list.append(os.path.join(root, file))

        if not image_file_list:
            print("文件夹中未找到支持格式的图片")
            return

        # 批量处理每张图片
        success_count = 0
        fail_count = 0
        total_count = len(image_file_list)

        print(f"共找到 {total_count} 张支持格式的图片，开始批量添加...")
        for idx, img_path in enumerate(image_file_list, 1):
            print(f"\n[{idx}/{total_count}] 处理: {os.path.basename(img_path)}")
            if self.add_image(img_path):
                success_count += 1
            else:
                fail_count += 1

        # 打印批量处理统计结果
        print(f"\n批量添加完成统计：")
        print(f"成功添加: {success_count} 张")
        print(f"添加失败: {fail_count} 张")
        print(f"当前图像集合总数量: {self.image_collection.count()}")

    def search_images(self, text_query, top_k=5):
        """以文搜图，根据文本描述搜索最相似的图片，优化向量匹配逻辑"""
        if not text_query or not isinstance(text_query, str):
            print("无效的搜索查询：必须为非空字符串")
            return []

        if top_k <= 0:
            print("无效的返回数量：必须为正整数")
            return []

        print(f"正在以文搜图：'{text_query}'（返回前{top_k}条结果）")

        # 验证图像集合是否有数据
        img_collection_count = self.image_collection.count()
        if img_collection_count == 0:
            print("图像集合为空，无搜索结果")
            return []

        try:
            # 1. 生成文本查询向量
            text_input = clip.tokenize([text_query]).to(self.clip_device)
            with torch.no_grad():
                text_embedding = self.clip_model.encode_text(text_input)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            # 2. 执行向量搜索
            search_results = self.image_collection.query(
                query_embeddings=[text_embedding.cpu().numpy().tolist()],
                n_results=min(top_k, img_collection_count),
                include=["metadatas", "distances"]
            )

            # 3. 解析并格式化搜索结果
            formatted_results = []
            if search_results and search_results["metadatas"][0]:
                for idx, metadata in enumerate(search_results["metadatas"][0]):
                    distance = search_results["distances"][0][idx]
                    result_item = {
                        "file_name": metadata.get("file_name", "未知文件名"),
                        "image_path": metadata.get("image_path", "未知路径"),
                        "similarity": round(1 - distance, 4),
                        "image_size": metadata.get("image_size", "未知尺寸"),
                        "add_time": metadata.get("added_timestamp", "未知时间"),
                        "file_format": metadata.get("file_format", "未知格式")
                    }
                    formatted_results.append(result_item)

            # 4. 打印预览结果
            print(f"搜索完成，找到 {len(formatted_results)} 张相关图片")
            if formatted_results:
                print("\n前3条结果预览：")
                for idx, res in enumerate(formatted_results[:3], 1):
                    print(f"   {idx}. {res['file_name']}（相似度：{res['similarity']}）")

            return formatted_results
        except Exception as e:
            print(f"图片搜索失败: {str(e)}")
            return []

    def show_database_status(self):
        """显示数据库当前状态，包括集合数据量与最近数据"""
        print("本地多模态数据库状态（重构版）")

        # 显示图像集合状态
        try:
            img_count = self.image_collection.count()
            print(f"\n图像集合状态：")
            print(f"   - 总图片数量: {img_count}")
            if img_count > 0:
                recent_imgs = self.image_collection.get(limit=3)
                print(f"   - 最近添加的3张图片：")
                for idx, (img_id, meta) in enumerate(zip(recent_imgs['ids'], recent_imgs['metadatas']), 1):
                    print(f"     {idx}. {meta.get('file_name', 'N/A')}（ID: {img_id[:12]}...）")
        except Exception as e:
            print(f"图像集合状态：无法访问（{str(e)}）")

        # 显示论文集合状态
        try:
            paper_count = self.paper_collection.count()
            print(f"\n论文集合状态：")
            print(f"   - 总论文数量: {paper_count}")
            if paper_count > 0:
                recent_papers = self.paper_collection.get(limit=3)
                print(f"   - 最近添加的3篇论文：")
                for idx, (paper_id, meta) in enumerate(zip(recent_papers['ids'], recent_papers['metadatas']), 1):
                    print(f"     {idx}. {meta.get('file_name', 'N/A')}（ID: {paper_id[:12]}...）")
        except Exception as e:
            print(f"论文集合状态：无法访问（{str(e)}）")


    def reset_database(self):
        """重置数据库，删除所有集合数据（谨慎使用）"""
        confirm_input = input("\n警告：此操作将删除所有数据库数据，无法恢复！确认重置？(y/n): ")
        if confirm_input.lower() != 'y':
            print("取消数据库重置操作")
            return

        try:
            # 删除现有集合
            self.chroma_client.delete_collection(name="images_v2")
            self.chroma_client.delete_collection(name="papers_v2")
            print("已删除现有图像与论文集合")

            # 重新初始化集合
            self.image_collection, self.paper_collection = self._init_chroma_collections()
            print("数据库已重置，新集合初始化完成")
        except Exception as e:
            print(f"数据库重置失败: {str(e)}")


def main():
    """命令行入口函数，优化参数解析与命令映射"""
    parser = argparse.ArgumentParser(description="本地多模态AI管理工具",
                                     formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='可用命令列表')

    # 1. 添加论文命令
    add_paper_parser = subparsers.add_parser('add_paper', help='添加论文并根据主题分类')
    add_paper_parser.add_argument('path', help='论文PDF文件路径')
    add_paper_parser.add_argument('--topics', required=True, help='分类主题')

    # 2. 搜索论文命令
    search_paper_parser = subparsers.add_parser('search_paper', help='根据关键词搜索论文')
    search_paper_parser.add_argument('query', help='搜索关键词/描述')
    search_paper_parser.add_argument('--top_k', type=int, default=5, help='返回结果数量（默认5）')

    # 3. 添加单张图片命令
    add_image_parser = subparsers.add_parser('add_image', help='添加单张图片到数据库')
    add_image_parser.add_argument('path', help='图片文件路径')

    # 4. 批量添加图片命令
    batch_add_img_parser = subparsers.add_parser('batch_add_images', help='批量添加文件夹中的所有图片')
    batch_add_img_parser.add_argument('folder', help='图片文件夹路径')

    # 5. 以文搜图命令
    search_image_parser = subparsers.add_parser('search_image', help='根据文本描述搜索相似图片')
    search_image_parser.add_argument('query', help='图片描述关键词')
    search_image_parser.add_argument('--top_k', type=int, default=5, help='返回结果数量（默认5）')

    # 6. 显示数据库状态命令
    status_parser = subparsers.add_parser('status', help='显示数据库当前状态与数据统计')

    # 7. 重置数据库命令
    reset_parser = subparsers.add_parser('reset', help='重置数据库（删除所有数据）')

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化多模态助手
    multimodal_agent = LocalMultiModalAgent()

    # 命令分发执行
    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'add_paper':
            multimodal_agent.add_paper(args.path, args.topics)
        elif args.command == 'search_paper':
            results = multimodal_agent.search_papers(args.query, args.top_k)
            if results:
                print("\n论文搜索结果详情：")
                for idx, res in enumerate(results, 1):
                    print(f"\n{idx}. 文件名：{res['file_name']}")
                    print(f"   相似度：{res['similarity']}")
                    print(f"   路径：{res['file_path']}")
                    print(f"   内容片段：{res['content_snippet']}")
        elif args.command == 'add_image':
            multimodal_agent.add_image(args.path)
        elif args.command == 'batch_add_images':
            multimodal_agent.batch_add_images(args.folder)
        elif args.command == 'search_image':
            results = multimodal_agent.search_images(args.query, args.top_k)
            if results:
                print("\n图片搜索结果详情：")
                for idx, res in enumerate(results, 1):
                    print(f"\n{idx}. 文件名：{res['file_name']}")
                    print(f"   相似度：{res['similarity']}")
                    print(f"   路径：{res['image_path']}")
                    print(f"   尺寸：{res['image_size']}")
        elif args.command == 'status':
            multimodal_agent.show_database_status()
        elif args.command == 'reset':
            multimodal_agent.reset_database()
        else:
            parser.print_help()
    except Exception as e:
        print(f"\n命令执行失败: {str(e)}")


if __name__ == "__main__":
    main()