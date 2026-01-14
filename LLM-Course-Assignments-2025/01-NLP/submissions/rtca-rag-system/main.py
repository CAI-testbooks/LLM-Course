#!/usr/bin/env python3
"""
RTCA DO-160G RAG系统主程序入口
支持多种运行模式：初始化知识库、启动Web界面、启动API服务、训练模型、评估系统
"""

from src.api_app import FastAPIApp
from src.web_app import GradioApp
from src.evaluator import Evaluator
from src.rag_system import RAGSystem
from src.retriever import HybridRetriever
from src.model_manager import QwenModelManager
from src.vector_store import VectorStoreManager
from src.document_processor import DocumentProcessor
from src.logger import setup_logger, get_logger
from src.config.config_manager import RAGConfig
import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ==================== 导入自定义模块 ====================

# ==================== 常量定义 ====================
DEFAULT_CONFIG_PATH = "configs/config.yaml"
MODES = ["init", "web", "api", "train", "eval", "benchmark", "export"]


class SystemInitializer:
    """系统初始化器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = get_logger("initializer")

    def initialize_system(self) -> RAGSystem:
        """初始化完整的RAG系统"""
        self.logger.info("开始初始化RAG系统...")

        try:
            # 1. 检查和处理知识库
            self._check_knowledge_base()

            # 2. 加载向量数据库
            vector_store = self._load_vector_store()

            # 3. 加载文档块
            chunks = self._load_chunks()

            # 4. 创建检索器
            retriever = self._create_retriever(vector_store, chunks)

            # 5. 创建RAG系统
            rag_system = self._create_rag_system(retriever)

            self.logger.info("RAG系统初始化完成")
            return rag_system

        except Exception as e:
            self.logger.error(f"初始化RAG系统失败: {str(e)}", exc_info=True)
            raise

    def _check_knowledge_base(self):
        """检查知识库状态"""
        processed_path = Path(self.config.data.processed_chunks_path)
        vector_db_path = Path(self.config.data.vector_db_path)

        # 检查是否已处理文档
        if not processed_path.exists():
            self.logger.warning(f"处理后的文档不存在: {processed_path}")
            self.logger.info("请先运行 'python main.py --mode init' 初始化知识库")

        # 检查向量数据库
        if not vector_db_path.exists():
            self.logger.warning(f"向量数据库不存在: {vector_db_path}")
            self.logger.info("请先运行 'python main.py --mode init' 初始化知识库")

    def _load_vector_store(self):
        """加载向量数据库"""
        self.logger.info("加载向量数据库...")
        try:
            vector_store_manager = VectorStoreManager(self.config)
            vector_store = vector_store_manager.load_vector_store()
            self.logger.info(f"向量数据库加载成功")
            return vector_store
        except Exception as e:
            self.logger.error(f"加载向量数据库失败: {str(e)}")
            raise

    def _load_chunks(self):
        """加载文档块"""
        self.logger.info("加载文档块...")
        chunks_path = Path(self.config.data.processed_chunks_path)

        if not chunks_path.exists():
            self.logger.error(f"文档块文件不存在: {chunks_path}")
            raise FileNotFoundError(f"文档块文件不存在: {chunks_path}")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        self.logger.info(f"已加载 {len(chunks)} 个文档块")
        return chunks

    def _create_retriever(self, vector_store, chunks):
        """创建检索器"""
        self.logger.info(f"创建{self.config.retrieval.strategy.value}检索器...")
        retriever = HybridRetriever(vector_store, chunks, self.config)
        return retriever

    def _create_rag_system(self, retriever):
        """创建RAG系统"""
        self.logger.info("创建RAG系统...")
        rag_system = RAGSystem(self.config)
        rag_system.retriever = retriever
        return rag_system


class KnowledgeBaseManager:
    """知识库管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = get_logger("kb_manager")

    def initialize_knowledge_base(self, force_rebuild: bool = False):
        """初始化知识库"""
        self.logger.info("开始初始化知识库...")
        start_time = time.time()

        try:
            # 1. 检查原始文档
            raw_path = Path(self.config.data.raw_documents_path)
            if not raw_path.exists():
                self.logger.error(f"原始文档目录不存在: {raw_path}")
                raise FileNotFoundError(f"原始文档目录不存在: {raw_path}")

            # 检查是否有PDF文件
            pdf_files = list(raw_path.glob("*.pdf"))
            if not pdf_files:
                self.logger.error(f"在 {raw_path} 中未找到PDF文件")
                raise FileNotFoundError(f"未找到PDF文件")

            self.logger.info(f"找到 {len(pdf_files)} 个PDF文件")

            # 2. 处理文档
            processor = DocumentProcessor(self.config)
            documents = processor.process_directory(str(raw_path))

            if not documents:
                self.logger.error("文档处理失败，未生成任何文档")
                return False

            self.logger.info(f"处理完成，共 {len(documents)} 个文档")

            # 3. 分块处理
            self.logger.info("开始文档分块...")
            chunks = processor.chunk_documents(documents)
            self.logger.info(f"分块完成，共 {len(chunks)} 个文本块")

            # 4. 保存处理后的块
            processed_path = Path(self.config.data.processed_chunks_path)
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            processor.save_chunks(chunks, str(processed_path))
            self.logger.info(f"已保存处理后的块到: {processed_path}")

            # 5. 创建向量数据库
            self.logger.info("创建向量数据库...")
            vector_store_manager = VectorStoreManager(self.config)
            vector_store = vector_store_manager.create_vector_store(chunks)

            # 6. 保存统计信息
            stats = self._save_statistics(documents, chunks, start_time)
            self.logger.info(f"知识库初始化完成，统计信息: {stats}")

            return True

        except Exception as e:
            self.logger.error(f"初始化知识库失败: {str(e)}", exc_info=True)
            return False

    def _save_statistics(self, documents, chunks, start_time):
        """保存统计信息"""
        stats = {
            "documents": len(documents),
            "chunks": len(chunks),
            "avg_chunk_size": sum(len(c['content']) for c in chunks) / len(chunks) if chunks else 0,
            "min_chunk_size": min(len(c['content']) for c in chunks) if chunks else 0,
            "max_chunk_size": max(len(c['content']) for c in chunks) if chunks else 0,
            "processing_time": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "chunk_size": self.config.retrieval.chunk_size,
                "chunk_overlap": self.config.retrieval.chunk_overlap,
                "retrieval_strategy": self.config.retrieval.strategy.value
            }
        }

        stats_path = Path(
            self.config.data.processed_chunks_path).parent / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return stats

    def update_knowledge_base(self, new_document_path: str):
        """更新知识库"""
        self.logger.info(f"更新知识库，添加文档: {new_document_path}")

        # 1. 处理新文档
        processor = DocumentProcessor(self.config)
        new_documents = processor.load_pdf(new_document_path)

        if not new_documents:
            self.logger.error(f"处理新文档失败: {new_document_path}")
            return False

        # 2. 分块处理
        new_chunks = processor.chunk_documents(new_documents)

        # 3. 加载现有向量数据库
        vector_store_manager = VectorStoreManager(self.config)
        try:
            vector_store = vector_store_manager.load_vector_store()
        except Exception as e:
            self.logger.error(f"加载现有向量数据库失败: {str(e)}")
            return False

        # 4. 更新向量数据库
        vector_store_manager.update_vector_store(new_chunks)

        # 5. 更新文档块文件
        self._update_chunks_file(new_chunks)

        self.logger.info(f"知识库更新完成，添加了 {len(new_chunks)} 个新块")
        return True

    def _update_chunks_file(self, new_chunks):
        """更新文档块文件"""
        chunks_path = Path(self.config.data.processed_chunks_path)

        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                existing_chunks = json.load(f)
            existing_chunks.extend(new_chunks)
        else:
            existing_chunks = new_chunks

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(existing_chunks, f, ensure_ascii=False, indent=2)


class TrainingManager:
    """训练管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = get_logger("training_manager")

    def prepare_training_data(self, data_source: str = "generated"):
        """准备训练数据"""
        self.logger.info(f"准备训练数据，数据源: {data_source}")

        # 检查训练数据目录
        train_data_path = Path(self.config.data.fine_tune_data_path)
        if not train_data_path.exists():
            train_data_path.mkdir(parents=True, exist_ok=True)

        if data_source == "generated":
            return self._generate_training_data()
        elif data_source == "manual":
            return self._load_manual_data()
        else:
            raise ValueError(f"未知数据源: {data_source}")

    def _generate_training_data(self):
        """生成训练数据（基于现有知识库）"""
        self.logger.info("生成训练数据...")

        # 1. 加载RAG系统
        initializer = SystemInitializer(self.config)
        rag_system = initializer.initialize_system()

        # 2. 生成问答对
        training_data = []

        # 示例问题（实际应用中可以从文件加载或动态生成）
        sample_questions = [
            "RTCA DO-160G标准的目的是什么？",
            "温度高度试验包括哪些类型？",
            "第4章中A1类设备的定义是什么？",
            "湿热试验分为哪几类？",
            "防水性试验的W类和R类有什么区别？",
        ]

        for question in sample_questions:
            try:
                result = rag_system.answer(question)
                if not result['uncertain']:
                    training_data.append({
                        "question": question,
                        "answer": result['answer'],
                        "context": [doc['content'] for doc in result['retrieved_docs']],
                        "references": result['references']
                    })
            except Exception as e:
                self.logger.warning(f"生成问题 '{question}' 的训练数据失败: {str(e)}")

        # 保存训练数据
        output_file = Path(
            self.config.data.fine_tune_data_path) / "generated_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"生成 {len(training_data)} 个训练样本，已保存到: {output_file}")
        return training_data

    def _load_manual_data(self):
        """加载手动准备的训练数据"""
        manual_data_path = Path(
            self.config.data.fine_tune_data_path) / "manual_data.json"

        if not manual_data_path.exists():
            self.logger.error(f"手动训练数据文件不存在: {manual_data_path}")
            return []

        with open(manual_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        self.logger.info(f"加载 {len(training_data)} 个手动准备的训练样本")
        return training_data

    def train_model(self, data_source: str = "generated"):
        """训练模型"""
        from src.models.fine_tuner import FineTuner

        self.logger.info("开始训练模型...")

        try:
            # 1. 准备数据
            training_data = self.prepare_training_data(data_source)

            if not training_data:
                self.logger.error("没有可用的训练数据")
                return False

            # 2. 创建训练器
            fine_tuner = FineTuner(self.config)

            # 3. 分割训练/验证集
            train_size = int(0.8 * len(training_data))
            train_data = training_data[:train_size]
            eval_data = training_data[train_size:]

            # 4. 开始训练
            trainer = fine_tuner.train(train_data, eval_data)

            self.logger.info("模型训练完成")
            return True

        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}", exc_info=True)
            return False


class EvaluationManager:
    """评估管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = get_logger("evaluation_manager")
        self.evaluator = Evaluator()

    def evaluate_system(self, test_set: str = "default"):
        """评估系统"""
        self.logger.info(f"评估RAG系统，测试集: {test_set}")

        # 1. 初始化RAG系统
        initializer = SystemInitializer(self.config)
        rag_system = initializer.initialize_system()

        # 2. 加载测试数据
        test_data = self._load_test_data(test_set)

        if not test_data:
            self.logger.error("没有测试数据")
            return None

        # 3. 运行评估
        results = self._run_evaluation(rag_system, test_data)

        # 4. 保存结果
        self._save_results(results, test_set)

        return results

    def _load_test_data(self, test_set: str):
        """加载测试数据"""
        test_data_path = Path(self.config.data.evaluation_data_path)

        if test_set == "default":
            test_file = test_data_path / "test_questions.json"
        elif test_set == "cmedqa":
            test_file = test_data_path / "cmedqa_benchmark.json"
        elif test_set == "legalbench":
            test_file = test_data_path / "legalbench_benchmark.json"
        else:
            test_file = test_data_path / f"{test_set}.json"

        if not test_file.exists():
            self.logger.warning(f"测试文件不存在: {test_file}")
            return self._create_default_test_data()

        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        self.logger.info(f"加载 {len(test_data)} 个测试样本")
        return test_data

    def _create_default_test_data(self):
        """创建默认测试数据"""
        test_data = [
            {
                "id": "test_001",
                "question": "RTCA DO-160G标准的目的是什么？",
                "expected_answer": "规定了机载设备一系列最低标准环境试验条件",
                "expected_references": ["第1章"]
            },
            {
                "id": "test_002",
                "question": "温度高度试验包括哪些类型？",
                "expected_answer": "地面低温耐受试验、低温工作试验、地面高温耐受试验、高温工作试验等",
                "expected_references": ["第4章"]
            },
            {
                "id": "test_003",
                "question": "防水性试验的W类和R类有什么区别？",
                "expected_answer": "W类设备经受滴水试验，R类设备经受喷水试验",
                "expected_references": ["第10章"]
            }
        ]

        # 保存默认测试数据
        test_data_path = Path(self.config.data.evaluation_data_path)
        test_data_path.mkdir(parents=True, exist_ok=True)

        with open(test_data_path / "test_questions.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"创建了 {len(test_data)} 个默认测试样本")
        return test_data

    def _run_evaluation(self, rag_system, test_data):
        """运行评估"""
        predictions = []
        references = []
        citations = []
        gold_citations = []

        self.logger.info("开始运行评估...")

        for i, item in enumerate(test_data):
            try:
                # 获取系统回答
                result = rag_system.answer(item["question"])

                predictions.append(result['answer'])
                references.append(item.get("expected_answer", ""))

                # 处理引用
                pred_citations = [str(ref['metadata'].get('page', ''))
                                  for ref in result['references']]
                citations.append(pred_citations)

                gold_citations.append(item.get("expected_references", []))

                self.logger.info(f"完成测试样本 {i+1}/{len(test_data)}")

            except Exception as e:
                self.logger.warning(f"测试样本 {item.get('id', i)} 评估失败: {str(e)}")
                predictions.append("")
                references.append(item.get("expected_answer", ""))
                citations.append([])
                gold_citations.append(item.get("expected_references", []))

        # 计算评估指标
        evaluation_results = self.evaluator.evaluate_rag(
            predictions, references)
        citation_results = self.evaluator.evaluate_citation(
            predictions, gold_citations)

        # 合并结果
        results = {
            **evaluation_results,
            **citation_results,
            "test_set_size": len(test_data),
            "completed_tests": len([p for p in predictions if p]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        return results

    def _save_results(self, results, test_set: str):
        """保存评估结果"""
        output_dir = Path("outputs/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"evaluation_{test_set}_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.logger.info(f"评估结果已保存到: {output_file}")

        # 打印简要结果
        self._print_summary(results)

    def _print_summary(self, results):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)

        if 'rouge' in results:
            rouge = results['rouge']
            print(f"ROUGE-1:    {rouge.get('rouge1', 0):.4f}")
            print(f"ROUGE-2:    {rouge.get('rouge2', 0):.4f}")
            print(f"ROUGE-L:    {rouge.get('rougeL', 0):.4f}")

        print(f"BLEU分数:   {results.get('bleu', 0):.4f}")
        print(f"幻觉率:     {results.get('hallucination_rate', 0):.4f}")

        if 'citation_f1' in results:
            print(f"引用F1:     {results.get('citation_f1', 0):.4f}")
            print(f"引用精度:   {results.get('citation_precision', 0):.4f}")
            print(f"引用召回率: {results.get('citation_recall', 0):.4f}")

        print(
            f"完成测试:   {results.get('completed_tests', 0)}/{results.get('test_set_size', 0)}")
        print("="*60)


class ExportManager:
    """导出管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = get_logger("export_manager")

    def export_model(self, export_format: str = "onnx", output_dir: str = "./exported_models"):
        """导出模型"""
        self.logger.info(f"导出模型，格式: {export_format}")

        try:
            # 加载模型
            model_manager = QwenModelManager(self.config)
            model_manager.load_base_model()

            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if export_format == "onnx":
                self._export_to_onnx(model_manager, output_path)
            elif export_format == "torchscript":
                self._export_to_torchscript(model_manager, output_path)
            elif export_format == "safetensors":
                self._export_to_safetensors(model_manager, output_path)
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")

            self.logger.info(f"模型已导出到: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"导出模型失败: {str(e)}", exc_info=True)
            return False

    def _export_to_onnx(self, model_manager, output_path):
        """导出为ONNX格式"""
        import torch.onnx

        # 创建示例输入
        dummy_input = model_manager.tokenizer(
            "测试输入", return_tensors="pt").input_ids

        # 导出模型
        torch.onnx.export(
            model_manager.model,
            dummy_input,
            output_path / "model.onnx",
            opset_version=14,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )

        # 保存tokenizer配置
        model_manager.tokenizer.save_pretrained(output_path)

    def _export_to_torchscript(self, model_manager, output_path):
        """导出为TorchScript格式"""
        # 转换为torchscript
        scripted_model = torch.jit.script(model_manager.model)
        scripted_model.save(str(output_path / "model.pt"))

        # 保存tokenizer
        model_manager.tokenizer.save_pretrained(output_path)

    def _export_to_safetensors(self, model_manager, output_path):
        """导出为SafeTensors格式"""
        from safetensors.torch import save_file

        # 获取模型状态字典
        state_dict = model_manager.model.state_dict()

        # 保存为safetensors格式
        save_file(state_dict, output_path / "model.safetensors")

        # 保存配置文件
        model_manager.model.config.save_pretrained(output_path)
        model_manager.tokenizer.save_pretrained(output_path)


# ==================== 主程序函数 ====================
def init_mode(config: RAGConfig, force_rebuild: bool = False):
    """初始化模式"""
    logger = get_logger("init_mode")
    logger.info("执行初始化模式...")

    kb_manager = KnowledgeBaseManager(config)
    success = kb_manager.initialize_knowledge_base(force_rebuild)

    if success:
        logger.info("知识库初始化成功")
        return True
    else:
        logger.error("知识库初始化失败")
        return False


def web_mode(config: RAGConfig):
    """Web模式"""
    logger = get_logger("web_mode")
    logger.info("执行Web模式...")

    try:
        # 初始化RAG系统
        initializer = SystemInitializer(config)
        rag_system = initializer.initialize_system()

        # 创建并启动Web应用
        web_app = GradioApp(rag_system)
        app = web_app.create_web_app()

        logger.info(
            f"启动Web界面，地址: http://{config.api.web_host}:{config.api.web_port}")

        # 启动服务
        app.launch(
            server_name=config.api.web_host,
            server_port=config.api.web_port,
            debug=config.api.web_debug,
            share=False
        )

    except KeyboardInterrupt:
        logger.info("Web服务已停止")
    except Exception as e:
        logger.error(f"启动Web服务失败: {str(e)}", exc_info=True)
        raise


def api_mode(config: RAGConfig):
    """API模式"""
    logger = get_logger("api_mode")
    logger.info("执行API模式...")

    try:
        # 初始化RAG系统
        initializer = SystemInitializer(config)
        rag_system = initializer.initialize_system()

        # 创建并启动API服务
        api_app = FastAPIApp(rag_system)

        logger.info(
            f"启动API服务，地址: http://{config.api.fastapi_host}:{config.api.fastapi_port}")
        logger.info(
            f"API文档: http://{config.api.fastapi_host}:{config.api.fastapi_port}/docs")

        # 启动服务
        api_app.run(
            host=config.api.fastapi_host,
            port=config.api.fastapi_port,
            workers=config.api.fastapi_workers
        )

    except KeyboardInterrupt:
        logger.info("API服务已停止")
    except Exception as e:
        logger.error(f"启动API服务失败: {str(e)}", exc_info=True)
        raise


def train_mode(config: RAGConfig, data_source: str):
    """训练模式"""
    logger = get_logger("train_mode")
    logger.info("执行训练模式...")

    training_manager = TrainingManager(config)
    success = training_manager.train_model(data_source)

    if success:
        logger.info("模型训练成功")
        return True
    else:
        logger.error("模型训练失败")
        return False


def eval_mode(config: RAGConfig, test_set: str):
    """评估模式"""
    logger = get_logger("eval_mode")
    logger.info("执行评估模式...")

    evaluation_manager = EvaluationManager(config)
    results = evaluation_manager.evaluate_system(test_set)

    if results:
        logger.info("系统评估完成")
        return True
    else:
        logger.error("系统评估失败")
        return False


def benchmark_mode(config: RAGConfig):
    """基准测试模式"""
    logger = get_logger("benchmark_mode")
    logger.info("执行基准测试模式...")

    # 运行多个基准测试
    benchmarks = ["cmedqa", "legalbench"]
    all_results = {}

    for benchmark in benchmarks:
        logger.info(f"运行基准测试: {benchmark}")
        try:
            evaluation_manager = EvaluationManager(config)
            results = evaluation_manager.evaluate_system(benchmark)
            all_results[benchmark] = results
        except Exception as e:
            logger.error(f"基准测试 {benchmark} 失败: {str(e)}")
            all_results[benchmark] = {"error": str(e)}

    # 保存综合结果
    output_file = Path("outputs/results/benchmark_comparison.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"基准测试完成，结果已保存到: {output_file}")

    # 打印对比结果
    print("\n" + "="*60)
    print("基准测试对比结果")
    print("="*60)

    for benchmark, results in all_results.items():
        if 'error' in results:
            print(f"{benchmark}: 错误 - {results['error']}")
        else:
            rouge = results.get('rouge', {})
            print(f"{benchmark}:")
            print(f"  ROUGE-L: {rouge.get('rougeL', 0):.4f}")
            print(f"  BLEU:    {results.get('bleu', 0):.4f}")
            print(f"  幻觉率:  {results.get('hallucination_rate', 0):.4f}")

    print("="*60)


def export_mode(config: RAGConfig, export_format: str, output_dir: str):
    """导出模式"""
    logger = get_logger("export_mode")
    logger.info("执行导出模式...")

    export_manager = ExportManager(config)
    success = export_manager.export_model(export_format, output_dir)

    if success:
        logger.info("模型导出成功")
        return True
    else:
        logger.error("模型导出失败")
        return False


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="RTCA DO-160G RAG智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )

    # 必需参数
    parser.add_argument(
        "--mode",
        choices=MODES,
        required=True,
        help=f"运行模式: {', '.join(MODES)}"
    )

    # 可选参数
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径"
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重建知识库（init模式）"
    )

    parser.add_argument(
        "--data-source",
        choices=["generated", "manual"],
        default="generated",
        help="训练数据源（train模式）"
    )

    parser.add_argument(
        "--test-set",
        choices=["default", "cmedqa", "legalbench", "custom"],
        default="default",
        help="测试集名称（eval模式）"
    )

    parser.add_argument(
        "--format",
        choices=["onnx", "torchscript", "safetensors"],
        default="onnx",
        help="导出格式（export模式）"
    )

    parser.add_argument(
        "--output-dir",
        default="./exported_models",
        help="导出目录（export模式）"
    )

    parser.add_argument(
        "--host",
        help="主机地址（覆盖配置文件）"
    )

    parser.add_argument(
        "--port",
        type=int,
        help="端口号（覆盖配置文件）"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="日志级别"
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        print(f"请从模板创建配置文件:")
        print(f"  cp configs/config.example.yaml {args.config}")
        sys.exit(1)

    # 加载配置
    try:
        config = RAGConfig.from_yaml(args.config)
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        sys.exit(1)

    # 覆盖配置
    if args.host:
        if args.mode == "web":
            config.api.web_host = args.host
        else:
            config.api.fastapi_host = args.host

    if args.port:
        if args.mode == "web":
            config.api.web_port = args.port
        else:
            config.api.fastapi_port = args.port

    if args.debug:
        config.api.web_debug = True
        config.logging.level = "DEBUG"

    if args.log_level:
        config.logging.level = args.log_level

    # 设置日志
    logger = setup_logger(
        name="main",
        log_file=config.logging.file,
        log_level=config.logging.level,
        log_format=config.logging.format
    )

    # 打印启动信息
    logger.info("=" * 60)
    logger.info(f"启动 RTCA DO-160G RAG 系统")
    logger.info(f"模式: {args.mode}")
    logger.info(f"版本: {config.version}")
    logger.info(f"环境: {config.environment}")
    logger.info(f"模型: {config.model.base_model}")
    logger.info(f"设备: {config.model.device}")
    logger.info("=" * 60)

    # 根据模式执行相应操作
    success = False

    try:
        if args.mode == "init":
            success = init_mode(config, args.force_rebuild)

        elif args.mode == "web":
            web_mode(config)  # 阻塞运行

        elif args.mode == "api":
            api_mode(config)  # 阻塞运行

        elif args.mode == "train":
            success = train_mode(config, args.data_source)

        elif args.mode == "eval":
            success = eval_mode(config, args.test_set)

        elif args.mode == "benchmark":
            benchmark_mode(config)
            success = True

        elif args.mode == "export":
            success = export_mode(config, args.format, args.output_dir)

        else:
            logger.error(f"未知模式: {args.mode}")
            sys.exit(1)

        # 对于非阻塞模式，输出结果
        if args.mode not in ["web", "api"]:
            if success:
                logger.info(f"{args.mode} 模式执行成功")
                sys.exit(0)
            else:
                logger.error(f"{args.mode} 模式执行失败")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(0)

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # 检查Python版本
    import sys
    if sys.version_info < (3, 9):
        print("错误: Python 3.9 或更高版本是必需的")
        sys.exit(1)

    # 运行主程序
    main()
