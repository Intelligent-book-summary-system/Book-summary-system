import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch
import gc
import os
from retrieval_db import RetrievalDatabase
from transformers import AutoTokenizer, AutoModel


def batch_encode(tokenizer, model, texts, batch_size=32):
    """Encode texts in batches to manage memory better"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # 添加指令前缀
        batch_texts = ["Represent this text for retrieval: " + text.replace("\n", " ")
                       for text in batch_texts]

        try:
            inputs = tokenizer(batch_texts,
                               padding=True,
                               truncation=True,
                               max_length=512,
                               return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0].numpy()
                # Normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
                all_embeddings.extend(embeddings)

            # Clear some memory
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        except Exception as e:
            print(f"Error processing batch {i}-{i + batch_size}: {e}")
            continue

    return np.array(all_embeddings)


def create_database(book_path, db_path, context_path=None, chunk_context_path=None, batch_size=32,
                    model_name="BAAI/bge-small-en-v1.5"):
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # 创建数据库
        print("Creating database...")
        db = RetrievalDatabase()

        # 加载书籍数据
        print("Loading book data...")
        with open(book_path, 'rb') as f:
            data = pickle.load(f)

        # 加载上下文数据（如果提供）
        book_contexts = {}
        if context_path and os.path.exists(context_path):
            print(f"Loading book contexts from {context_path}...")
            with open(context_path, 'rb') as f:
                book_contexts = pickle.load(f)

        # 加载块特定上下文（如果提供）
        chunk_contexts = {}
        if chunk_context_path and os.path.exists(chunk_context_path):
            print(f"Loading chunk contexts from {chunk_context_path}...")
            with open(chunk_context_path, 'rb') as f:
                chunk_contexts = pickle.load(f)

        # 收集所有要处理的文本
        print("Processing texts for database...")
        all_texts = []
        text_to_book_mapping = {}  # 用于跟踪每个文本来自哪本书

        # 处理方式根据提供的上下文类型
        using_context = (context_path is not None and book_contexts) or (
                    chunk_context_path is not None and chunk_contexts)

        for book_title, book_chunks in data.items():
            if chunk_context_path and book_title in chunk_contexts:
                # 使用针对每个块的特定上下文
                chunk_context_list = chunk_contexts[book_title]
                for i, context in enumerate(chunk_context_list):
                    if i < len(book_chunks):  # 确保索引有效
                        all_texts.append(context)
                        text_to_book_mapping[len(all_texts) - 1] = book_title
            elif context_path and book_title in book_contexts:
                # 使用整本书的总体上下文，为每个块创建上下文条目
                book_context = book_contexts[book_title]

                # 合并所有上下文类型
                combined_context = " ".join([
                    info for context_type, info in book_context.items()
                    if info and not info.startswith("Failed")
                ])

                # 为每个块添加相同的书籍上下文
                for _ in book_chunks:
                    all_texts.append(combined_context)
                    text_to_book_mapping[len(all_texts) - 1] = book_title
            else:
                # 如果没有上下文数据，使用原始文本
                for chunk in book_chunks:
                    all_texts.append(chunk)
                    text_to_book_mapping[len(all_texts) - 1] = book_title

        total_texts = len(all_texts)
        print(f"Total texts to process: {total_texts}")

        if total_texts == 0:
            raise ValueError("No texts to process. Check context paths or book data.")

        # 分批处理嵌入
        embeddings = []
        for i in range(0, total_texts, batch_size):
            print(f"Processing batch {i // batch_size + 1}/{(total_texts + batch_size - 1) // batch_size}")
            batch = all_texts[i:i + batch_size]
            batch_embeddings = batch_encode(tokenizer, model, batch, batch_size)
            embeddings.extend(batch_embeddings)

            # 定期保存进度
            if (i + batch_size) % 1000 == 0 or i + batch_size >= total_texts:
                processed = min(i + batch_size, total_texts)  # 修正进度计算
                print(f"Saving interim progress... ({processed}/{total_texts})")
                current_embeddings = np.array(embeddings)
                current_texts = all_texts[:len(embeddings)]

                # 清除现有数据库并重新创建
                db = RetrievalDatabase()
                db.add(current_embeddings, current_texts)
                db.save(f"{db_path}_interim")

        # 最终保存
        print("Saving final database...")
        embeddings = np.array(embeddings)
        db = RetrievalDatabase()
        db.add(embeddings, all_texts)
        db.save(db_path)

        # 保存文本到书籍的映射，用于后期分析或调试
        with open(f"{db_path}_mapping.pkl", 'wb') as f:
            pickle.dump(text_to_book_mapping, f)

        print("Database creation completed successfully!")
        # 展示数据库内容
        db.display_contents(num_entries=5)

        # 返回使用的上下文类型信息
        return {
            "using_context": using_context,
            "context_type": "chunk_specific" if chunk_context_path and chunk_contexts else "book_level" if context_path and book_contexts else "original_text"
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_path", type=str, required=True,
                        help="Path to the pickled book chunks file")
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to save the database")
    parser.add_argument("--context_path", type=str, default=None,
                        help="Path to the book contexts file (optional)")
    parser.add_argument("--chunk_context_path", type=str, default=None,
                        help="Path to the chunk-specific contexts file (optional)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")

    args = parser.parse_args()

    create_database(
        args.book_path,
        args.db_path,
        args.context_path,
        args.chunk_context_path,
        args.batch_size
    )