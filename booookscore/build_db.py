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


def create_database(book_path, db_path, batch_size=32, model_name="BAAI/bge-small-en-v1.5"):
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

        # 收集所有文本块
        print("Processing chunks...")
        all_chunks = []
        for book_chunks in data.values():
            all_chunks.extend(book_chunks)

        total_chunks = len(all_chunks)
        print(f"Total chunks to process: {total_chunks}")

        # 分批处理嵌入
        embeddings = []
        for i in range(0, total_chunks, batch_size):
            print(f"Processing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size}")
            batch = all_chunks[i:i + batch_size]
            batch_embeddings = batch_encode(tokenizer, model, batch, batch_size)
            embeddings.extend(batch_embeddings)

            # 定期保存进度
            if (i + batch_size) % 1000 == 0 or i + batch_size >= total_chunks:
                processed = min(i + batch_size, total_chunks)  # 修正进度计算
                print(f"Saving interim progress... ({processed}/{total_chunks})")
                current_embeddings = np.array(embeddings)
                current_chunks = all_chunks[:len(embeddings)]

                # 清除现有数据库并重新创建
                db = RetrievalDatabase()
                db.add(current_embeddings, current_chunks)
                db.save(f"{db_path}_interim")

        # 最终保存
        print("Saving final database...")
        embeddings = np.array(embeddings)
        db = RetrievalDatabase()
        db.add(embeddings, all_chunks)
        db.save(db_path)

        print("Database creation completed successfully!")
        # 展示数据库内容
        # db.display_contents(num_entries=10)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_path", type=str, required=True,
                        help="Path to the pickled book chunks file")
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to save the database")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")

    args = parser.parse_args()

    create_database(args.book_path, args.db_path, args.batch_size)