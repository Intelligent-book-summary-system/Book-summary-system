import os
import pickle
import time
import argparse
import json
import math
import tiktoken
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any, Union, Tuple
from .utils import APIClient, count_tokens
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from .retrieval_db import RetrievalDatabase


class Summarizer:
    def __init__(self,
                 model,
                 api,
                 api_key,
                 summ_path,
                 method,
                 chunk_size,
                 max_context_len, # 获取上下文的长度
                 max_summary_len, # 摘要的长度
                 retrieval_database=None,
                 hf_model="BAAI/bge-small-en-v1.5",
                 word_ratio=0.5 # 原文和和摘要的比例
                 ):
        self.client = APIClient(api, api_key, model)
        self.summ_path = summ_path
        assert method in ['inc', 'hier']
        self.method = method
        self.chunk_size = chunk_size
        self.max_context_len = max_context_len
        self.max_summary_len = max_summary_len
        self.word_ratio = word_ratio

        # Minimal Hugging Face Setup
        try:
            # Only load tokenizer initially
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
            self.embedding_model = None  # Will be loaded on first use
            self.hf_model_name = hf_model
            self.retrieval_database = retrieval_database

        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            raise

    def check_summary_validity(self, summary, token_limit):
        if len(summary) == 0:
            raise ValueError("Empty summary returned")

        # 检查是否包含请求用户提供更多信息的语句
        invalid_starts = ["unfortunately", "i don't", "please provide", "i'm ready", "it seems", "i don't see"]
        if any(summary.lower().startswith(start) for start in invalid_starts):
            return False

        if count_tokens(summary) > token_limit or summary[-1] not in ['.', '?', '!', '\"', '\'']:
            return False
        else:
            return True

    def summarize(self, texts, token_limit, level):
        text = texts['text']
        context = texts['context']
        word_limit = round(token_limit * self.word_ratio)
        if level == 0:
            prompt = self.templates['init_template'].format(text, word_limit)
        else:
            prompt = self.templates['template'].format(text, word_limit)
            if len(context) > 0 and level > 0:
                prompt = self.templates['context_template'].format(context, text, word_limit)
        response = self.client.obtain_response(prompt, max_tokens=token_limit, temperature=0.5)

        while len(response) == 0:
            print("Empty summary, retrying in 10 seconds...")
            time.sleep(10)
            response = self.client.obtain_response(prompt, max_tokens=token_limit, temperature=0.5)

        attempts = 0
        while not self.check_summary_validity(response, token_limit):
            word_limit = word_limit * (1 - 0.1 * attempts)
            if level == 0:
                prompt = self.templates['init_template'].format(text, word_limit)
            else:
                prompt = self.templates['template'].format(text, word_limit)
                if len(context) > 0 and level > 0:
                    prompt = self.templates['context_template'].format(context, text, word_limit)
            if attempts == 6:
                print("Failed to generate valid summary after 6 attempts, skipping")
                return response
            print(f"Invalid summary, retrying: attempt {attempts}")
            response = self.client.obtain_response(prompt, max_tokens=token_limit, temperature=1)
            attempts += 1
        return response

    def estimate_levels(self, book_chunks, summary_limit=450):
        num_chunks = len(book_chunks)
        chunk_limit = self.chunk_size
        levels = 0

        while num_chunks > 1:
            chunks_that_fit = (self.max_context_len - count_tokens(self.templates['template'].format('',
                                                                                                     0)) - 20) // chunk_limit  # number of chunks that could fit into the current context
            num_chunks = math.ceil(num_chunks / chunks_that_fit)  # number of chunks after merging
            chunk_limit = summary_limit
            levels += 1

        summary_limits = [self.max_summary_len]
        for _ in range(levels - 1):
            summary_limits.append(int(summary_limits[-1] * self.word_ratio))
        summary_limits.reverse()  # since we got the limits from highest to lowest, but we need them from lowest to highest
        return levels, summary_limits

    def get_huggingface_embedding(self, text):
        """Generate embeddings using BGE model"""
        # Print the text being embedded
        print("\nGenerating embedding for text:")
        print(f"Text snippet: {text[:200]}...")  # Print first 200 chars

        instruction = "Represent this text for retrieval: "
        text_with_instruction = instruction + text.replace("\n", " ")

        try:
            if self.embedding_model is None:
                print("\nLoading embedding model for the first time...")
                self.embedding_model = AutoModel.from_pretrained(self.hf_model_name)
                print("Model loaded successfully")

            inputs = self.tokenizer(
                text_with_instruction,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0].numpy()

            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            print("Embedding generated successfully")
            return embeddings[0]

        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def retrieve_context_for_chunk(self, chunk_text, top_k=3):
        """Retrieve context for a single chunk"""
        print(f"\nAttempting to retrieve context for chunk:")
        print(f"Chunk text preview: {chunk_text[:200]}...")  # Print first 200 chars

        if self.retrieval_database is None:
            print("No retrieval database available")
            return ""

        try:
            query_embedding = self.get_huggingface_embedding(chunk_text)
            if query_embedding is None:
                print("Failed to generate embedding for chunk")
                return ""

            print(f"\nSearching database with top_k={top_k}")
            results = self.retrieval_database.search(
                query_vector=query_embedding,
                k=top_k
            )

            contexts = []
            print("\nRetrieved contexts:")
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    context = result.get('text', '')
                    contexts.append(context)
                    print(f"Context {i + 1} preview: {context[:200]}...")  # Print first 200 chars
                elif isinstance(result, tuple) and len(result) >= 2:
                    context = result[1]
                    contexts.append(context)
                    print(f"Context {i + 1} preview: {context[:200]}...")  # Print first 200 chars

            combined_context = "\n\n".join(contexts)
            print(f"\nTotal contexts retrieved: {len(contexts)}")
            return combined_context

        except Exception as e:
            print(f"Chunk context retrieval error: {e}")
            return ""

    def recursive_summary(self, book, summaries, level, chunks, summary_limits):
        """Modified recursive_summary with per-chunk embeddings"""
        i = 0
        if level == 0 and len(summaries[book]['summaries_dict'][0]) > 0:
            i = len(summaries[book]['summaries_dict'][0])

        if level >= len(summary_limits):
            summary_limit = self.max_summary_len
        else:
            summary_limit = summary_limits[level]

        summaries_dict = summaries[book]['summaries_dict']

        # Calculate available tokens for context and concat
        if level > 0 and len(summaries_dict[level]) > 0:
            if count_tokens('\n\n'.join(chunks)) + self.max_summary_len + count_tokens(
                    self.templates['context_template'].format('', '', 0)) + 20 <= self.max_context_len:
                summary_limit = self.max_summary_len
            num_tokens = self.max_context_len - summary_limit - count_tokens(
                self.templates['context_template'].format('', '', 0)) - 20
        else:
            if count_tokens('\n\n'.join(chunks)) + self.max_summary_len + count_tokens(
                    self.templates['template'].format('', 0)) + 20 <= self.max_context_len:
                summary_limit = self.max_summary_len
            num_tokens = self.max_context_len - summary_limit - count_tokens(
                self.templates['template'].format('', 0)) - 20

        while i < len(chunks):
            # Get previous context
            context = ""
            if len(summaries_dict[level]) > 0:
                context = summaries_dict[level][-1]
                context_len = math.floor(0.2 * num_tokens)
                if count_tokens(context) > context_len:
                    context_tokens = encoding.encode(context)[:context_len]
                    context = encoding.decode(context_tokens)
                    if '.' in context:
                        context = context.rsplit('.', 1)[0] + '.'

            # Prepare current text
            if level == 0:
                text = chunks[i]
                # For level 0, retrieve context for individual chunk
                retrieved_context = self.retrieve_context_for_chunk(text)
                if retrieved_context:
                    if context:
                        context = f"{context}\n\nRelevant Context:\n{retrieved_context}"
                    else:
                        context = retrieved_context
            else:
                j = 1
                text = f"Summary {j}:\n\n{chunks[i]}"
                current_chunks = [chunks[i]]

                # Concatenate chunks that fit
                while i + 1 < len(chunks) and count_tokens(
                        context + text + f"\n\nSummary {j + 1}:\n\n{chunks[i + 1]}") + 20 <= num_tokens:
                    i += 1
                    j += 1
                    text += f"\n\nSummary {j}:\n\n{chunks[i]}"
                    current_chunks.append(chunks[i])

                # For higher levels, retrieve context based on concatenated chunks
                if self.retrieval_database is not None:
                    combined_context = []
                    for chunk in current_chunks:
                        chunk_context = self.retrieve_context_for_chunk(chunk)
                        if chunk_context:
                            combined_context.append(chunk_context)

                    if combined_context:
                        retrieved_context = "\n\n".join(combined_context)
                        if context:
                            context = f"{context}\n\nRelevant Context:\n{retrieved_context}"
                        else:
                            context = retrieved_context

            texts = {
                'text': text,
                'context': context
            }

            # Generate summary
            summary = self.summarize(texts, summary_limit, level)
            summaries_dict[level].append(summary)
            i += 1

        # Continue to next level if needed
        if len(summaries_dict[level]) > 1:
            return self.recursive_summary(book, summaries, level + 1, summaries_dict[level], summary_limits)
        else:
            return summaries_dict[level][0]

    def summarize_book(self, book, chunks, summaries):
        levels, summary_limits = self.estimate_levels(chunks)
        level = 0
        if len(summaries[book]['summaries_dict']) > 0:
            if len(summaries[book]['summaries_dict']) == 1:  # if there is only one level so far
                if len(summaries[book]['summaries_dict'][0]) == len(chunks):  # if level 0 is finished, set level to 1
                    level = 1
                elif len(summaries[book]['summaries_dict'][0]) < len(chunks):  # else, resume at level 0
                    level = 0
                else:
                    raise ValueError(f"Invalid summaries_dict at level 0 for {book}")
            else:  # if there're more than one level so far, resume at the last level
                level = len(summaries[book]['summaries_dict'])
            print(f"Resuming at level {level}")

        final_summary = self.recursive_summary(book, summaries, level, chunks, summary_limits)

        return final_summary, summaries

    def get_hierarchical_summaries(self, book_path):
        data = pickle.load(open(book_path, 'rb'))
        self.templates = {
            'init_template': open("prompts/get_summaries_hier/init.txt", "r").read(),
            'template': open("prompts/get_summaries_hier/merge.txt", "r").read(),
            'context_template': open("prompts/get_summaries_hier/merge_context.txt", "r").read()
        }
        summaries = defaultdict(dict)
        if os.path.exists(self.summ_path):
            print("Loading existing summaries...")
            summaries = json.load(open(self.summ_path, 'r'))
            # convert all keys into int
            for book in summaries:
                summaries[book]['summaries_dict'] = defaultdict(list, {int(k): v for k, v in
                                                                       summaries[book]['summaries_dict'].items()})

        for i, book in tqdm(enumerate(data), total=len(data), desc="Iterating over books"):
            if book in summaries:
                print("Already processed, skipping...")
                continue
            chunks = data[book]
            summaries[book] = {
                'summaries_dict': defaultdict(list)
            }
            final_summary, summaries = self.summarize_book(book, chunks, summaries)
            summaries[book]['final_summary'] = final_summary
            with open(self.summ_path, 'w') as f:
                json.dump(summaries, f)

    def compress(self, response, summary, chunk, summary_len, word_limit, num_chunks, j):
        chunk_trims = 0
        compressed_summary = None
        summary_words = len(summary.split())
        ori_expected_words = int(
            word_limit * j / num_chunks)  # no need to be j + 1 since we're compressing the summary at the previous chunk
        expected_words = ori_expected_words
        actual_words = expected_words

        dic = {}  # keep track of each trimmed summary and their actual number of words

        while response[-1] not in ['.', '?', '!', '\"', '\''] \
                or count_tokens(response) >= summary_len \
                or actual_words < int(expected_words * 0.8) or actual_words > int(expected_words * 1.2):
            if chunk_trims == 6:
                print(f"\nCOMPRESSION FAILED AFTER 6 ATTEMPTS, SKIPPING\n")
                if not all([v['valid_response'] == False for v in dic.values()]):
                    dic = {k: v for k, v in dic.items() if v['valid_response'] == True}
                closest_key = min(dic, key=lambda x: abs(
                    x - ori_expected_words))  # find the trimmed summary with actual # words closest to the expected # words
                return dic[closest_key]['compressed_summary'], dic[closest_key]['response'], chunk_trims, 1

            expected_words = int(ori_expected_words * (1 - chunk_trims * 0.05))
            prompt = self.templates["compress_template"].format(summary, summary_words, expected_words, expected_words)

            response = self.client.obtain_response(prompt, max_tokens=summary_len, temperature=1)
            compressed_summary = response
            actual_words = len(compressed_summary.split())
            current_tokens = count_tokens(compressed_summary)

            if compressed_summary[-1] not in ['.', '?', '!', '\"', '\''] \
                    or count_tokens(compressed_summary) >= summary_len \
                    or actual_words < int(expected_words * 0.8) or actual_words > int(expected_words * 1.2):
                chunk_trims += 1
                continue

            num_words = int(word_limit * (j + 1) / num_chunks)
            prompt = self.templates['template'].format(chunk, compressed_summary, num_words, num_words)
            response = self.client.obtain_response(prompt, max_tokens=summary_len, temperature=0.5)

            dic[actual_words] = {
                'compressed_summary': compressed_summary,
                'response': response,
                'valid_response': response[-1] in ['.', '?', '!', '\"', '\''] \
                                  and count_tokens(response) < summary_len
            }
            chunk_trims += 1

        return compressed_summary, response, chunk_trims, 0

    def get_incremental_summaries(self, book_path):
        data = pickle.load(open(book_path, 'rb'))
        self.templates = {
            "init_template": open("prompts/get_summaries_inc/init.txt", "r").read(),
            "template": open("prompts/get_summaries_inc/intermediate.txt", "r").read(),
            "compress_template": open("prompts/get_summaries_inc/compress.txt", "r").read()
        }

        num_trims = 0
        total_chunks = 0
        skipped_chunks = 0

        new_data = {}
        if os.path.exists(self.summ_path):
            new_data = json.load(open(self.summ_path, "r"))

        for i, book in tqdm(enumerate(data), total=len(data), desc="Iterating over books"):
            if book in new_data and len(new_data[book]) >= len(data[book]):
                print(f"Skipping {book}")
                continue
            total_chunks += len(data[book])
            new_chunks = []
            prev_summary = None
            if len(new_data) > i:
                new_chunks = new_data[book]
                prev_summary = new_chunks[-1]
            dd = data[book]
            word_limit = int(self.max_summary_len * self.word_ratio)
            num_chunks = len(dd)

            for j, chunk in tqdm(enumerate(dd), total=len(dd), desc="Iterating over chunks"):
                if j < len(new_chunks):
                    print(f"Skipping chunk {j}...")
                    continue
                new_chunk = {}
                num_words = int(word_limit * (j + 1) / len(dd))
                if prev_summary is None:
                    prompt = self.templates['init_template'].format(chunk)
                else:
                    prompt = self.templates['template'].format(chunk, prev_summary)

                response = self.client.obtain_response(prompt, max_tokens=self.max_summary_len, temperature=0.5)
                actual_words = len(response.split())

                # compress prev_summary if the current one is too long or doesn't end in punctuation
                if prev_summary is not None and (response[-1] not in ['.', '?', '!', '\"', '\''] \
                                                 or count_tokens(response) >= self.max_summary_len):
                    compressed_summary, response, chunk_trims, skipped = self.compress(response, prev_summary, chunk,
                                                                                       self.max_summary_len, word_limit,
                                                                                       num_chunks, j)
                    num_trims += chunk_trims
                    skipped_chunks += skipped
                    new_chunks[j - 1] = compressed_summary

                prev_summary = response
                new_chunks.append(response)

                if (j + 1) % 10 == 0:
                    new_data[book] = new_chunks
                    json.dump(new_data, open(self.summ_path, 'w'))

            new_data[book] = new_chunks
            json.dump(new_data, open(self.summ_path, 'w'))

    def get_summaries(self, book_path):
        if self.method == 'inc':
            self.get_incremental_summaries(book_path)
        elif self.method == 'hier':
            self.get_hierarchical_summaries(book_path)
        else:
            raise ValueError("Invalid method")


def create_retrieval_database(chunks: List[str], embedder) -> RetrievalDatabase:
    """Create and populate a retrieval database from text chunks

    Args:
        chunks: List of text chunks to index
        embedder: Function or object with an embed method to create embeddings

    Returns:
        Populated RetrievalDatabase instance
    """
    # 创建数据库实例
    db = RetrievalDatabase()

    # 批量生成嵌入
    print("Generating embeddings for chunks...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            print(f"Processing chunk {i}/{len(chunks)}")
        embedding = embedder.get_huggingface_embedding(chunk)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            print(f"Warning: Failed to generate embedding for chunk {i}")

    # 将嵌入和文本添加到数据库
    embeddings = np.array(embeddings)
    db.add(embeddings, chunks)

    print(f"Created database with {len(chunks)} chunks")
    return db


if __name__ == "__main__":
    encoding = tiktoken.get_encoding('cl100k_base')

    parser = argparse.ArgumentParser()
    parser.add_argument("--book_path", type=str, help="path to the file containing the chunked data")
    parser.add_argument("--summ_path", type=str, help="path to the json file to save the data")
    parser.add_argument("--model", type=str, help="summarizer model")
    parser.add_argument("--api", type=str, help="api to use", choices=["openai", "anthropic", "together"])
    parser.add_argument("--api_key", type=str, help="path to a txt file storing your OpenAI api key")
    parser.add_argument("--method", type=str, help="method for summarization", choices=['inc', 'hier'])
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--max_context_len", type=int, help="max content length of the model")
    parser.add_argument("--max_summary_len", type=int, default=900, help="max length of the final summary")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of all books")
    parser.add_argument("--db_path", type=str, help="path to save/load the retrieval database")
    args = parser.parse_args()

    # Create summarizer first (needed for embedding model)
    summarizer = Summarizer(
        args.model,
        args.api,
        args.api_key,
        args.summ_path,
        args.method,
        args.chunk_size,
        args.max_context_len,
        args.max_summary_len
    )

    # Load or create retrieval database
    retrieval_db = None
    if args.db_path:
        if os.path.exists(f"{args.db_path}_index"):
            print("Loading existing retrieval database...")
            retrieval_db = RetrievalDatabase.load(args.db_path)
        else:
            print("Creating new retrieval database...")
            # Load book chunks
            with open(args.book_path, 'rb') as f:
                data = pickle.load(f)

            # Flatten all chunks from all books into a single list
            all_chunks = []
            for book_chunks in data.values():
                all_chunks.extend(book_chunks)

            # Create and save database
            retrieval_db = create_retrieval_database(all_chunks, summarizer)
            retrieval_db.save(args.db_path)

        # Set the database in the summarizer
        summarizer.retrieval_database = retrieval_db

    # Add code to handle force_reprocess
    if args.force_reprocess and os.path.exists(args.summ_path):
        print(f"Removing existing summaries file: {args.summ_path}")
        os.remove(args.summ_path)

    # Run summarization
    summarizer.get_summaries(args.book_path)