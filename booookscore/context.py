import os
import pickle
import argparse
from tqdm import tqdm
import time
from typing import Dict, List


class BookContextGenerator:
    def __init__(self, api_client, output_path=None):
        """
        初始化书籍背景信息生成器

        Args:
            api_client: API客户端，用于调用大模型API
            output_path: 生成的背景信息输出路径
        """
        self.api_client = api_client
        self.output_path = output_path or "data/book_contexts.pkl"
        self.context_templates = {
            "book_info": self._load_template("prompts/context/book_info.txt"),
            "character_info": self._load_template("prompts/context/character_info.txt"),
            "setting_info": self._load_template("prompts/context/setting_info.txt"),
            "themes_info": self._load_template("prompts/context/themes_info.txt"),
            "historical_context": self._load_template("prompts/context/historical_context.txt")
        }

    def _load_template(self, path):
        """加载提示模板"""
        try:
            with open(path, "r") as f:
                return f.read()
        except FileNotFoundError:
            # 如果模板文件不存在，返回默认模板
            if "book_info" in path:
                return "Provide detailed information about the book '{book_title}'. Include author, publication date, genre, and a brief synopsis."
            elif "character_info" in path:
                return "List and describe the main characters in the book '{book_title}'. Include their backgrounds, motivations, and roles in the story."
            elif "setting_info" in path:
                return "Describe the setting of the book '{book_title}'. Include time period, locations, and any important world-building elements."
            elif "themes_info" in path:
                return "Analyze the main themes and motifs in the book '{book_title}'."
            elif "historical_context" in path:
                return "Provide historical or cultural context relevant to understanding the book '{book_title}'."
            else:
                return "Provide information about {aspect} of the book '{book_title}'."

    def generate_book_contexts(self, book_titles: List[str], max_retries=3) -> Dict[str, Dict[str, str]]:
        """
        为一组书籍生成背景信息

        Args:
            book_titles: 书籍标题列表
            max_retries: 调用API失败时的最大重试次数

        Returns:
            Dict[str, Dict[str, str]]: 书籍标题到各类背景信息的映射
        """
        contexts = {}

        # 如果已有保存的上下文数据，先加载
        if os.path.exists(self.output_path):
            with open(self.output_path, 'rb') as f:
                contexts = pickle.load(f)
                print(f"Loaded {len(contexts)} existing book contexts")

        for book_title in tqdm(book_titles, desc="Generating book contexts"):
            if book_title in contexts:
                print(f"Context for '{book_title}' already exists. Skipping...")
                continue

            book_context = {}

            # 为每种上下文类型生成信息
            for context_type, template in self.context_templates.items():
                prompt = template.format(book_title=book_title)

                # 尝试调用API，失败后重试
                for attempt in range(max_retries):
                    try:
                        response = self.api_client.obtain_response(
                            prompt=prompt,
                            max_tokens=1000,
                            temperature=0.3
                        )
                        book_context[context_type] = response
                        break
                    except Exception as e:
                        print(f"Error calling API for '{book_title}' ({context_type}): {e}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # 指数退避
                            print(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(f"Failed after {max_retries} attempts")
                            book_context[context_type] = "Failed to generate information."

            # 保存当前书籍的上下文信息
            contexts[book_title] = book_context

            # 每处理完一本书就保存一次，以防中断
            with open(self.output_path, 'wb') as f:
                pickle.dump(contexts, f)

        return contexts

    def generate_chunk_specific_contexts(self, book_chunks: Dict[str, List[str]], max_retries=3) -> Dict[
        str, List[str]]:
        """
        为书籍的每个文本块生成特定的上下文信息

        Args:
            book_chunks: 书名到文本块列表的映射
            max_retries: 调用API失败时的最大重试次数

        Returns:
            Dict[str, List[str]]: 书名到上下文块列表的映射
        """
        chunk_contexts = {}
        chunk_context_path = os.path.join(os.path.dirname(self.output_path), "chunk_contexts.pkl")

        # 如果已有保存的块上下文数据，先加载
        if os.path.exists(chunk_context_path):
            with open(chunk_context_path, 'rb') as f:
                chunk_contexts = pickle.load(f)
                print(f"Loaded chunk contexts for {len(chunk_contexts)} books")

        # 加载书籍整体上下文
        book_contexts = {}
        if os.path.exists(self.output_path):
            with open(self.output_path, 'rb') as f:
                book_contexts = pickle.load(f)

        for book_title, chunks in tqdm(book_chunks.items(), desc="Generating chunk contexts"):
            if book_title in chunk_contexts and len(chunk_contexts[book_title]) == len(chunks):
                print(f"Chunk contexts for '{book_title}' already complete. Skipping...")
                continue

            # 如果书籍没有整体上下文，先生成
            if book_title not in book_contexts:
                self.generate_book_contexts([book_title], max_retries)
                with open(self.output_path, 'rb') as f:
                    book_contexts = pickle.load(f)

            book_info = " ".join([info for info in book_contexts[book_title].values()])

            # 为每个块生成上下文
            chunk_context_list = chunk_contexts.get(book_title, [])
            start_idx = len(chunk_context_list)

            for i, chunk in enumerate(chunks[start_idx:], start=start_idx):
                # 创建针对特定文本块的提示
                chunk_prompt = f"""
                Below is the book information:
                {book_info}

                Below is a chunk from the book:
                {chunk[:500]}...

                Based on the book information and this chunk, provide relevant contextual information that would help 
                understand this part of the story better. Include character backgrounds, relevant plot points, 
                setting details, or thematic elements that relate to this specific section.
                """

                # 尝试调用API，失败后重试
                context = None
                for attempt in range(max_retries):
                    try:
                        context = self.api_client.obtain_response(
                            prompt=chunk_prompt,
                            max_tokens=800,
                            temperature=0.3
                        )
                        break
                    except Exception as e:
                        print(f"Error calling API for '{book_title}' (chunk {i}): {e}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(f"Failed after {max_retries} attempts")
                            context = "Failed to generate context for this chunk."

                chunk_context_list.append(context)

                # 每处理10个块保存一次
                if (len(chunk_context_list) % 10 == 0) or (i == len(chunks) - 1):
                    chunk_contexts[book_title] = chunk_context_list
                    with open(chunk_context_path, 'wb') as f:
                        pickle.dump(chunk_contexts, f)

            chunk_contexts[book_title] = chunk_context_list

        return chunk_contexts


def main():
    parser = argparse.ArgumentParser(description="Generate book contexts using LLM API")
    parser.add_argument("--book_path", type=str, required=True, help="Path to the pickled book chunks file")
    parser.add_argument("--output_path", type=str, default="data/book_contexts.pkl", help="Path to save the contexts")
    parser.add_argument("--api", type=str, required=True, choices=["openai", "anthropic", "together"],
                        help="API to use")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--api_key", type=str, required=True, help="API key or path to key file")

    args = parser.parse_args()

    # 导入APIClient，与summpy.py中使用相同的客户端
    from utils import APIClient

    # 创建API客户端
    client = APIClient(args.api, args.api_key, args.model)

    # 加载书籍数据
    with open(args.book_path, 'rb') as f:
        book_chunks = pickle.load(f)

    # 初始化上下文生成器
    generator = BookContextGenerator(client, args.output_path)

    # 生成书籍整体上下文
    book_titles = list(book_chunks.keys())
    book_contexts = generator.generate_book_contexts(book_titles)

    # 生成每个块的特定上下文
    chunk_contexts = generator.generate_chunk_specific_contexts(book_chunks)

    print(
        f"Generated contexts for {len(book_contexts)} books and {sum(len(chunks) for chunks in chunk_contexts.values())} chunks")


if __name__ == "__main__":
    main()