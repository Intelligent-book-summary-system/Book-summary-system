# BookSummary

An intelligent book summarization system that leverages hierarchical merging, RAG (Retrieval-Augmented Generation), and fine-tuned LLMs to generate high-quality summaries of long texts.

## 📦 Repositories  
- Model: https://github.com/Intelligent-book-summary-system/Book-summary-system
- Frontend: https://github.com/Intelligent-book-summary-system/Frontend
- Backend: https://github.com/Intelligent-book-summary-system/Backend

## 📚 Overview

BookSummary is a comprehensive pipeline designed to tackle the challenge of summarizing long-form text documents such as books and lengthy articles. The system addresses LLM token limitations through an end-to-end workflow:

1. **Text Chunking**: Split long documents into manageable segments
2. **Hierarchical/Incremental Summarization**: Generate summaries using two distinct methods:
   - **Hierarchical Method**: Bottom-up merging approach that progressively combines chunk summaries
   - **Incremental Method**: Sequential processing with rolling context preservation
3. **RAG Enhancement** (Optional): Leverage multi-level retrieval using FAISS and BGE models for context-aware summarization
4. **LoRA Fine-tuning** (Optional): Apply Parameter-Efficient Fine-Tuning on LLaMA 3.1 models for improved performance
5. **Post-processing**: Clean and refine generated summaries, removing artifacts and inconsistencies
6. **Evaluation**: Assess summary quality using two complementary approaches:
   - **ROUGE Metrics**: Traditional n-gram overlap measurements (ROUGE-1, ROUGE-2, ROUGE-L)
   - **BookScore**: LLM-based evaluation that analyzes hallucinations and coherence at the sentence level

The entire workflow can be automated through orchestration tools, reducing manual operational steps by 70%.

## 🌟 Features

- **Hierarchical & Incremental Summarization**: Two methods to handle long documents effectively
- **RAG Pipeline**: Multi-level retrieval system using FAISS and BGE models for context-aware summarization
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning on LLaMA 3.1 models
- **Automated Workflow**: End-to-end orchestration from chunking to final summary generation
- **Dual Evaluation System**: ROUGE metrics for n-gram analysis and BookScore for semantic quality assessment
- **Flexible Architecture**: Support for multiple LLM APIs (OpenAI, Anthropic, Together AI)

## 📊 Performance

- ROUGE-1 scores improved from 0.52 to 0.70 on public datasets after fine-tuning
- 70% reduction in manual operational steps through automated workflows
- Intelligent context retrieval enhances information density in final summaries

## 🚀 Quick Start

### Prerequisites

```bash
# Python environment with required dependencies
conda create -n booksummary python=3.8
conda activate booksummary
pip install -r requirements.txt
```

### API Key Setup

Create an `api_key.txt` file in the project root with your API key (supports OpenAI, Anthropic, or Together AI).

## 📖 Usage Guide

### 1. Text Chunking

Split your input text into manageable chunks:

```bash
python -m booookscore.chunk \
    --chunk_size 512 \
    --input_path "./data/News/news.pkl" \
    --output_path ./data/News/news_chunked_512.pkl
```

**Parameters:**
- `--chunk_size`: Size of each text chunk (e.g., 512, 1024, 2048)
- `--input_path`: Path to your input pickle file
- `--output_path`: Where to save the chunked output

### 2. Generate Summaries

#### Method A: Hierarchical Summarization (Original)

```bash
python -m booookscore.summ \
    --book_path ./data/News/news_chunked_512.pkl \
    --summ_path ./data/News/new_hier_origin.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --api together \
    --api_key api_key.txt \
    --method hier \
    --chunk_size 512 \
    --max_context_len 8192 \
    --max_summary_len 500
```

#### Method B: Incremental Summarization

```bash
python -m booookscore.summ \
    --book_path ./data/all_books_chunked_2048.pkl \
    --summ_path ./data/summaries_inc.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --api together \
    --api_key api_key.txt \
    --method inc \
    --chunk_size 2048 \
    --max_context_len 8192
```

**Parameters:**
- `--method`: Choose `hier` for hierarchical or `inc` for incremental
- `--max_summary_len`: Maximum length of the final summary

### 3. RAG-Enhanced Summarization

#### Step 3.1: Extract Context

```bash
python ./booookscore/context.py \
    --book_path ./data/News/news_chunked_512.pkl \
    --output_path ./data/News/new_contexts.pkl \
    --api together \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --api_key api_key.txt
```

#### Step 3.2: Build Retrieval Database

```bash
python ./booookscore/build_db.py \
    --book_path ./data/HP/hp_chunked_1024.pkl \
    --db_path ./data/HP/database/retrieval_db \
    --context_path ./data/HP/book_contexts.pkl \
    --batch_size 32
```

**Parameters:**
- `--db_path`: Where to store the FAISS retrieval database
- `--context_path`: Path to extracted contexts
- `--batch_size`: Batch size for embedding generation

#### Step 3.3: Generate Summary with RAG

```bash
python -m booookscore.summ \
    --book_path ./data/HP/hp_chunked_1024.pkl \
    --summ_path ./data/HP/hp_hier_rag.json \
    --db_path ./data/HP/database/retrieval_db \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --api together \
    --api_key api_key.txt \
    --method hier \
    --chunk_size 2048 \
    --max_context_len 8192 \
    --max_summary_len 500
```

### 4. Fine-tuned Model (LoRA)

Use a fine-tuned model with LoRA adapters for improved summarization:

```bash
python -m booookscore.summ-lora \
    --book_path ./data/News/news_chunked_512.pkl \
    --summ_path ./data/News/new_hier_lora.json \
    --db_path ./data/News/database/retrieval_db \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo:./llama3-lora" \
    --api together \
    --api_key api_key.txt \
    --method hier \
    --chunk_size 2048 \
    --max_context_len 8192 \
    --max_summary_len 500
```

**Note:** The model parameter format is `base-model:lora-adapter-path`

### 5. Post-processing

Clean and refine generated summaries to remove artifacts and improve quality:

```bash
# For original hierarchical summaries
python -m booookscore.postprocess \
    --input_path ./data/News/new_hier_origin.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --api together \
    --api_key api_key.txt \
    --remove_artifacts

# For LoRA fine-tuned summaries
python -m booookscore.postprocess \
    --input_path ./data/News/new_hier_lora.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --api together \
    --api_key api_key.txt \
    --remove_artifacts
```

**Output:** Creates a cleaned version with suffix `_cleaned.json`

## 📈 Evaluation

### BookScore Evaluation

BookScore is an LLM-based evaluation metric that analyzes summaries at the sentence level to detect hallucinations and assess coherence.

#### Basic Usage (Original BookScore)

```bash
python -m booookscore.score \
    --summ_path ./data/News/new_hier_origin_cleaned.json \
    --annot_path ./data/News/annotations_origin.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --api together \
    --api_key api_key.txt
```

#### Advanced Usage (v2 with Sentence Batching)

The v2 version uses sentence batching for more efficient evaluation:

```bash
python -m booookscore.score \
    --summ_path ./data/News/new_hier_lora_cleaned.json \
    --annot_path ./data/News/annotations_lora.json \
    --model gpt-4 \
    --api openai \
    --api_key api_key.txt \
    --v2 \
    --batch_size 10
```

**Parameters:**
- `--summ_path`: Path to summaries JSON file (maps book names to summaries)
- `--annot_path`: Path to save/load model-generated annotations
- `--model`: Model to use for evaluation (e.g., gpt-4, claude-3, llama-3.1)
- `--api`: API provider (`openai`, `anthropic`, or `together`)
- `--api_key`: Path to API key file
- `--v2` (optional): Use v2 evaluation with sentence batching
- `--batch_size` (optional): Batch size for v2 evaluation (default: 10)

**Note:** The input summaries must be in JSON format mapping book/document names to their final summaries.

### ROUGE Evaluation

Evaluate summaries using traditional ROUGE metrics (n-gram overlap):

```bash
# For hierarchical summaries
python -m booookscore.rouge_eval \
    --summaries ./data/News/new_hier_lora_cleaned.json \
    --references ./data/News/reference.json \
    --type hier \
    --metrics rouge1 rouge2 rougeL \
    --output-dir ./data/News/rouge_results_lora \
    --verbose

# For RAG-enhanced summaries
python -m booookscore.rouge_eval \
    --summaries ./data/Alice/summaries_hier_rag_cleaned.json \
    --references ./data/Alice/summaries_reference.json \
    --type hier \
    --metrics rouge1 rouge2 rougeL \
    --output-dir ./data/Alice/rouge_results_rag \
    --verbose
```

**Parameters:**
- `--summaries`: Path to generated summaries
- `--references`: Path to reference summaries
- `--type`: Summarization method (`hier` or `inc`)
- `--metrics`: Which ROUGE metrics to compute
- `--output-dir`: Directory to save results
- `--verbose`: Show detailed output

## 🏗️ Architecture

The system addresses LLM token limitations through:

1. **Hierarchical Merging**: Bottom-up approach to progressively summarize chunks
2. **Incremental Updates**: Sequential processing with context preservation
3. **RAG Pipeline**: Dynamic similarity-based retrieval using FAISS to enhance context
4. **PEFT Fine-tuning**: LoRA adapters for efficient model optimization without full retraining

## 📁 Project Structure

```
data/
├── Alice/         # Alice in Wonderland data
├── HP/            # Harry Potter book data
└──

booookscore/
├── chunk.py       # Text chunking module
├── context.py     # Context extraction
├── build_db.py    # FAISS database creation
├── summ.py        # Summarization (base)
├── summ-lora.py   # LoRA fine-tuned summarization
├── postprocess.py # Summary post-processing
├── score.py       # BookScore evaluation
└── rouge_eval.py  # ROUGE metrics evaluation
```

## 🔗 Repository

[GitHub Organization](https://github.com/orgs/Intelligent-book-summary-system/repositories)

## 💡 Tips

- Start with smaller chunk sizes (512-1024) for initial experiments
- Use RAG enhancement for complex documents with intricate relationships
- Post-processing is crucial for removing artifacts from hierarchical merging
- BookScore provides more nuanced evaluation than ROUGE for semantic quality
- The v2 BookScore with batching is more efficient for large-scale evaluation

## 📝 Notes

- Ensure your input files are in pickle (`.pkl`) format
- Output summaries are saved in JSON format
- The system supports integration with Spring Boot backend for automated workflow orchestration
- Compatible with multiple APIs: Together AI, OpenAI, and Anthropic
- Reference summaries are required for ROUGE evaluation
  
