import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed, gpt_4o_mini_complete
from lightrag.utils import EmbeddingFunc
import numpy as np
import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

DEFAULT_RAG_DIR = "index_default"

# # Configure working directory
# WORKING_DIR = os.environ.get("RAG_DIR", f"{DEFAULT_RAG_DIR}")
# print(f"WORKING_DIR: {WORKING_DIR}")
# LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
# print(f"LLM_MODEL: {LLM_MODEL}")
# EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
# print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
# EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
# print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")
# BASE_URL = os.environ.get("BASE_URL", "https://api.openai.com/v1")
# print(f"BASE_URL: {BASE_URL}")
# API_KEY = os.environ.get("API_KEY", "xxxxxxxx")
# print(f"API_KEY: {API_KEY}")

if not os.path.exists("./dickens"):
    os.mkdir("./dickens")


# LLM model function
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="http://0.0.0.0:8000/v1",
        api_key="blahblah",
        **kwargs,
    )


# Embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts=texts,
        model="intfloat/e5-mistral-7b-instruct",
        base_url="http://0.0.0.0:8001/v1",
        api_key="blahblah",
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    print(f"{embedding_dim=}")
    return embedding_dim


# Initialize RAG instance
# rag = LightRAG(
#     working_dir="./dickens-llama",
#     llm_model_func=llm_model_func,
#     embedding_func=EmbeddingFunc(
#         embedding_dim=asyncio.run(get_embedding_dim()),
#         max_token_size=8192,
#         func=embedding_func,
#     ),
# )

rag = LightRAG(
    working_dir="./dickens-gpt-4o-mini",
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete
)

# with open("./book.txt", "r", encoding="utf-8") as f:
#     rag.insert(f.read())

breakpoint()
# Perform naive search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Perform local search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Perform global search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Perform hybrid search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))

# Perform mix search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="mix")))
