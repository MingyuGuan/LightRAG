import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed, gpt_4o_mini_complete
from lightrag.utils import EmbeddingFunc
import numpy as np
import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

model = "gpt-4o-mini" # "llama" # "gpt-4o-mini"
rag_type = "graphloom" # "lightrag"
WORKING_DIR = f"../rags/dickens-{rag_type}-{model}"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

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
        # max_tokens=4096,
        **kwargs,
    )

# Embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts=texts,
        model="intfloat/e5-mistral-7b-instruct",
        base_url="http://0.0.0.0:8003/v1",
        api_key="blahblah",
    )

async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    print(f"{embedding_dim=}")
    return embedding_dim


# Initialize RAG instance
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete, #llm_model_func
    embedding_func=EmbeddingFunc(
        embedding_dim=asyncio.run(get_embedding_dim()),
        max_token_size=8192,
        func=embedding_func,
    ), #openai_embed,
    graphloom=True,
    graph_storage="NetworkXHeteroStorage",
    log_level = 10, # DEBUG
    log_file_path = "/mnt/ssd1/mingyu/LightRAG/dev_scripts/graphloom.log"
)

# with open("../datasets/book.txt", "r", encoding="utf-8") as f:
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
