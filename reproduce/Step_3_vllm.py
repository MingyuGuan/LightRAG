import os
import re
import json
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop
import numpy as np

import asyncio

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

def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")

    queries = re.findall(r"- Question \d+: (.+)", data)

    return queries


async def process_query(query_text, rag_instance, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()

    with open(output_file, "a", encoding="utf-8") as result_file, open(
        error_file, "a", encoding="utf-8"
    ) as err_file:
        result_file.write("[\n")
        first_entry = True

        for query_text in queries:
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )

            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")


if __name__ == "__main__":
    # cls = "agriculture" # "mix"
    # mode = "hybrid"
    # WORKING_DIR = f"../rag/{cls}"

    # rag = LightRAG(working_dir=WORKING_DIR)
    # rag = LightRAG(
    #     working_dir=WORKING_DIR,
    #     llm_model_func=llm_model_func,
    #     embedding_func=EmbeddingFunc(
    #         embedding_dim=4096, max_token_size=8192, func=embedding_func
    #     ),
    # )

    cls = "agriculture" # "mix"
    model="llama" # "gpt-4o-mini"
    mode = "hybrid"
    WORKING_DIR = f"../rags/{cls}-{model}"

    if not os.path.exists(WORKING_DIR):
        raise FileNotFoundError(f"Working directory {WORKING_DIR} does not exist")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=asyncio.run(get_embedding_dim()),
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    query_param = QueryParam(mode=mode)

    base_dir = "../datasets/questions"
    queries = extract_queries(f"{base_dir}/{cls}_questions.txt")
    run_queries_and_save_to_json(
        queries, rag, query_param, f"{base_dir}/result.json", f"{base_dir}/errors.json"
    )
