import re
import json
from lightrag import LightRAG, QueryParam
from lightrag.utils import always_get_an_event_loop, EmbeddingFunc

from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

import asyncio
import numpy as np

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

if __name__ == "__main__":
    cls = "agriculture"
    index_model="llama" # "gpt-4o-mini"
    query_model="gpt-4o-mini"
    mode = "hybrid"
    WORKING_DIR = f"../rags/{cls}-{index_model}"

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=asyncio.run(get_embedding_dim()),
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    query_param = QueryParam(mode=mode)

    queries = extract_queries(f"../datasets/questions/{cls}_questions_small.txt") # {cls}_questions.txt
    run_queries_and_save_to_json(
        queries, rag, query_param, f"{cls}_{index_model}_query_{query_model}_result.json", f"{cls}_{index_model}_query_{query_model}_errors.json"
    )
