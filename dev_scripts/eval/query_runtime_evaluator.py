import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
import asyncio
import nest_asyncio
from pathlib import Path
import cProfile
import pstats
import io
import shutil

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

model = "llama" 
rag_type = "graphloom" # "lightrag"
identifier = "dickens" #prefix of thew working directory

BASE_DIR = Path(__file__).resolve().parent
WORKING_DIR = BASE_DIR.parent / f"ragsRunWorkingDir/{identifier}-{rag_type}-{model}" # or substitute the directory for which index has already been created

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True) #create parent directories also if it doesn't exist

print(f"Working directory: {WORKING_DIR}")

# LLM model function for Llama
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

# Embedding function for Llama
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

if rag_type == "graphloom":
    graphloom_enabled = True
else:
    graphloom_enabled = False

# Initialize RAG instance
rag = LightRAG(
    working_dir=str(WORKING_DIR),
    llm_model_func=llm_model_func, 
    embedding_func=EmbeddingFunc(
        embedding_dim=asyncio.run(get_embedding_dim()),
        max_token_size=8192,
        func=embedding_func,
    ), #openai_embed,
    graphloom=graphloom_enabled,
    graphloom_summary=True,
    reset_retrieval_count=False,
    graph_storage="NetworkXHeteroStorage",
    log_level = 10, # DEBUG
    log_file_path = str(WORKING_DIR / "graphloom.log")
)

# Create profiles directory if it doesn't exist
profiles_dir = Path("profiles")
profiles_dir.mkdir(exist_ok=True)
# Create a table to store profiling results
profiling_results = []

def get_high_precision_stats(profiler, function_filter):
    stats = pstats.Stats(profiler)
    res = {
        "ncalls": 0,
        "tottime": 0,
        "cumtime": 0,
        "funcname": "",
        "lineno": 0,
            }
    for func, stat in stats.stats.items():
        filename, lineno, funcname = func
        #if function_filter in funcname or function_filter in filename:
        if funcname in function_filter:
            ncalls = stat[0]
            tottime = stat[2]
            cumtime = stat[3]
            
            res = {
                "ncalls": ncalls,
                "tottime": tottime,
                "cumtime": cumtime,
                "funcname": funcname,
                "lineno": lineno,
            }
            break
    return res

def profile_query(query_func, query_text, mode):
    """Profile a query and save the results"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute the query
    result = query_func(query_text, param=QueryParam(mode=mode))
    
    profiler.disable()
    
    # Save profile stats
    global graphloom_enabled
    if graphloom_enabled:
        stats_file = profiles_dir / f"profile_graphloom_dickens_{mode}.prof"
    else:
        stats_file = profiles_dir / f"profile_dickens_{mode}.prof"
    profiler.dump_stats(str(stats_file))

    stats = get_high_precision_stats(profiler, ["gl_kg_query",  #local, global, hybrid => graphloom
        "kg_query", #local, global, hybrid => graphloom
        "naive_query", #naive
        "mix_kg_vector_query"] #mix
        )
    stats["mode"] = mode
    profiling_results.append(stats)
    
    return result

# Perform naive search with profiling
print("=== NAIVE SEARCH ===")
result = profile_query(
    rag.query, 
    "What are the top themes in this story?", 
    "naive"
)
print(result)

# Perform local search with profiling
print("=== LOCAL SEARCH ===")
result = profile_query(
    rag.query, 
    "What are the top themes in this story?", 
    "local"
)
print(result)

# Perform global search with profiling
print("=== GLOBAL SEARCH ===")
result = profile_query(
    rag.query, 
    "What are the top themes in this story?", 
    "global"  
)
print(result)

# Perform hybrid search with profiling
print("=== HYBRID SEARCH ===")
result = profile_query(
    rag.query, 
    "What are the top themes in this story?", 
    "hybrid"
)
print(result)

# Perform mix search with profiling
print("=== MIX SEARCH ===")
result = profile_query(
    rag.query, 
    "What are the top themes in this story?", 
    "mix"
)
print(result)

# Print all results at the end
print("\n" + "="*80)
print("FINAL PROFILING RESULTS SUMMARY")
print("="*80)
print()

# Print table header
print(f"{'MODE':<10} {'NCALLS':>7} {'TOTTIME':>15} {'CUMTIME':>15} {'FUNC_NAME':<15} {'LINENO':>7}")
print("-" * 80)

# Print each row with high precision for floats
for entry in profiling_results:
    print(f"{entry['mode']:<10} "
          f"{entry['ncalls']:>7} "
          f"{entry['tottime']:>15.8f} "
          f"{entry['cumtime']:>15.8f} "
          f"{entry['funcname']:<15} "
          f"{entry['lineno']:>7}")