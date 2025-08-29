import os
from lightrag import LightRAG, QueryParam

# #Check to see the correct module is getting imported in case of multiple versions of lightrag in your system
# import lightrag
# assert "GraphloomWithoutRC" in os.path.abspath(lightrag.__file__), (
#     f"ERROR: Imported lightrag from unexpected path: {lightrag.__file__}"
# )
from lightrag.llm.openai import openai_complete_if_cache, openai_embed, gpt_4o_mini_complete
from lightrag.utils import EmbeddingFunc
import numpy as np
import asyncio
import nest_asyncio
from pathlib import Path
import argparse
import time
import socket
import json

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

"""
To run the script:
python evaluation_result_generator.py \
    --rag_type lightrag \
    --rag_mode index_creation \
    --graphloom_summary False \
    --reset_retrieval_count False \
    --dir_identifier eval1 \
    --create_new_dir True

python evaluation_result_generator.py \
    --rag_type lightrag \
    --rag_mode query \
    --graphloom_summary False \
    --reset_retrieval_count False \
    --dir_identifier eval1 \
    --create_new_dir False

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode index_creation \
    --graphloom_summary True \
    --reset_retrieval_count True \
    --dir_identifier eval1 \
    --create_new_dir True

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary True \
    --reset_retrieval_count True \
    --dir_identifier eval1 \
    --create_new_dir False \
    --response_ouput_file "dickens_evaluation_output_graphloom_base.json"

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary True \
    --reset_retrieval_count False \
    --dir_identifier eval1 \
    --create_new_dir False \
    --response_ouput_file "dickens_evaluation_output_graphloom_summary_on_rc_off.json"

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary False \
    --reset_retrieval_count True \
    --dir_identifier eval1 \
    --create_new_dir False \
    --response_ouput_file "dickens_evaluation_output_graphloom_summary_off_rc_on.json"

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary False \
    --reset_retrieval_count False \
    --dir_identifier eval1 \
    --create_new_dir False \
    --response_ouput_file "dickens_evaluation_output_graphloom_summary_off_rc_off.json"

python evaluation_result_generator.py \
    --rag_type lightrag \
    --rag_mode index_creation \
    --graphloom_summary False \
    --reset_retrieval_count False \
    --dir_identifier agriculture_eval1 \
    --create_new_dir True

python evaluation_result_generator.py \
    --rag_type lightrag \
    --rag_mode index_creation \
    --graphloom_summary False \
    --reset_retrieval_count False \
    --dir_identifier agriculture_eval1 \
    --create_new_dir False    

python evaluation_result_generator.py \
    --rag_type lightrag \
    --rag_mode query \
    --graphloom_summary False \
    --reset_retrieval_count False \
    --dir_identifier agriculture_eval1_base \
    --create_new_dir False \
    --query_input_file "../datasets/questions/agriculture_questions.json" \
    --response_ouput_file "agriculture_evaluation_output_lightrag_base.json" 

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode index_creation \
    --graphloom_summary True \
    --reset_retrieval_count True \
    --dir_identifier agriculture_eval1 \
    --create_new_dir True    

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode index_creation \
    --graphloom_summary True \
    --reset_retrieval_count True \
    --dir_identifier agriculture_eval1 \
    --create_new_dir False 

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary True \
    --reset_retrieval_count True \
    --dir_identifier agriculture_eval1_summary_on_rc_exists \
    --create_new_dir False \
    --query_input_file "../datasets/questions/agriculture_questions.json" \
    --response_ouput_file "agriculture_evaluation_output_graphloom_summary_on_rc_exists.json"    

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary False \
    --reset_retrieval_count True \
    --dir_identifier agriculture_eval1_summary_off_rc_exists \
    --create_new_dir False \
    --query_input_file "../datasets/questions/agriculture_questions.json" \
    --response_ouput_file "agriculture_evaluation_output_graphloom_summary_off_rc_exists.json"       

////////////////////////////////////////////////////////////////

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode index_creation \
    --graphloom_summary True \
    --dir_identifier eval1 \
    --create_new_dir True

export PYTHONPATH=/mnt/ssd1/aparna/GraphLoom/GraphloomWithoutRC/LightRAG
python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode index_creation \
    --graphloom_summary True \
    --dir_identifier dickens_eval2 \
    --create_new_dir True    

export PYTHONPATH=/mnt/ssd1/aparna/GraphLoom/GraphloomWithoutRC/LightRAG
python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary True \
    --dir_identifier dickens_eval2 \
    --create_new_dir False \
    --response_ouput_file "dickens_evaluation_output_graphloom_summary_on_rc_dne.json"    

#########Copied over the rag directory so that query can be done parallely and the graphloom.log does not get corrupted########
export PYTHONPATH=/mnt/ssd1/aparna/GraphLoom/GraphloomWithoutRC/LightRAG
python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary False \
    --dir_identifier dickens_eval2_query_summ_off \
    --create_new_dir False \
    --response_ouput_file "dickens_evaluation_output_graphloom_summary_off_rc_dne.json"    

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode index_creation \
    --graphloom_summary True \
    --dir_identifier agriculture_eval1 \
    --create_new_dir True

python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode index_creation \
    --graphloom_summary True \
    --dir_identifier agriculture_eval1 \
    --create_new_dir False

export PYTHONPATH=/mnt/ssd1/aparna/GraphLoom/GraphloomWithoutRC/LightRAG
python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode index_creation \
    --graphloom_summary True \
    --dir_identifier agriculture_eval2 \
    --create_new_dir True



export PYTHONPATH=/mnt/ssd1/aparna/GraphLoom/GraphloomWithoutRC/LightRAG
python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode index_creation \
    --graphloom_summary True \
    --dir_identifier agriculture_eval2 \
    --create_new_dir False    

export PYTHONPATH=/mnt/ssd1/aparna/GraphLoom/GraphloomWithoutRC/LightRAG
python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary True \
    --dir_identifier agriculture_eval2_summary_on_rc_dne \
    --create_new_dir False \
    --query_input_file "../datasets/questions/agriculture_questions.json" \
    --response_ouput_file "agriculture_evaluation_output_graphloom_summary_on_rc_dne.json"    

#########Copied over the rag directory so that query can be done parallely and the graphloom.log does not get corrupted########
export PYTHONPATH=/mnt/ssd1/aparna/GraphLoom/GraphloomWithoutRC/LightRAG
python evaluation_result_generator.py \
    --rag_type graphloom \
    --rag_mode query \
    --graphloom_summary False \
    --dir_identifier agriculture_eval2_summary_off_rc_dne \
    --create_new_dir False \
    --query_input_file "../datasets/questions/agriculture_questions.json" \
    --response_ouput_file "agriculture_evaluation_output_graphloom_summary_off_rc_dne.json"   

    
"""

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run RAG Script")
parser.add_argument("--rag_type", choices=["lightrag", "graphloom"], default="graphloom", help="Type of RAG to use")
parser.add_argument("--rag_mode", choices=["index_creation", "query", "index_creation+query"], default="index_creation", help="RAG mode")
parser.add_argument("--graphloom_summary", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable graphloom summary (True/False)")
parser.add_argument("--reset_retrieval_count", type=lambda x: (str(x).lower() == 'true'), default=True, help="Reset retrieval count (True/False)")
parser.add_argument("--dir_identifier", default="evaluation", help="Suffix for the working directory")
parser.add_argument("--create_new_dir", type=lambda x: (str(x).lower() == 'true'), default=True, help="Create new directory if it does not exist (True/False)")
parser.add_argument("--query_input_file", default="dickens_evaluation_input.json", help="JSON file containing the queries")
parser.add_argument("--response_ouput_file", default="dickens_evaluation_output.json", help="JSON file name for stroing the response of the queries")
args = parser.parse_args()

model = "llama" 
rag_type = args.rag_type
rag_mode = args.rag_mode

# LightRAG params
if rag_type == "graphloom":
    graphloom_enabled = True
    graph_storage="NetworkXHeteroStorage"
else:
    graphloom_enabled = False
    graph_storage="NetworkXStorage"

graphloom_summary = args.graphloom_summary
reset_retrieval_count = args.reset_retrieval_count

dir_identifier = args.dir_identifier
create_new_dir = args.create_new_dir

BASE_DIR = Path(__file__).resolve().parent
WORKING_DIR = BASE_DIR.parent / f"ragsRunWorkingDir/{rag_type}-{model}-{dir_identifier}"

if create_new_dir:
    if not os.path.exists(WORKING_DIR):
        print(f"Creating new working directory: {WORKING_DIR}")
        os.makedirs(WORKING_DIR, exist_ok=True) # create parent directories also if it doesn't exist
    else:
        import shutil
        print(f"Directory {WORKING_DIR} already exists. Deleting and recreating it.")
        shutil.rmtree(WORKING_DIR)
        os.makedirs(WORKING_DIR)
else:
    if not os.path.exists(WORKING_DIR):
        print(f"Directory {WORKING_DIR} does not exist and will not be created (create_new_dir=False).")
        raise FileNotFoundError(f"{WORKING_DIR} does not exist and create_new_dir is False")

def is_port_open(host, port):
    """Check if a TCP port is open on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # 1 second timeout
        result = sock.connect_ex((host, port))
        return result == 0

# List of candidate ports (update as needed)
candidate_ports = [8004, #proxy
#8000 #1st model
]
host = "0.0.0.0"  # or "localhost"

open_port = None
for port in candidate_ports:
    if is_port_open(host, port):
        open_port = port
        break

if open_port is None:
    raise RuntimeError("No open LLM server ports found!")
else:
    base_url = f"http://{host}:{open_port}/v1"

print("RAG params:")
print(f"{graph_storage=}")
print(f"{rag_type=}")
print(f"{rag_mode=}")
print(f"{graphloom_enabled=}")
print(f"{graphloom_summary=}")
print(f"{reset_retrieval_count=}")
print(f"{dir_identifier=}")
print(f"{create_new_dir=}")
print(f"{WORKING_DIR=}")
print(f"{base_url=}")
print(f"{args.query_input_file=}")
print(f"{args.response_ouput_file=}")

# LLM model function for Llama
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
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

# Initialize RAG instance
rag = LightRAG(
    working_dir=str(WORKING_DIR),
    llm_model_func=llm_model_func, 
    embedding_func=EmbeddingFunc(
        embedding_dim=asyncio.run(get_embedding_dim()),
        max_token_size=8192,
        func=embedding_func,
    ), 
    graphloom=graphloom_enabled,
    graphloom_summary=graphloom_summary,
    reset_retrieval_count=reset_retrieval_count,
    graph_storage=graph_storage,
    log_level = 10, # DEBUG
    log_file_path = str(WORKING_DIR / "graphloom.log")
)

start_time = time.time()
if "index_creation" in rag_mode:
    data_path = "../datasets/unique_contexts/agriculture_unique_contexts.json" # "../datasets/book.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        rag.insert(f.read())
        
if "query" in rag_mode:
    with open(args.query_input_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = {}

    cnt = 1
    for user in queries["users"]:
        role = user["role"]
        results[role] = {}
        for task in user["tasks"]:
            task_name = task["task"]
            results[role][task_name] = {}
            for question in task["questions"]:
                print(f"{cnt=}")
                cnt+=1
                res = rag.query(question, param=QueryParam(mode="hybrid"))
                results[role][task_name][question] = res

    output_file = args.response_ouput_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


end_time = time.time()
print(f"Time taken for run: {end_time - start_time:.2f} seconds")