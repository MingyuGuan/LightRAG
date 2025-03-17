### Enable GraphLoom
```
rag = LightRAG(
    ...,
    graphloom=True,
    graph_storage="NetworkXHeteroStorage" # currently GraphLoom only supports NetworkX
)
```

### Serve local models using vLLM

```bash
# start inference model(s
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000 --disable-log-requests
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8001 --disable-log-requests
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=2 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8002 --disable-log-requests

# start litellm proxy if multiple instances are running
litellm --config litellm_config.yaml

# start embedding model
# do not use V1 for embedding model
CUDA_VISIBLE_DEVICES=3 vllm serve intfloat/e5-mistral-7b-instruct --port 8003 --disable-log-requests
```
### Some Tips
- Do not reinsert the same document multiple times - it treats them as multiple documents and may mess things up
- To regenerate the graph index and reuse cached responses (to save cost), in the rag dir, delete all files EXCEPT *kv_store_llm_response_cache.json*.
- By default, it always use cached llm responses (*kv_store_llm_response_cache.json*). To disable it, you can either add *enable_llm_cache=False* when initialize the rag, or simply delete certain response by searching the keyword in *kv_store_llm_response_cache.json*, e.g., local, global, extract, etc., so that you can reuse others but regenerate specific one.

### Todo List
 - Local Search:
    - In operate.py,  _get_local_data should also direct match relationships and gather entities related to those relationships.
    - Should local search also return themes? i.e., getting themes from entities by the-ent edges.
- In-context Query Keyword Exaction:
    - Instead of extracting keywords solely based on the query, we can also provide a summary of the RAG to the LLM for more accurate extraction.
    - For example, the global keywords extracted from the query "What are the top themes in this story?" are "Themes", "Story analysis", "Literary elements", which naturally results in a poor match to the actually content/information in the RAG.
    - There are several ways to generate/maintain the summary of RAG. For example, in the kg_extraction prompt, we can also ask llm to briefly summarize the current chunk, and gradually adding the chunk summary to augment the summary of RAG and save it. Then during the query time, we can simply use this summary of RAG for query keyword extraction, and update it whenever the RAG gets new documents inserted.
- Adaptive RAG:
    - will update..