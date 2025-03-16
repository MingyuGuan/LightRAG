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