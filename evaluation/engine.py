
from collections.abc import AsyncIterator
from pathlib import Path
from lightrag import QueryParam
from lightrag import LightRAG
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete
from lightrag.utils import EmbeddingFunc
from evaluation.config import GraphLoomConfig
import numpy as np
import asyncio
import nest_asyncio

nest_asyncio.apply()

"""
RAG Engine
"""
class RAGEngine:
    
    def __init__(self, config: GraphLoomConfig, identifier: str):
        self.graphloom_enabled = config.gl_enabled
        self.graphloom_summary_enabled = config.gl_summ_enabled
        
        self.working_dir: Path = config.add_path(identifier)
        
        self.engine = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=EmbeddingFunc(
                embedding_dim=asyncio.run(self.get_embedding_dim()),
                max_token_size=8192,
                func=self.embedding_func,
            ),
            graphloom=self.graphloom_enabled,
            graphloom_summary=self.graphloom_summary_enabled,
            graph_storage=config.graph_storage if hasattr(config, 'graph_storage') else "NetworkXHeteroStorage",
            log_level=config.log_level if hasattr(config, 'log_level') else 10,
            log_file_path=f"{self.working_dir}" + (config.log_file_path if hasattr(config, 'log_file_path') else "/graphloom.log")
        )

    async def insert(self, document: str):
        await self.engine.ainsert(document)

    async def query(self, query: str) -> str | AsyncIterator[str]:
        return await self.engine.aquery(query, param=QueryParam(mode="hybrid"))

    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        return await openai_embed(texts=texts)
    
    async def get_embedding_dim(self):
        test_text = ["This is a test sentence."]
        embedding = await self.embedding_func(test_text)
        return embedding.shape[1]