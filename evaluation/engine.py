from collections.abc import AsyncIterator
from pathlib import Path
from typing import Callable
from lightrag import QueryParam
from lightrag import LightRAG
from lightrag.llm.openai import openai_embed, litellm_complete
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
        self.config = config
        self.graphloom_enabled = config.gl_enabled
        self.graphloom_summary_enabled = config.gl_summ_enabled

        self.working_dir: Path = config.add_path(identifier)

        self.engine = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self.model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=asyncio.run(self.get_embedding_dim()),
                max_token_size=8192,
                func=self.embedding_func,
            ),
            graphloom=self.graphloom_enabled,
            graphloom_summary=self.graphloom_summary_enabled,
            graph_storage=config.graph_storage
            if hasattr(config, "graph_storage")
            else "NetworkXHeteroStorage",
            log_level=config.log_level if hasattr(config, "log_level") else 10,
            log_file_path=f"{self.working_dir}"
            + (
                config.log_file_path
                if hasattr(config, "log_file_path")
                else "/graphloom.log"
            ),
        )

    async def insert(self, document: str):
        await self.engine.ainsert(document)

    async def query(self, query: str) -> str | AsyncIterator[str]:
        return await self.engine.aquery(query, param=QueryParam(mode="hybrid"))

    async def model_func(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> Callable[..., str]:
        return await litellm_complete(
            prompt,
            model_name=self.config.model_config.inference_model,
            base_url=self.config.proxy.base_url,
            api_key=self.config.proxy.api_key,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
            **kwargs,
        )

    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        return await openai_embed(
            texts=texts,
            model=self.config.model_config.embedding_model,
            base_url=self.config.proxy.base_url,
            api_key=self.config.proxy.api_key,
        )

    async def get_embedding_dim(self):
        test_text = ["This is a test sentence."]
        embedding = await self.embedding_func(test_text)
        return embedding.shape[1]
