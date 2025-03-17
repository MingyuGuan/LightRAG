import os
from dataclasses import dataclass
from typing import Any, final

import numpy as np


from lightrag.types import KnowledgeGraph
from lightrag.utils import (
    logger,
)

from lightrag.base import (
    BaseGraphStorage,
)
import pipmaster as pm

if not pm.is_installed("networkx"):
    pm.install("networkx")

if not pm.is_installed("graspologic"):
    pm.install("graspologic")

import networkx as nx
from graspologic import embed

########################################################
# Node types: 
# - ent: entity
# - the: theme
# Edge types: 
# - rel: entity-entity edge
# - the-ent: theme-entity edge
# - the-hrc: theme-theme edge
# Assumptions:
# - Node id is unique. Entities and themes cannot have the same id. (can be removed by adding prefix to node id to avoid collision, but it's not implemented for simplicity)
########################################################

@final
@dataclass
class NetworkXHeteroStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXHeteroStorage.load_nx_graph(self._graphml_xml_file)
        logger.info(f"Preloaded graph: {preloaded_graph}")
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self) -> None:
        NetworkXHeteroStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str, node_type: str) -> bool:
        return self._graph.has_node(node_id) and self._graph.nodes[node_id]["type"] == node_type

    async def has_edge(self, source_node_id: str, target_node_id: str, edge_type: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id) and self._graph.edges[source_node_id, target_node_id]["type"] == edge_type
    
    async def get_node(self, node_id: str, node_type: str) -> dict[str, str] | None:
        exists = await self.has_node(node_id, node_type)
        if exists:
            return self._graph.nodes[node_id]
        return None

    async def node_degree(self, node_id: str, edge_type: str) -> int:
        edges_with_edge_type = [(u, v) for u, v in self._graph.edges(node_id) if self._graph.edges[u, v]["type"] == edge_type]
        return len(edges_with_edge_type)

    async def edge_degree(self, src_id: str, tgt_id: str, edge_type: str) -> int:
        src_degree = await self.node_degree(src_id, edge_type)
        tgt_degree = await self.node_degree(tgt_id, edge_type)
        return src_degree + tgt_degree
    
    async def get_edge(
        self, source_node_id: str, target_node_id: str, edge_type: str
    ) -> dict[str, str] | None:
        exists = await self.has_edge(source_node_id, target_node_id, edge_type)
        if exists:
            return self._graph.edges[(source_node_id, target_node_id)]
        return None

    async def get_node_edges(self, source_node_id: str, node_type: str, edge_type: str) -> list[tuple[str, str]] | None:
        exists = await self.has_node(source_node_id, node_type)
        if exists:
            return [(u, v) for u, v, t in self._graph.edges(source_node_id, data="type") if t == edge_type]
        return None
    
    async def upsert_node(self, node_id: str, node_data: dict[str, str], node_type: str) -> None:
        assert node_data["type"] in {"ent", "the"} and node_data["type"] == node_type
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str], edge_type: str
    ) -> None:
        assert edge_data["type"] in {"rel", "the-ent", "the-hrc"} and edge_data["type"] == edge_type
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str) -> None:
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.debug(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    # Mingyu: Not sure why original networkx_impl not make this async
    def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node IDs to be deleted
        """
        for node in nodes:
            if self._graph.has_node(node):
                self._graph.remove_node(node)


    # Mingyu: Not sure why original networkx_impl not make this async
    def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:
            if self._graph.has_edge(source, target):
                self._graph.remove_edge(source, target)

    async def get_all_labels(self) -> list[str]:
        raise NotImplementedError

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
        raise NotImplementedError
    

