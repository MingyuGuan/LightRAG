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
# ASSUMPTIONS:
# - When adding an entity, if there is already a theme with the same name, 
#   it will be converted to an entity (theme-theme edge -> entity-theme edge, 
#   and theme-entity edge -> entity-entity edge).
# - When adding a theme, if there is already an entity with the same name, 
#   it will be merged into the entity (theme-theme edge -> entity-theme edge, 
#   and theme-entity edge -> entity-entity edge).
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
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

        self._entities = [node for node, type in self._graph.nodes(data='type') if type == "ent"]
        self._entity_subgraph = self._graph.subgraph(self._entities) # a view of the graph, need to update when the graph is updated
        self._themes = [node for node, type in self._graph.nodes(data='type') if type == "the"]
        self._theme_subgraph = self._graph.subgraph(self._themes) # a view of the graph, need to update when the graph is updated

    async def index_done_callback(self) -> None:
        NetworkXHeteroStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)
    
    async def has_entity(self, entity_id: str) -> bool:
        return self._entity_subgraph.has_node(entity_id)
    
    async def has_theme(self, theme_id: str) -> bool:
        return self._theme_subgraph.has_node(theme_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)
    
    async def has_relationship(self, source_node_id: str, target_node_id: str) -> bool:
        return self._entity_subgraph.has_edge(source_node_id, target_node_id)
    
    async def has_theme_hierarchy(self, parent_theme_id: str, child_theme_id: str) -> bool:
        return self._theme_subgraph.has_edge(parent_theme_id, child_theme_id)
    
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        return self._graph.nodes.get(node_id)

    async def get_entity(self, entity_id: str) -> dict[str, str] | None:
        return self._entity_subgraph.nodes.get(entity_id)
    
    async def get_theme(self, theme_id: str) -> dict[str, str] | None:
        return self._theme_subgraph.nodes.get(theme_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)
    
    async def entity_degree(self, entity_id: str) -> int:
        return self._entity_subgraph.degree(entity_id)
    
    async def theme_degree(self, theme_id: str) -> int:
        return self._theme_subgraph.degree(theme_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)
    
    async def relationship_degree(self, src_id: str, tgt_id: str) -> int:
        return self._entity_subgraph.degree(src_id) + self._entity_subgraph.degree(tgt_id)
    
    async def theme_hierarchy_degree(self, parent_id: str, child_id: str) -> int:
        return self._theme_subgraph.degree(parent_id) + self._theme_subgraph.degree(child_id)
    
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        return self._graph.edges.get((source_node_id, target_node_id))
    
    async def get_relationship(
        self, source_entity_id: str, target_entity_id: str
    ) -> dict[str, str] | None:
        return self._entity_subgraph.edges.get((source_entity_id, target_entity_id))
    
    async def get_theme_hierarchy(
        self, parent_theme_id: str, child_theme_id: str
    ) -> dict[str, str] | None:
        return self._theme_subgraph.edges.get((parent_theme_id, child_theme_id))

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None
    
    async def get_entity_relationships(self, source_entity_id: str) -> list[tuple[str, str]] | None:
        if self._entity_subgraph.has_node(source_entity_id):
            return list(self._entity_subgraph.edges(source_entity_id))
        return None
    
    async def get_theme_hierarchy(self, source_theme_id: str) -> list[tuple[str, str]] | None:
        if self._theme_subgraph.has_node(source_theme_id):
            return list(self._theme_subgraph.edges(source_theme_id))
        return None
    
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        self._graph.add_node(node_id, **node_data)

    # WARNING: this will NOT update the entity_subgraph, please call update_entity_subgraph() after (batched) upsert_entity()
    async def upsert_entity(self, entity_id: str, entity_data: dict[str, str]) -> None:
        assert entity_data["type"] == "ent"
        self._graph.add_node(entity_id, **entity_data)
        if entity_id not in self._entities:
            self._entities.append(entity_id)

    # WARNING: this will NOT update the theme_subgraph, please call update_theme_subgraph() after (batched) upsert_theme()
    async def upsert_theme(self, theme_id: str, theme_data: dict[str, str]) -> None:
        assert theme_data["type"] == "the"
        self._graph.add_node(theme_id, **theme_data)
        if theme_id not in self._themes:
            self._themes.append(theme_id)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    # WARNING: this will NOT update the entity_subgraph, please call update_entity_subgraph() after (batched) upsert_edge()
    async def upsert_relationship(self, 
        source_entity_id: str, target_entity_id: str, relationship_data: dict[str, str]
    ) -> None:
        assert relationship_data["type"] == "rel"
        self._graph.add_edge(source_entity_id, target_entity_id, **relationship_data)

    # WARNING: this will NOT update the theme_subgraph, please call update_theme_subgraph() after (batched) upsert_theme_hierarchy()
    async def upsert_theme_hierarchy(self, 
        parent_theme_id: str, child_theme_id: str, theme_hierarchy_data: dict[str, str]
    ) -> None:
        assert theme_hierarchy_data["type"] == "the-hrc"
        self._graph.add_edge(parent_theme_id, child_theme_id, **theme_hierarchy_data)

    async def delete_node(self, node_id: str) -> None:
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    # WARNING: this will NOT update the entity_subgraph, please call update_entity_subgraph() after (batched) delete_entity()
    async def delete_entity(self, entity_id: str) -> None:
        if self._graph.has_node(entity_id) and self._graph.nodes[entity_id]["type"] == "ent":
            self._graph.remove_node(entity_id)
            logger.info(f"Entity {entity_id} deleted from the graph.")
            self._entities.remove(entity_id) if entity_id in self._entities else None
        else:
            logger.warning(f"Entity {entity_id} not found in the graph for deletion.")

    # WARNING: this will NOT update the theme_subgraph, please call update_theme_subgraph() after (batched) delete_theme()
    async def delete_theme(self, theme_id: str) -> None:
        if self._graph.has_node(theme_id) and self._graph.nodes[theme_id]["type"] == "the":
            self._graph.remove_node(theme_id)
            logger.info(f"Theme {theme_id} deleted from the graph.")
            self._themes.remove(theme_id) if theme_id in self._themes else None
        else:
            logger.warning(f"Theme {theme_id} not found in the graph for deletion.")

    async def update_entity_subgraph(self) -> None:
        self._entity_subgraph = self._G.subgraph(self._entities)

    async def update_theme_subgraph(self) -> None:
        self._theme_subgraph = self._G.subgraph(self._themes)

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

    def remove_entities(self, entities: list[str]):
        """Delete multiple entities,
        and update the entity_subgraph after batched deletion.

        Args:
            entities: List of entity IDs to be deleted
        """
        for entity in entities:
            if self._graph.has_node(entity) and self._graph.nodes[entity]["type"] == "ent":
                self._graph.remove_node(entity)
                self._entities.remove(entity) if entity in self._entities else None
        self.update_entity_subgraph()

    def remove_themes(self, themes: list[str]):
        """Delete multiple themes,
        and update the theme_subgraph after batched deletion.

        Args:
            themes: List of theme IDs to be deleted
        """
        for theme in themes:
            if self._graph.has_node(theme) and self._graph.nodes[theme]["type"] == "the":
                self._graph.remove_node(theme)
                self._themes.remove(theme) if theme in self._themes else None
        self.update_theme_subgraph()

    # Mingyu: Not sure why original networkx_impl not make this async
    def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:
            if self._graph.has_edge(source, target):
                self._graph.remove_edge(source, target)

    
    def remove_relationships(self, relationships: list[tuple[str, str]]):
        """Delete multiple relationships,
        and update the entity_subgraph after batched deletion.

        Args:
            relationships: List of relationships to be deleted, each relationship is a (source, target) tuple
        """
        for source, target in relationships:
            if self._graph.has_edge(source, target) and self._graph.edges[source, target]["type"] == "rel":
                self._graph.remove_edge(source, target)
        self.update_entity_subgraph()

    def remove_theme_hierarchy(self, theme_hierarchies: list[tuple[str, str]]):
        """Delete multiple theme hierarchies,
        and update the theme_subgraph after batched deletion.

        Args:
            theme_hierarchies: List of theme hierarchies to be deleted, each theme hierarchy is a (parent, child) tuple
        """
        for parent, child in theme_hierarchies:
            if self._graph.has_edge(parent, child) and self._graph.edges[parent, child]["type"] == "the-hrc":
                self._graph.remove_edge(parent, child)
        self.update_theme_subgraph()

    async def get_all_labels(self) -> list[str]:
        raise NotImplementedError

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
        raise NotImplementedError
    

