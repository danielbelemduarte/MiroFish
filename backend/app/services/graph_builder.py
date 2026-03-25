"""
Graph builder service
Uses Neo4j + LLM extraction (replaces Zep Cloud)
"""

import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from .local_graph_client import LocalGraphClient
from .llm_entity_extractor import LLMEntityExtractor
from .text_processor import TextProcessor


@dataclass
class GraphInfo:
    """Graph metadata"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    Graph builder service — builds knowledge graphs using
    Neo4j for storage and LLM for entity/relation extraction.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = LocalGraphClient()
        self.extractor = LLMEntityExtractor()
        self.task_manager = TaskManager()

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        """Async graph build — returns a task ID immediately."""
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name,
                  chunk_size, chunk_overlap, batch_size)
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int
    ):
        """Background worker thread for graph building."""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="Starting graph build..."
            )

            # 1. Create graph
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=f"Graph created: {graph_id}"
            )

            # 2. Store ontology
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message="Ontology saved"
            )

            # 3. Chunk text
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=f"Text split into {total_chunks} chunks"
            )

            # 4. Process chunks — extract entities/relations with LLM, write to Neo4j
            self._process_chunks(
                graph_id, chunks, ontology, batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.7),  # 20-90%
                    message=msg
                )
            )

            # 5. Collect graph info
            self.task_manager.update_task(
                task_id, progress=90,
                message="Collecting graph info..."
            )
            graph_info = self._get_graph_info(graph_id)

            # Done
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    # ------------------------------------------------------------------
    # Public helpers (also called directly by graph.py endpoints)
    # ------------------------------------------------------------------

    def create_graph(self, name: str) -> str:
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        self.client.create_graph(
            graph_id=graph_id, name=name,
            description="MiroFish Social Simulation Graph"
        )
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        self.client.set_ontology(graph_id, ontology)

    def _process_chunks(
        self,
        graph_id: str,
        chunks: List[str],
        ontology: Dict[str, Any],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
        start_from: int = 0,
        entity_uuid_map: Optional[Dict[str, str]] = None,
        save_progress_callback: Optional[Callable[[int], None]] = None,
    ):
        """
        For each text chunk:
        1. Store as episode
        2. LLM-extract entities & relations
        3. Write extracted data to Neo4j

        Args:
            start_from: chunk index to resume from (0-based, skip chunks before this)
            entity_uuid_map: pre-built name→uuid map (for resume)
            save_progress_callback: called with chunks_processed count after each chunk
        """
        total = len(chunks)
        # Keep a name->uuid map to link relations correctly
        if entity_uuid_map is None:
            entity_uuid_map = {}

        for i, chunk in enumerate(chunks):
            if i < start_from:
                continue

            batch_num = i + 1

            if progress_callback:
                progress_callback(
                    f"Processing chunk {batch_num}/{total}...",
                    (i + 1) / total
                )

            # Store episode
            ep_uuid = self.client.add_episode(graph_id, chunk)

            # LLM extraction
            extraction = self.extractor.extract(chunk, ontology)

            # Create nodes
            for ent in extraction.get("entities", []):
                name = ent["name"]
                if name not in entity_uuid_map:
                    node = self.client.create_node(
                        graph_id=graph_id,
                        name=name,
                        entity_type=ent["type"],
                        summary=ent.get("summary", ""),
                    )
                    entity_uuid_map[name] = node.uuid_

            # Create edges
            for rel in extraction.get("relationships", []):
                src_name = rel["source"]
                tgt_name = rel["target"]

                # Auto-create nodes referenced in relations but not yet seen
                for ref_name in (src_name, tgt_name):
                    if ref_name not in entity_uuid_map:
                        node = self.client.create_node(
                            graph_id=graph_id,
                            name=ref_name,
                            entity_type="Entity",
                            summary="",
                        )
                        entity_uuid_map[ref_name] = node.uuid_

                self.client.create_edge(
                    graph_id=graph_id,
                    source_node_uuid=entity_uuid_map[src_name],
                    target_node_uuid=entity_uuid_map[tgt_name],
                    name=rel["name"],
                    fact=rel["fact"],
                )

            # Mark episode processed
            self.client.mark_episode_processed(ep_uuid)

            # Persist resume progress
            if save_progress_callback:
                save_progress_callback(i + 1)

    def rebuild_entity_map(self, graph_id: str) -> Dict[str, str]:
        """Rebuild entity name→uuid map from existing Neo4j nodes (for resume)."""
        nodes = self.client.get_nodes_by_graph(graph_id)
        return {node.name: node.uuid_ for node in nodes}

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        nodes = self.client.get_nodes_by_graph(graph_id)
        edges = self.client.get_edges_by_graph(graph_id)

        entity_types = set()
        for node in nodes:
            for label in node.labels:
                if label not in ("Entity", "Node"):
                    entity_types.add(label)

        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=list(entity_types),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """Get full graph data for visualization."""
        nodes = self.client.get_nodes_by_graph(graph_id)
        edges = self.client.get_edges_by_graph(graph_id)

        node_map = {n.uuid_: n.name for n in nodes}

        nodes_data = []
        for node in nodes:
            nodes_data.append({
                "uuid": node.uuid_,
                "name": node.name,
                "labels": node.labels or [],
                "summary": node.summary or "",
                "attributes": node.attributes or {},
                "created_at": node.created_at,
            })

        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": edge.uuid_,
                "name": edge.name or "",
                "fact": edge.fact or "",
                "fact_type": edge.fact_type or edge.name or "",
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "source_node_name": node_map.get(edge.source_node_uuid, ""),
                "target_node_name": node_map.get(edge.target_node_uuid, ""),
                "attributes": edge.attributes or {},
                "created_at": edge.created_at,
                "valid_at": edge.valid_at,
                "invalid_at": edge.invalid_at,
                "expired_at": edge.expired_at,
                "episodes": edge.episodes or [],
            })

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        self.client.delete_graph(graph_id=graph_id)
