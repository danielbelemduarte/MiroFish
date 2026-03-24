"""
Local Graph Client — Neo4j-based replacement for Zep Cloud.

Provides a client interface compatible with the rest of the MiroFish codebase
so that graph_builder, entity_reader, tools, and memory updater can work
without any Zep Cloud dependency.
"""

import uuid
import json
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from neo4j import GraphDatabase

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.local_graph')

# ---------------------------------------------------------------------------
# Data classes mirroring what the rest of the codebase expects
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    uuid_: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    created_at: Optional[str] = None

@dataclass
class GraphEdge:
    uuid_: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    fact_type: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: Optional[List[str]] = None

@dataclass
class GraphEpisode:
    uuid_: str
    graph_id: str
    data: str
    type: str = "text"
    processed: bool = False


# ---------------------------------------------------------------------------
# Singleton Neo4j driver manager
# ---------------------------------------------------------------------------

class Neo4jManager:
    """Thread-safe singleton that holds one Neo4j driver instance."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._driver = None
        return cls._instance

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            )
            self._ensure_indexes()
        return self._driver

    def _ensure_indexes(self):
        """Create indexes / constraints on first connection."""
        with self._driver.session() as s:
            # Unique constraint on Entity uuid
            s.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.uuid IS UNIQUE"
            )
            # Index on graph_id for fast filtering
            s.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.graph_id)"
            )
            # Full-text index for keyword search on facts
            try:
                s.run(
                    "CREATE FULLTEXT INDEX edge_fact_fulltext IF NOT EXISTS "
                    "FOR ()-[r:RELATION]-() ON EACH [r.fact, r.name]"
                )
            except Exception:
                pass  # May not be supported on all Neo4j editions
            # Full-text index on node name + summary
            try:
                s.run(
                    "CREATE FULLTEXT INDEX node_fulltext IF NOT EXISTS "
                    "FOR (n:Entity) ON EACH [n.name, n.summary]"
                )
            except Exception:
                pass

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None


# ---------------------------------------------------------------------------
# Local Graph Client — public API used by the rest of MiroFish
# ---------------------------------------------------------------------------

class LocalGraphClient:
    """
    Drop-in replacement for the Zep client.

    Usage is intentionally similar to the old Zep calls so that call-sites
    require minimal changes.
    """

    def __init__(self):
        self._neo4j = Neo4jManager()

    # ---- Graph lifecycle ---------------------------------------------------

    def create_graph(self, graph_id: str, name: str, description: str = "") -> str:
        """Register a new graph (stored as metadata node)."""
        with self._neo4j.driver.session() as s:
            s.run(
                "MERGE (g:_GraphMeta {graph_id: $gid}) "
                "SET g.name = $name, g.description = $desc, g.created_at = $now",
                gid=graph_id, name=name, desc=description,
                now=datetime.utcnow().isoformat(),
            )
        logger.info(f"Graph created: {graph_id}")
        return graph_id

    def delete_graph(self, graph_id: str):
        """Delete all nodes, edges and episodes for a graph."""
        with self._neo4j.driver.session() as s:
            s.run(
                "MATCH (n {graph_id: $gid}) DETACH DELETE n",
                gid=graph_id,
            )
            s.run(
                "MATCH (g:_GraphMeta {graph_id: $gid}) DELETE g",
                gid=graph_id,
            )
        logger.info(f"Graph deleted: {graph_id}")

    # ---- Ontology ----------------------------------------------------------

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """
        Store the ontology definition for the graph.
        Kept as a JSON property on the _GraphMeta node so we can use it
        during LLM extraction.
        """
        with self._neo4j.driver.session() as s:
            s.run(
                "MERGE (g:_GraphMeta {graph_id: $gid}) "
                "SET g.ontology = $ont",
                gid=graph_id, ont=json.dumps(ontology, ensure_ascii=False),
            )
        logger.info(f"Ontology set for graph {graph_id}")

    def get_ontology(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored ontology for a graph."""
        with self._neo4j.driver.session() as s:
            result = s.run(
                "MATCH (g:_GraphMeta {graph_id: $gid}) RETURN g.ontology AS ont",
                gid=graph_id,
            )
            record = result.single()
            if record and record["ont"]:
                return json.loads(record["ont"])
        return None

    # ---- Nodes (entities) --------------------------------------------------

    def create_node(
        self,
        graph_id: str,
        name: str,
        entity_type: str,
        summary: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> GraphNode:
        """Create or merge an entity node. Returns the node."""
        node_uuid = uuid.uuid4().hex
        now = datetime.utcnow().isoformat()
        attrs_json = json.dumps(attributes or {}, ensure_ascii=False)

        with self._neo4j.driver.session() as s:
            # MERGE on (graph_id, name, entity_type) to avoid duplicates
            s.run(
                "MERGE (n:Entity {graph_id: $gid, name: $name, entity_type: $etype}) "
                "ON CREATE SET n.uuid = $uuid, n.summary = $summary, "
                "  n.attributes_json = $attrs, n.created_at = $now "
                "ON MATCH SET n.summary = CASE WHEN size(n.summary) < size($summary) "
                "  THEN $summary ELSE n.summary END",
                gid=graph_id, name=name, etype=entity_type,
                uuid=node_uuid, summary=summary, attrs=attrs_json, now=now,
            )
            # Retrieve the actual node (could be existing due to MERGE)
            rec = s.run(
                "MATCH (n:Entity {graph_id: $gid, name: $name, entity_type: $etype}) "
                "RETURN n.uuid AS uuid, n.summary AS summary, "
                "  n.attributes_json AS attrs, n.created_at AS created_at",
                gid=graph_id, name=name, etype=entity_type,
            ).single()

        return GraphNode(
            uuid_=rec["uuid"],
            name=name,
            labels=["Entity", entity_type],
            summary=rec["summary"] or "",
            attributes=json.loads(rec["attrs"]) if rec["attrs"] else {},
            created_at=rec["created_at"],
        )

    def get_node(self, node_uuid: str) -> Optional[GraphNode]:
        """Get a single node by UUID."""
        with self._neo4j.driver.session() as s:
            rec = s.run(
                "MATCH (n:Entity {uuid: $uuid}) "
                "RETURN n.uuid AS uuid, n.name AS name, n.entity_type AS etype, "
                "  n.summary AS summary, n.attributes_json AS attrs, "
                "  n.created_at AS created_at, n.graph_id AS gid",
                uuid=node_uuid,
            ).single()
        if not rec:
            return None
        return GraphNode(
            uuid_=rec["uuid"],
            name=rec["name"] or "",
            labels=["Entity", rec["etype"]] if rec["etype"] else ["Entity"],
            summary=rec["summary"] or "",
            attributes=json.loads(rec["attrs"]) if rec["attrs"] else {},
            created_at=rec["created_at"],
        )

    def get_nodes_by_graph(self, graph_id: str) -> List[GraphNode]:
        """Get all entity nodes for a graph."""
        with self._neo4j.driver.session() as s:
            result = s.run(
                "MATCH (n:Entity {graph_id: $gid}) "
                "RETURN n.uuid AS uuid, n.name AS name, n.entity_type AS etype, "
                "  n.summary AS summary, n.attributes_json AS attrs, "
                "  n.created_at AS created_at "
                "ORDER BY n.name",
                gid=graph_id,
            )
            nodes = []
            for rec in result:
                nodes.append(GraphNode(
                    uuid_=rec["uuid"],
                    name=rec["name"] or "",
                    labels=["Entity", rec["etype"]] if rec["etype"] else ["Entity"],
                    summary=rec["summary"] or "",
                    attributes=json.loads(rec["attrs"]) if rec["attrs"] else {},
                    created_at=rec["created_at"],
                ))
        return nodes

    # ---- Edges (relations) -------------------------------------------------

    def create_edge(
        self,
        graph_id: str,
        source_node_uuid: str,
        target_node_uuid: str,
        name: str,
        fact: str,
        fact_type: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> GraphEdge:
        """Create a relationship between two nodes."""
        edge_uuid = uuid.uuid4().hex
        now = datetime.utcnow().isoformat()
        attrs_json = json.dumps(attributes or {}, ensure_ascii=False)

        with self._neo4j.driver.session() as s:
            s.run(
                "MATCH (src:Entity {uuid: $src_uuid}) "
                "MATCH (tgt:Entity {uuid: $tgt_uuid}) "
                "CREATE (src)-[r:RELATION {"
                "  uuid: $uuid, graph_id: $gid, name: $name, fact: $fact, "
                "  fact_type: $ftype, attributes_json: $attrs, "
                "  created_at: $now, valid_at: $now"
                "}]->(tgt)",
                src_uuid=source_node_uuid, tgt_uuid=target_node_uuid,
                uuid=edge_uuid, gid=graph_id, name=name, fact=fact,
                ftype=fact_type, attrs=attrs_json, now=now,
            )
        return GraphEdge(
            uuid_=edge_uuid, name=name, fact=fact,
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            attributes=attributes or {},
            fact_type=fact_type, created_at=now, valid_at=now,
        )

    def get_edges_by_graph(self, graph_id: str) -> List[GraphEdge]:
        """Get all edges for a graph."""
        with self._neo4j.driver.session() as s:
            result = s.run(
                "MATCH (src:Entity)-[r:RELATION {graph_id: $gid}]->(tgt:Entity) "
                "RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact, "
                "  r.fact_type AS ftype, r.attributes_json AS attrs, "
                "  src.uuid AS src_uuid, tgt.uuid AS tgt_uuid, "
                "  r.created_at AS created_at, r.valid_at AS valid_at, "
                "  r.invalid_at AS invalid_at, r.expired_at AS expired_at",
                gid=graph_id,
            )
            edges = []
            for rec in result:
                edges.append(GraphEdge(
                    uuid_=rec["uuid"] or "",
                    name=rec["name"] or "",
                    fact=rec["fact"] or "",
                    source_node_uuid=rec["src_uuid"] or "",
                    target_node_uuid=rec["tgt_uuid"] or "",
                    attributes=json.loads(rec["attrs"]) if rec["attrs"] else {},
                    fact_type=rec["ftype"] or "",
                    created_at=rec["created_at"],
                    valid_at=rec["valid_at"],
                    invalid_at=rec["invalid_at"],
                    expired_at=rec["expired_at"],
                ))
        return edges

    def get_node_edges(self, node_uuid: str) -> List[GraphEdge]:
        """Get all edges connected to a specific node."""
        with self._neo4j.driver.session() as s:
            result = s.run(
                "MATCH (n:Entity {uuid: $uuid})-[r:RELATION]-(other:Entity) "
                "WITH r, startNode(r) AS src, endNode(r) AS tgt "
                "RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact, "
                "  r.fact_type AS ftype, r.attributes_json AS attrs, "
                "  src.uuid AS src_uuid, tgt.uuid AS tgt_uuid, "
                "  r.created_at AS created_at, r.valid_at AS valid_at, "
                "  r.invalid_at AS invalid_at, r.expired_at AS expired_at",
                uuid=node_uuid,
            )
            edges = []
            for rec in result:
                edges.append(GraphEdge(
                    uuid_=rec["uuid"] or "",
                    name=rec["name"] or "",
                    fact=rec["fact"] or "",
                    source_node_uuid=rec["src_uuid"] or "",
                    target_node_uuid=rec["tgt_uuid"] or "",
                    attributes=json.loads(rec["attrs"]) if rec["attrs"] else {},
                    fact_type=rec["ftype"] or "",
                    created_at=rec["created_at"],
                    valid_at=rec["valid_at"],
                    invalid_at=rec["invalid_at"],
                    expired_at=rec["expired_at"],
                ))
        return edges

    # ---- Search ------------------------------------------------------------

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> Dict[str, Any]:
        """
        Keyword search over the graph.

        Uses Neo4j full-text indexes when available, falls back to
        CONTAINS-based search.

        Returns dict with 'edges' and 'nodes' lists.
        """
        edges_result = []
        nodes_result = []

        query_lower = query.lower()
        keywords = [
            w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split()
            if len(w.strip()) > 1
        ]

        if scope in ("edges", "both"):
            edges_result = self._search_edges(graph_id, keywords, query_lower, limit)

        if scope in ("nodes", "both"):
            nodes_result = self._search_nodes(graph_id, keywords, query_lower, limit)

        return {"edges": edges_result, "nodes": nodes_result}

    def _search_edges(
        self, graph_id: str, keywords: List[str], query_lower: str, limit: int
    ) -> List[GraphEdge]:
        """Search edges by keywords."""
        all_edges = self.get_edges_by_graph(graph_id)

        def score(edge: GraphEdge) -> int:
            s = 0
            text = f"{edge.fact} {edge.name}".lower()
            if query_lower in text:
                s += 100
            for kw in keywords:
                if kw in text:
                    s += 10
            return s

        scored = [(score(e), e) for e in all_edges]
        scored = [(s, e) for s, e in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def _search_nodes(
        self, graph_id: str, keywords: List[str], query_lower: str, limit: int
    ) -> List[GraphNode]:
        """Search nodes by keywords."""
        all_nodes = self.get_nodes_by_graph(graph_id)

        def score(node: GraphNode) -> int:
            s = 0
            text = f"{node.name} {node.summary}".lower()
            if query_lower in text:
                s += 100
            for kw in keywords:
                if kw in text:
                    s += 10
            return s

        scored = [(score(n), n) for n in all_nodes]
        scored = [(s, n) for s, n in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:limit]]

    # ---- Episodes (text ingestion tracking) --------------------------------

    def add_episode(self, graph_id: str, data: str, type: str = "text") -> str:
        """Store a text episode. Returns its UUID."""
        ep_uuid = uuid.uuid4().hex
        with self._neo4j.driver.session() as s:
            s.run(
                "CREATE (:_Episode {"
                "  uuid: $uuid, graph_id: $gid, data: $data, type: $type, "
                "  processed: false, created_at: $now"
                "})",
                uuid=ep_uuid, gid=graph_id, data=data, type=type,
                now=datetime.utcnow().isoformat(),
            )
        return ep_uuid

    def mark_episode_processed(self, episode_uuid: str):
        """Mark an episode as processed."""
        with self._neo4j.driver.session() as s:
            s.run(
                "MATCH (e:_Episode {uuid: $uuid}) SET e.processed = true",
                uuid=episode_uuid,
            )

    def is_episode_processed(self, episode_uuid: str) -> bool:
        """Check if an episode has been processed."""
        with self._neo4j.driver.session() as s:
            rec = s.run(
                "MATCH (e:_Episode {uuid: $uuid}) RETURN e.processed AS p",
                uuid=episode_uuid,
            ).single()
        return bool(rec and rec["p"])

    def get_unprocessed_episodes(self, graph_id: str) -> List[GraphEpisode]:
        """Get all unprocessed episodes for a graph."""
        with self._neo4j.driver.session() as s:
            result = s.run(
                "MATCH (e:_Episode {graph_id: $gid, processed: false}) "
                "RETURN e.uuid AS uuid, e.data AS data, e.type AS type "
                "ORDER BY e.created_at",
                gid=graph_id,
            )
            return [
                GraphEpisode(uuid_=r["uuid"], graph_id=graph_id,
                             data=r["data"], type=r["type"])
                for r in result
            ]
