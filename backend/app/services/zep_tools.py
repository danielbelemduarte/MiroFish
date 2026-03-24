"""
Graph search & retrieval tools service
Provides InsightForge, PanoramaSearch, QuickSearch, interview_agents
(Replaces former Zep Cloud implementation — now uses Neo4j via LocalGraphClient)
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from .local_graph_client import LocalGraphClient, GraphNode, GraphEdge

logger = get_logger('mirofish.zep_tools')


# ---------------------------------------------------------------------------
# Data classes (unchanged — used widely across report_agent, etc.)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts, "edges": self.edges,
            "nodes": self.nodes, "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        parts = [f"搜索查询: {self.query}", f"找到 {self.total_count} 条相关信息"]
        if self.facts:
            parts.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                parts.append(f"{i}. {fact}")
        return "\n".join(parts)


@dataclass
class NodeInfo:
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"uuid": self.uuid, "name": self.name, "labels": self.labels,
                "summary": self.summary, "attributes": self.attributes}

    def to_text(self) -> str:
        etype = next((l for l in self.labels if l not in ("Entity", "Node")), "未知类型")
        return f"实体: {self.name} (类型: {etype})\n摘要: {self.summary}"


@dataclass
class EdgeInfo:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid, "name": self.name, "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at, "valid_at": self.valid_at,
            "invalid_at": self.invalid_at, "expired_at": self.expired_at,
        }

    def to_text(self, include_temporal: bool = False) -> str:
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base = f"关系: {source} --[{self.name}]--> {target}\n事实: {self.fact}"
        if include_temporal:
            va = self.valid_at or "未知"
            ia = self.invalid_at or "至今"
            base += f"\n时效: {va} - {ia}"
            if self.expired_at:
                base += f" (已过期: {self.expired_at})"
        return base

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
        }

    def to_text(self) -> str:
        parts = [
            "## 未来预测深度分析",
            f"分析问题: {self.query}",
            f"预测场景: {self.simulation_requirement}",
            f"\n### 预测数据统计",
            f"- 相关预测事实: {self.total_facts}条",
            f"- 涉及实体: {self.total_entities}个",
            f"- 关系链: {self.total_relationships}条",
        ]
        if self.sub_queries:
            parts.append("\n### 分析的子问题")
            for i, sq in enumerate(self.sub_queries, 1):
                parts.append(f"{i}. {sq}")
        if self.semantic_facts:
            parts.append("\n### 【关键事实】(请在报告中引用这些原文)")
            for i, fact in enumerate(self.semantic_facts, 1):
                parts.append(f'{i}. "{fact}"')
        if self.entity_insights:
            parts.append("\n### 【核心实体】")
            for entity in self.entity_insights:
                parts.append(f"- **{entity.get('name', '未知')}** ({entity.get('type', '实体')})")
                if entity.get("summary"):
                    parts.append(f'  摘要: "{entity.get("summary")}"')
                if entity.get("related_facts"):
                    parts.append(f"  相关事实: {len(entity.get('related_facts', []))}条")
        if self.relationship_chains:
            parts.append("\n### 【关系链】")
            for chain in self.relationship_chains:
                parts.append(f"- {chain}")
        return "\n".join(parts)


@dataclass
class PanoramaResult:
    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes, "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count,
        }

    def to_text(self) -> str:
        parts = [
            "## 广度搜索结果（未来全景视图）",
            f"查询: {self.query}",
            f"\n### 统计信息",
            f"- 总节点数: {self.total_nodes}",
            f"- 总边数: {self.total_edges}",
            f"- 当前有效事实: {self.active_count}条",
            f"- 历史/过期事实: {self.historical_count}条",
        ]
        if self.active_facts:
            parts.append("\n### 【当前有效事实】(模拟结果原文)")
            for i, f in enumerate(self.active_facts, 1):
                parts.append(f'{i}. "{f}"')
        if self.historical_facts:
            parts.append("\n### 【历史/过期事实】(演变过程记录)")
            for i, f in enumerate(self.historical_facts, 1):
                parts.append(f'{i}. "{f}"')
        if self.all_nodes:
            parts.append("\n### 【涉及实体】")
            for node in self.all_nodes:
                etype = next((l for l in node.labels if l not in ("Entity", "Node")), "实体")
                parts.append(f"- **{node.name}** ({etype})")
        return "\n".join(parts)


@dataclass
class AgentInterview:
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name, "agent_role": self.agent_role,
            "agent_bio": self.agent_bio, "question": self.question,
            "response": self.response, "key_quotes": self.key_quotes,
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_简介: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**关键引言:**\n"
            for quote in self.key_quotes:
                clean_quote = quote.replace('“', '').replace('”', '').replace('"', '')
                clean_quote = clean_quote.replace('「', '').replace('」', '').strip()
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                skip = False
                for d in '123456789':
                    if f'问题{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('。', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    interview_topic: str
    interview_questions: List[str]
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)
    selection_reasoning: str = ""
    summary: str = ""
    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count,
        }

    def to_text(self) -> str:
        parts = [
            "## 深度采访报告",
            f"**采访主题:** {self.interview_topic}",
            f"**采访人数:** {self.interviewed_count} / {self.total_agents} 位模拟Agent",
            "\n### 采访对象选择理由",
            self.selection_reasoning or "（自动选择）",
            "\n---",
            "\n### 采访实录",
        ]
        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                parts.append(f"\n#### 采访 #{i}: {interview.agent_name}")
                parts.append(interview.to_text())
                parts.append("\n---")
        else:
            parts.append("（无采访记录）\n\n---")
        parts.append("\n### 采访摘要与核心观点")
        parts.append(self.summary or "（无摘要）")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class ZepToolsService:
    """
    Graph retrieval tools (Neo4j-backed).

    Kept as 'ZepToolsService' for import compatibility.
    """

    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.client = LocalGraphClient()
        self._llm_client = llm_client
        logger.info("ZepToolsService initialized (Neo4j backend)")

    @property
    def llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    # ---- helpers to convert internal types to NodeInfo/EdgeInfo ----------

    @staticmethod
    def _to_node_info(n: GraphNode) -> NodeInfo:
        return NodeInfo(
            uuid=n.uuid_, name=n.name or "", labels=n.labels or [],
            summary=n.summary or "", attributes=n.attributes or {},
        )

    @staticmethod
    def _to_edge_info(e: GraphEdge) -> EdgeInfo:
        return EdgeInfo(
            uuid=e.uuid_, name=e.name or "", fact=e.fact or "",
            source_node_uuid=e.source_node_uuid or "",
            target_node_uuid=e.target_node_uuid or "",
            created_at=e.created_at, valid_at=e.valid_at,
            invalid_at=e.invalid_at, expired_at=e.expired_at,
        )

    # ---- Basic retrieval ---------------------------------------------------

    def search_graph(
        self, graph_id: str, query: str, limit: int = 10, scope: str = "edges"
    ) -> SearchResult:
        logger.info(f"Graph search: graph_id={graph_id}, query={query[:50]}...")
        raw = self.client.search_graph(graph_id, query, limit, scope)

        facts = []
        edges = []
        nodes = []

        for e in raw.get("edges", []):
            if e.fact:
                facts.append(e.fact)
            edges.append({
                "uuid": e.uuid_, "name": e.name, "fact": e.fact,
                "source_node_uuid": e.source_node_uuid,
                "target_node_uuid": e.target_node_uuid,
            })

        for n in raw.get("nodes", []):
            nodes.append({
                "uuid": n.uuid_, "name": n.name,
                "labels": n.labels, "summary": n.summary,
            })
            if n.summary:
                facts.append(f"[{n.name}]: {n.summary}")

        logger.info(f"Search done: {len(facts)} facts found")
        return SearchResult(facts=facts, edges=edges, nodes=nodes,
                            query=query, total_count=len(facts))

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        nodes = self.client.get_nodes_by_graph(graph_id)
        return [self._to_node_info(n) for n in nodes]

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        edges = self.client.get_edges_by_graph(graph_id)
        return [self._to_edge_info(e) for e in edges]

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        n = self.client.get_node(node_uuid)
        return self._to_node_info(n) if n else None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        edges = self.client.get_node_edges(node_uuid)
        return [self._to_edge_info(e) for e in edges]

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        all_nodes = self.get_all_nodes(graph_id)
        return [n for n in all_nodes if entity_type in n.labels]

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        search_result = self.search_graph(graph_id, entity_name, limit=20)
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        related_edges = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        entity_types: Dict[str, int] = {}
        for node in nodes:
            for label in node.labels:
                if label not in ("Entity", "Node"):
                    entity_types[label] = entity_types.get(label, 0) + 1
        relation_types: Dict[str, int] = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes), "total_edges": len(edges),
            "entity_types": entity_types, "relation_types": relation_types,
        }

    def get_simulation_context(
        self, graph_id: str, simulation_requirement: str, limit: int = 30
    ) -> Dict[str, Any]:
        search_result = self.search_graph(graph_id, simulation_requirement, limit)
        stats = self.get_graph_statistics(graph_id)
        all_nodes = self.get_all_nodes(graph_id)
        entities = []
        for node in all_nodes:
            custom = [l for l in node.labels if l not in ("Entity", "Node")]
            if custom:
                entities.append({"name": node.name, "type": custom[0], "summary": node.summary})
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities),
        }

    # ---- Advanced retrieval tools ------------------------------------------

    def insight_forge(
        self, graph_id: str, query: str, simulation_requirement: str,
        report_context: str = "", max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        logger.info(f"InsightForge: {query[:50]}...")
        result = InsightForgeResult(query=query, simulation_requirement=simulation_requirement, sub_queries=[])

        sub_queries = self._generate_sub_queries(query, simulation_requirement, report_context, max_sub_queries)
        result.sub_queries = sub_queries

        all_facts: List[str] = []
        all_edges: List[Dict] = []
        seen_facts: set = set()

        for sq in sub_queries:
            sr = self.search_graph(graph_id, sq, limit=15, scope="edges")
            for fact in sr.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            all_edges.extend(sr.edges)

        main_sr = self.search_graph(graph_id, query, limit=20, scope="edges")
        for fact in main_sr.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        entity_uuids: set = set()
        for ed in all_edges:
            if isinstance(ed, dict):
                for k in ("source_node_uuid", "target_node_uuid"):
                    if ed.get(k):
                        entity_uuids.add(ed[k])

        entity_insights = []
        node_map: Dict[str, NodeInfo] = {}
        for uid in entity_uuids:
            if not uid:
                continue
            node = self.get_node_detail(uid)
            if node:
                node_map[uid] = node
                etype = next((l for l in node.labels if l not in ("Entity", "Node")), "实体")
                related = [f for f in all_facts if node.name.lower() in f.lower()]
                entity_insights.append({
                    "uuid": node.uuid, "name": node.name, "type": etype,
                    "summary": node.summary, "related_facts": related,
                })
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        chains: List[str] = []
        for ed in all_edges:
            if isinstance(ed, dict):
                src = node_map.get(ed.get("source_node_uuid", ""))
                tgt = node_map.get(ed.get("target_node_uuid", ""))
                sn = src.name if src else (ed.get("source_node_uuid", "")[:8])
                tn = tgt.name if tgt else (ed.get("target_node_uuid", "")[:8])
                chain = f"{sn} --[{ed.get('name', '')}]--> {tn}"
                if chain not in chains:
                    chains.append(chain)
        result.relationship_chains = chains
        result.total_relationships = len(chains)

        logger.info(f"InsightForge done: {result.total_facts} facts, {result.total_entities} entities, {result.total_relationships} rels")
        return result

    def _generate_sub_queries(
        self, query: str, simulation_requirement: str,
        report_context: str = "", max_queries: int = 5,
    ) -> List[str]:
        system_prompt = (
            "你是一个专业的问题分析专家。你的任务是将一个复杂问题分解为多个可以在模拟世界中独立观察的子问题。\n\n"
            "要求：\n1. 每个子问题应该足够具体\n2. 子问题应该覆盖原问题的不同维度\n3. 子问题应该与模拟场景相关\n4. 返回JSON格式：{\"sub_queries\": [\"子问题1\", \"子问题2\", ...]}"
        )
        user_prompt = (
            f"模拟需求背景：\n{simulation_requirement}\n\n"
            f"{f'报告上下文：{report_context[:500]}' if report_context else ''}\n\n"
            f"请将以下问题分解为{max_queries}个子问题：\n{query}\n\n"
            "返回JSON格式的子问题列表。"
        )
        try:
            resp = self.llm.chat_json(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.3,
            )
            return [str(sq) for sq in resp.get("sub_queries", [])[:max_queries]]
        except Exception as e:
            logger.warning(f"Sub-query generation failed: {e}")
            return [query, f"{query} 的主要参与者",
                    f"{query} 的原因和影响",
                    f"{query} 的发展过程"][:max_queries]

    def panorama_search(
        self, graph_id: str, query: str,
        include_expired: bool = True, limit: int = 50,
    ) -> PanoramaResult:
        logger.info(f"PanoramaSearch: {query[:50]}...")
        result = PanoramaResult(query=query)

        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        active_facts: List[str] = []
        historical_facts: List[str] = []

        for edge in all_edges:
            if not edge.fact:
                continue
            is_historical = edge.is_expired or edge.is_invalid
            if is_historical:
                va = edge.valid_at or "未知"
                ia = edge.invalid_at or edge.expired_at or "未知"
                historical_facts.append(f"[{va} - {ia}] {edge.fact}")
            else:
                active_facts.append(edge.fact)

        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def rel_score(fact: str) -> int:
            fl = fact.lower()
            s = 0
            if query_lower in fl:
                s += 100
            for kw in keywords:
                if kw in fl:
                    s += 10
            return s

        active_facts.sort(key=rel_score, reverse=True)
        historical_facts.sort(key=rel_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(f"PanoramaSearch done: {result.active_count} active, {result.historical_count} historical")
        return result

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        logger.info(f"QuickSearch: {query[:50]}...")
        return self.search_graph(graph_id, query, limit, scope="edges")

    # ---- Interview (unchanged logic, no Zep dependency) --------------------

    def interview_agents(
        self, simulation_id: str, interview_requirement: str,
        simulation_requirement: str = "", max_agents: int = 5,
        custom_questions: List[str] = None,
    ) -> InterviewResult:
        from .simulation_runner import SimulationRunner

        logger.info(f"InterviewAgents: {interview_requirement[:50]}...")
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        profiles = self._load_agent_profiles(simulation_id)
        if not profiles:
            result.summary = "未找到可采访的Agent人设文件"
            return result

        result.total_agents = len(profiles)

        selected_agents, selected_indices, reasoning = self._select_agents_for_interview(
            profiles, interview_requirement, simulation_requirement, max_agents,
        )
        result.selected_agents = selected_agents
        result.selection_reasoning = reasoning

        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement, simulation_requirement, selected_agents,
            )

        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        INTERVIEW_PROMPT_PREFIX = (
            "你正在接受一次采访。请结合你的人设、所有的过往记忆与行动，"
            "以纯文本方式直接回答以下问题。\n"
            "回复要求：\n"
            "1. 直接用自然语言回答，不要调用任何工具\n"
            "2. 不要返回JSON格式或工具调用格式\n"
            "3. 不要使用Markdown标题（如#、##、###）\n"
            "4. 按问题编号逐一回答，每个回答以「问题X：」开头\n"
            "5. 每个问题的回答之间用空行分隔\n"
            "6. 回答要有实质内容，每个问题至少回答2-3句话\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        try:
            interviews_request = [
                {"agent_id": idx, "prompt": optimized_prompt}
                for idx in selected_indices
            ]
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0,
            )

            if not api_result.get("success", False):
                result.summary = f"采访API调用失败：{api_result.get('error', '未知错误')}"
                return result

            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "未知")
                agent_bio = agent.get("bio", "")

                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                twitter_response = self._clean_tool_call_response(twitter_result.get("response", ""))
                reddit_response = self._clean_tool_call_response(reddit_result.get("response", ""))

                twitter_text = twitter_response if twitter_response else "（该平台未获得回复）"
                reddit_text = reddit_response if reddit_response else "（该平台未获得回复）"
                response_text = f"【Twitter平台回答】\n{twitter_text}\n\n【Reddit平台回答】\n{reddit_text}"

                import re
                combined_responses = f"{twitter_response} {reddit_response}"
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'问题\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', '问题'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                if not key_quotes:
                    paired = re.findall(r'“([^“”]{15,100})”', clean_text)
                    paired += re.findall(r'「([^「」]{15,100})」', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]

                result.interviews.append(AgentInterview(
                    agent_name=agent_name, agent_role=agent_role,
                    agent_bio=agent_bio[:1000], question=combined_prompt,
                    response=response_text, key_quotes=key_quotes[:5],
                ))

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            result.summary = f"采访失败：{str(e)}"
            return result
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"采访过程发生错误：{str(e)}"
            return result

        if result.interviews:
            result.summary = self._generate_interview_summary(result.interviews, interview_requirement)

        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        import os, csv
        sim_dir = os.path.join(os.path.dirname(__file__), f'../../uploads/simulations/{simulation_id}')
        profiles: List[Dict[str, Any]] = []

        reddit_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_path):
            try:
                with open(reddit_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read reddit_profiles.json: {e}")

        twitter_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_path):
            try:
                with open(twitter_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "未知",
                        })
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read twitter_profiles.csv: {e}")

        return profiles

    def _select_agents_for_interview(self, profiles, interview_requirement, simulation_requirement, max_agents):
        agent_summaries = []
        for i, p in enumerate(profiles):
            agent_summaries.append({
                "index": i,
                "name": p.get("realname", p.get("username", f"Agent_{i}")),
                "profession": p.get("profession", "未知"),
                "bio": p.get("bio", "")[:200],
                "interested_topics": p.get("interested_topics", []),
            })

        system_prompt = (
            "你是一个专业的采访策划专家。根据采访需求从模拟Agent列表中选择最适合采访的对象。\n"
            "选择标准：\n1. Agent的身份/职业与采访主题相关\n2. Agent可能持有独特观点\n"
            "3. 选择多样化视角\n4. 优先选择与事件直接相关的角色\n\n"
            "返回JSON: {\"selected_indices\": [...], \"reasoning\": \"...\"}"
        )
        user_prompt = (
            f"采访需求：{interview_requirement}\n\n"
            f"模拟背景：{simulation_requirement or '未提供'}\n\n"
            f"可选Agent列表（共{len(agent_summaries)}个）：\n"
            f"{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}\n\n"
            f"请选择最多{max_agents}个最适合的Agent。"
        )

        try:
            resp = self.llm.chat_json(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.3,
            )
            selected_indices = resp.get("selected_indices", [])[:max_agents]
            reasoning = resp.get("reasoning", "基于相关性自动选择")
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            return selected_agents, valid_indices, reasoning
        except Exception:
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "使用默认选择策略"

    def _generate_interview_questions(self, interview_requirement, simulation_requirement, selected_agents):
        agent_roles = [a.get("profession", "未知") for a in selected_agents]
        system_prompt = (
            "你是一个专业的记者。根据采访需求生成3-5个深度采访问题。\n"
            "问题要求：\n1. 开放性\n2. 针对不同角色\n3. 覆盖多维度\n4. 自然\n5. 每个50字内\n6. 直接提问\n\n"
            "返回JSON: {\"questions\": [...]}"
        )
        user_prompt = (
            f"采访需求：{interview_requirement}\n"
            f"模拟背景：{simulation_requirement or '未提供'}\n"
            f"采访对象角色：{', '.join(agent_roles)}\n"
            "请生成3-5个采访问题。"
        )
        try:
            resp = self.llm.chat_json(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.5,
            )
            return resp.get("questions", [f"关于{interview_requirement}，您有什么看法？"])
        except Exception:
            return [
                f"关于{interview_requirement}，您的观点是什么？",
                "这件事对您或您所代表的群体有什么影响？",
                "您认为应该如何解决或改进这个问题？",
            ]

    def _generate_interview_summary(self, interviews, interview_requirement):
        if not interviews:
            return "未完成任何采访"
        texts = []
        for iv in interviews:
            texts.append(f"【{iv.agent_name}（{iv.agent_role}）】\n{iv.response[:500]}")
        system_prompt = (
            "你是一个专业的新闻编辑。请根据多位受访者的回答生成采访摘要。\n"
            "摘要要求：\n1. 提炼各方主要观点\n2. 指出共识和分歧\n3. 突出有价值的引言\n"
            "4. 客观中立\n5. 控制在1000字内\n\n"
            "格式约束：\n- 使用纯文本段落\n- 不要Markdown标题\n- 不要分割线\n"
            "- 引用用「」\n- 可以用**加粗**"
        )
        user_prompt = f"采访主题：{interview_requirement}\n\n采访内容：\n{''.join(texts)}\n\n请生成采访摘要。"
        try:
            return self.llm.chat(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.3, max_tokens=800,
            )
        except Exception:
            return f"共采访了{len(interviews)}位受访者，包括：" + "、".join([i.agent_name for i in interviews])
