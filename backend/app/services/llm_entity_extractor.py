"""
LLM-based entity and relation extractor.

Replaces Zep Cloud's automatic entity extraction. Uses the configured LLM
(Azure OpenAI / OpenAI) to extract entities and relationships from text
chunks based on the stored ontology.
"""

import json
from typing import Dict, Any, List, Optional

from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient

logger = get_logger('mirofish.llm_extractor')


class LLMEntityExtractor:
    """
    Extract entities and relationships from text using an LLM.

    Given an ontology (entity types + edge types) and a text chunk,
    the extractor asks the LLM to return structured JSON with the
    entities and relationships found.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def extract(
        self,
        text: str,
        ontology: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from *text* according to *ontology*.

        Returns::

            {
                "entities": [
                    {"name": "...", "type": "...", "summary": "..."},
                    ...
                ],
                "relationships": [
                    {
                        "source": "entity_name",
                        "target": "entity_name",
                        "name": "relationship_type",
                        "fact": "descriptive sentence"
                    },
                    ...
                ]
            }
        """
        entity_types = ontology.get("entity_types", [])
        edge_types = ontology.get("edge_types", [])

        # Build a concise description of the ontology for the prompt
        entity_desc = self._format_entity_types(entity_types)
        edge_desc = self._format_edge_types(edge_types)

        system_prompt = (
            "You are a knowledge-graph extraction engine. "
            "Given a text and an ontology definition, extract all entities "
            "and relationships that match the ontology.\n\n"
            "Rules:\n"
            "1. Only extract entities whose type matches one of the defined entity types.\n"
            "2. Only extract relationships whose type matches one of the defined edge types.\n"
            "3. Each entity must have: name, type, summary (1-2 sentence description).\n"
            "4. Each relationship must have: source (entity name), target (entity name), "
            "name (relationship type), fact (a sentence describing the relationship).\n"
            "5. Entity names should be normalized (consistent casing, no extra whitespace).\n"
            "6. If the same entity appears multiple times, merge into one entry.\n"
            "7. Return valid JSON only, no markdown fences.\n"
            "8. If no entities or relationships are found, return empty lists.\n\n"
            f"## Entity Types\n{entity_desc}\n\n"
            f"## Edge Types\n{edge_desc}\n\n"
            "Return JSON: {\"entities\": [...], \"relationships\": [...]}"
        )

        user_prompt = f"Extract entities and relationships from this text:\n\n{text}"

        try:
            result = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )

            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            # Validate structure
            valid_entities = []
            for e in entities:
                if isinstance(e, dict) and "name" in e and "type" in e:
                    valid_entities.append({
                        "name": str(e["name"]).strip(),
                        "type": str(e["type"]).strip(),
                        "summary": str(e.get("summary", "")).strip(),
                    })

            valid_rels = []
            for r in relationships:
                if isinstance(r, dict) and all(k in r for k in ("source", "target", "name", "fact")):
                    valid_rels.append({
                        "source": str(r["source"]).strip(),
                        "target": str(r["target"]).strip(),
                        "name": str(r["name"]).strip(),
                        "fact": str(r["fact"]).strip(),
                    })

            logger.debug(
                f"Extracted {len(valid_entities)} entities, {len(valid_rels)} relationships "
                f"from text ({len(text)} chars)"
            )

            return {"entities": valid_entities, "relationships": valid_rels}

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {"entities": [], "relationships": []}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_entity_types(entity_types: List[Dict]) -> str:
        lines = []
        for et in entity_types:
            name = et.get("name", "?")
            desc = et.get("description", "")
            attrs = [a.get("name", "") for a in et.get("attributes", [])]
            attr_str = f" (attributes: {', '.join(attrs)})" if attrs else ""
            lines.append(f"- {name}: {desc}{attr_str}")
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _format_edge_types(edge_types: List[Dict]) -> str:
        lines = []
        for et in edge_types:
            name = et.get("name", "?")
            desc = et.get("description", "")
            sts = et.get("source_targets", [])
            st_str = ""
            if sts:
                pairs = [f"{st.get('source','?')}->{st.get('target','?')}" for st in sts]
                st_str = f" [{', '.join(pairs)}]"
            lines.append(f"- {name}: {desc}{st_str}")
        return "\n".join(lines) if lines else "(none)"
