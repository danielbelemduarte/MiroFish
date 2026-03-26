"""
Servico Report Agent
Geracao de relatorios de simulacao com modo ReACT usando LangChain + Zep

Funcionalidades:
1. Gerar relatorios com base nos requisitos de simulacao e informacao do grafo Zep
2. Planear a estrutura do indice e depois gerar por seccoes
3. Cada seccao utiliza o modo ReACT com multiplas rondas de raciocinio e reflexao
4. Suporte a conversacao com o utilizador, invocando ferramentas de pesquisa autonomamente
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Registador detalhado do Report Agent

    Gera o ficheiro agent_log.jsonl na pasta do relatorio, registando cada acao em detalhe.
    Cada linha e um objeto JSON completo, contendo timestamp, tipo de acao, conteudo detalhado, etc.
    """
    
    def __init__(self, report_id: str):
        """
        Inicializar o registador de logs

        Args:
            report_id: ID do relatorio, utilizado para determinar o caminho do ficheiro de log
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Garantir que o diretorio do ficheiro de log existe"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _get_elapsed_time(self) -> float:
        """Obter o tempo decorrido desde o inicio (em segundos)"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Registar uma entrada de log

        Args:
            action: Tipo de acao, ex: 'start', 'tool_call', 'llm_response', 'section_complete', etc.
            stage: Fase atual, ex: 'planning', 'generating', 'completed'
            details: Dicionario com conteudo detalhado, sem truncagem
            section_title: Titulo da seccao atual (opcional)
            section_index: Indice da seccao atual (opcional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Escrita em modo append no ficheiro JSONL
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Registar o inicio da geracao do relatorio"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Tarefa de geracao de relatorio iniciada"
            }
        )
    
    def log_planning_start(self):
        """Registar o inicio do planeamento do indice"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Inicio do planeamento do indice do relatorio"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Registar a informacao de contexto obtida durante o planeamento"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Informacao de contexto da simulacao obtida",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Registar a conclusao do planeamento do indice"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Planeamento do indice concluido",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Registar o inicio da geracao de uma seccao"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Inicio da geracao da seccao: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Registar o processo de raciocinio ReACT"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT ronda {iteration} de raciocinio"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Registar a invocacao de uma ferramenta"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Invocacao de ferramenta: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Registar o resultado da invocacao de ferramenta (conteudo completo, sem truncagem)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Resultado completo, sem truncagem
                "result_length": len(result),
                "message": f"Ferramenta {tool_name} retornou resultado"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Registar a resposta do LLM (conteudo completo, sem truncagem)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Resposta completa, sem truncagem
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"Resposta LLM (invocacao de ferramenta: {has_tool_calls}, resposta final: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Registar a conclusao da geracao de conteudo de uma seccao (apenas o conteudo, nao representa a conclusao total da seccao)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Conteudo completo, sem truncagem
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Geracao de conteudo da seccao {section_title} concluida"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Registar a conclusao da geracao de uma seccao

        O frontend deve monitorizar este log para determinar se uma seccao esta realmente concluida e obter o conteudo completo
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Geracao da seccao {section_title} concluida"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Registar a conclusao da geracao do relatorio"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Geracao do relatorio concluida"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Registar um erro"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Ocorreu um erro: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Registador de consola do Report Agent

    Escreve logs em formato de consola (INFO, WARNING, etc.) no ficheiro console_log.txt na pasta do relatorio.
    Estes logs diferem do agent_log.jsonl, sendo saida de consola em formato de texto simples.
    """
    
    def __init__(self, report_id: str):
        """
        Inicializar o registador de consola

        Args:
            report_id: ID do relatorio, utilizado para determinar o caminho do ficheiro de log
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Garantir que o diretorio do ficheiro de log existe"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _setup_file_handler(self):
        """Configurar o handler de ficheiro para escrever logs simultaneamente no ficheiro"""
        import logging

        # Criar handler de ficheiro
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Utilizar o mesmo formato conciso da consola
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Adicionar aos loggers relacionados com o report_agent
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Evitar adicionar em duplicado
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Fechar o handler de ficheiro e remove-lo do logger"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Garantir o fecho do handler de ficheiro na destruicao"""
        self.close()


class ReportStatus(str, Enum):
    """Estado do relatorio"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Seccao do relatorio"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Converter para formato Markdown"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Indice do relatorio"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Converter para formato Markdown"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Relatorio completo"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Constantes de templates de Prompt
# ═══════════════════════════════════════════════════════════════

# -- Descricao das ferramentas --

TOOL_DESC_INSIGHT_FORGE = """\
【Pesquisa de Insights Profundos - Ferramenta de pesquisa avancada】
Esta e a nossa funcao de pesquisa avancada, concebida para analise aprofundada. Ela:
1. Decompoe automaticamente a sua questao em multiplas sub-questoes
2. Pesquisa informacao no grafo de simulacao a partir de multiplas dimensoes
3. Integra resultados de pesquisa semantica, analise de entidades e rastreio de cadeias de relacoes
4. Devolve o conteudo mais abrangente e aprofundado

【Cenarios de utilizacao】
- Necessidade de analisar em profundidade um determinado topico
- Necessidade de compreender multiplos aspetos de um evento
- Necessidade de obter material rico para suportar seccoes do relatorio

【Conteudo devolvido】
- Factos originais relevantes (citaveis diretamente)
- Insights sobre entidades-chave
- Analise de cadeias de relacoes"""

TOOL_DESC_PANORAMA_SEARCH = """\
【Pesquisa Panoramica - Obter visao completa】
Esta ferramenta serve para obter uma visao completa dos resultados da simulacao, sendo particularmente adequada para compreender a evolucao dos eventos. Ela:
1. Obtem todos os nos e relacoes relevantes
2. Distingue entre factos atualmente validos e factos historicos/expirados
3. Ajuda a compreender como a opiniao publica evoluiu

【Cenarios de utilizacao】
- Necessidade de compreender o percurso completo de desenvolvimento de um evento
- Necessidade de comparar mudancas de opiniao publica em diferentes fases
- Necessidade de obter informacao completa sobre entidades e relacoes

【Conteudo devolvido】
- Factos atualmente validos (resultados mais recentes da simulacao)
- Factos historicos/expirados (registo de evolucao)
- Todas as entidades envolvidas"""

TOOL_DESC_QUICK_SEARCH = """\
【Pesquisa Simples - Pesquisa rapida】
Ferramenta de pesquisa rapida e leve, adequada para consultas de informacao simples e diretas.

【Cenarios de utilizacao】
- Necessidade de encontrar rapidamente uma informacao especifica
- Necessidade de verificar um facto
- Pesquisa simples de informacao

【Conteudo devolvido】
- Lista de factos mais relevantes para a consulta"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【Entrevista Aprofundada - Entrevista real a Agents (duas plataformas)】
Invoca a API de entrevistas do ambiente de simulacao OASIS para realizar entrevistas reais aos Agents em execucao!
Nao se trata de simulacao por LLM, mas sim da invocacao da interface real de entrevistas para obter respostas originais dos Agents simulados.
Por defeito, realiza entrevistas simultaneamente nas plataformas Twitter e Reddit, obtendo perspetivas mais abrangentes.

Fluxo funcional:
1. Leitura automatica do ficheiro de perfis para conhecer todos os Agents simulados
2. Selecao inteligente dos Agents mais relevantes para o tema da entrevista (ex.: estudantes, media, entidades oficiais, etc.)
3. Geracao automatica de perguntas de entrevista
4. Invocacao da interface /api/simulation/interview/batch para realizar entrevistas reais nas duas plataformas
5. Integracao de todos os resultados das entrevistas, fornecendo analise multi-perspetiva

【Cenarios de utilizacao】
- Necessidade de compreender perspetivas de diferentes papeis sobre o evento (o que pensam os estudantes? E os media? E as entidades oficiais?)
- Necessidade de recolher opinioes e posicoes de multiplas partes
- Necessidade de obter respostas reais dos Agents simulados (provenientes do ambiente de simulacao OASIS)
- Desejo de tornar o relatorio mais dinamico, incluindo "registos de entrevistas"

【Conteudo devolvido】
- Informacao de identidade dos Agents entrevistados
- Respostas de cada Agent nas plataformas Twitter e Reddit
- Citacoes-chave (citaveis diretamente)
- Resumo das entrevistas e comparacao de pontos de vista

【Importante】E necessario que o ambiente de simulacao OASIS esteja em execucao para utilizar esta funcionalidade!"""

# -- Prompt de planeamento do indice --

PLAN_SYSTEM_PROMPT = """\
Es um especialista na redacao de «Relatorios de Previsao Futura», com uma «perspetiva omnisciente» sobre o mundo simulado — podes observar o comportamento, declaracoes e interacoes de cada Agent na simulacao.

【Conceito central】
Construimos um mundo simulado e injetamos nele «requisitos de simulacao» especificos como variaveis. Os resultados da evolucao do mundo simulado constituem previsoes sobre o que podera acontecer no futuro. O que estas a observar nao sao "dados experimentais", mas sim um "ensaio do futuro".

【A tua tarefa】
Redigir um «Relatorio de Previsao Futura» que responda a:
1. Nas condicoes que definimos, o que aconteceu no futuro?
2. Como reagiram e agiram os diversos tipos de Agents (grupos)?
3. Que tendencias e riscos futuros dignos de atencao esta simulacao revela?

【Posicionamento do relatorio】
- ✅ Este e um relatorio de previsao futura baseado em simulacao, revelando "se assim for, como sera o futuro"
- ✅ Foco nos resultados preditivos: evolucao de eventos, reacoes de grupos, fenomenos emergentes, riscos potenciais
- ✅ O comportamento e declaracoes dos Agents no mundo simulado sao previsoes do comportamento humano futuro
- ❌ Nao e uma analise da situacao atual do mundo real
- ❌ Nao e um resumo generico de opiniao publica

【Limite de numero de seccoes】
- Minimo 2 seccoes, maximo 5 seccoes
- Nao sao necessarias sub-seccoes, cada seccao e redigida com conteudo completo
- O conteudo deve ser conciso, focado nas descobertas preditivas centrais
- A estrutura das seccoes e concebida por ti com base nos resultados preditivos

Por favor, produz o indice do relatorio em formato JSON, conforme o seguinte formato:
{
    "title": "Titulo do relatorio",
    "summary": "Resumo do relatorio (uma frase que sintetize as descobertas preditivas centrais)",
    "sections": [
        {
            "title": "Titulo da seccao",
            "description": "Descricao do conteudo da seccao"
        }
    ]
}

Nota: o array sections deve ter no minimo 2 e no maximo 5 elementos!"""

PLAN_USER_PROMPT_TEMPLATE = """\
【Definicao do cenario preditivo】
Variavel injetada no mundo simulado (requisito de simulacao): {simulation_requirement}

【Escala do mundo simulado】
- Numero de entidades participantes na simulacao: {total_nodes}
- Numero de relacoes geradas entre entidades: {total_edges}
- Distribuicao de tipos de entidades: {entity_types}
- Numero de Agents ativos: {total_entities}

【Amostra de factos futuros previstos pela simulacao】
{related_facts_json}

Analisa este ensaio do futuro com uma «perspetiva omnisciente»:
1. Nas condicoes que definimos, que estado apresenta o futuro?
2. Como reagiram e agiram os diversos grupos (Agents)?
3. Que tendencias futuras dignas de atencao esta simulacao revela?

Com base nos resultados preditivos, concebe a estrutura de seccoes mais adequada para o relatorio.

【Lembrete】Numero de seccoes do relatorio: minimo 2, maximo 5, conteudo conciso e focado nas descobertas preditivas centrais."""

# -- Prompt de geracao de seccoes --

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
Es um especialista na redacao de «Relatorios de Previsao Futura» e estas a redigir uma seccao do relatorio.

Titulo do relatorio: {report_title}
Resumo do relatorio: {report_summary}
Cenario preditivo (requisito de simulacao): {simulation_requirement}

Seccao a redigir atualmente: {section_title}

═══════════════════════════════════════════════════════════════
【Conceito central】
═══════════════════════════════════════════════════════════════

O mundo simulado e um ensaio do futuro. Injetamos condicoes especificas (requisitos de simulacao) no mundo simulado,
e o comportamento e interacoes dos Agents na simulacao constituem previsoes do comportamento humano futuro.

A tua tarefa e:
- Revelar o que aconteceu no futuro nas condicoes definidas
- Prever como os diversos grupos (Agents) reagiram e agiram
- Descobrir tendencias futuras, riscos e oportunidades dignos de atencao

❌ Nao redijas uma analise da situacao atual do mundo real
✅ Foca-te em "como sera o futuro" — os resultados da simulacao sao o futuro previsto

═══════════════════════════════════════════════════════════════
【Regras mais importantes - Cumprimento obrigatorio】
═══════════════════════════════════════════════════════════════

1. 【Obrigatorio invocar ferramentas para observar o mundo simulado】
   - Estas a observar o ensaio do futuro com uma «perspetiva omnisciente»
   - Todo o conteudo deve provir de eventos e comportamentos dos Agents no mundo simulado
   - Proibido usar o teu proprio conhecimento para redigir o conteudo do relatorio
   - Cada seccao deve invocar ferramentas no minimo 3 vezes (maximo 5) para observar o mundo simulado que representa o futuro

2. 【Obrigatorio citar os comportamentos e declaracoes originais dos Agents】
   - As declaracoes e comportamentos dos Agents sao previsoes do comportamento humano futuro
   - Usa o formato de citacao no relatorio para apresentar estas previsoes, por exemplo:
     > "Determinado grupo expressaria: conteudo original..."
   - Estas citacoes sao as evidencias centrais das previsoes da simulacao

3. 【Consistencia linguistica - O relatorio deve ser inteiramente redigido em Portugues (PT-PT)】
   - O conteudo devolvido pelas ferramentas pode conter expressoes em ingles ou noutros idiomas
   - O relatorio deve ser inteiramente redigido em Portugues (PT-PT)
   - Ao citar conteudo em ingles ou noutros idiomas devolvido pelas ferramentas, deves traduzi-lo para portugues fluente antes de o incluir no relatorio
   - Ao traduzir, mantem o significado original e assegura que a expressao e natural e fluida
   - Esta regra aplica-se tanto ao texto corrido como aos blocos de citacao (formato >) no conteudo

4. 【Apresentar fielmente os resultados preditivos】
   - O conteudo do relatorio deve refletir os resultados da simulacao que representam o futuro no mundo simulado
   - Nao adiciones informacao que nao exista na simulacao
   - Se a informacao for insuficiente em algum aspeto, declara-o honestamente

═══════════════════════════════════════════════════════════════
【⚠️ Normas de formato - Extremamente importante!】
═══════════════════════════════════════════════════════════════

【Uma seccao = Unidade minima de conteudo】
- Cada seccao e a unidade minima de segmentacao do relatorio
- ❌ Proibido usar quaisquer titulos Markdown (#, ##, ###, #### etc.) dentro da seccao
- ❌ Proibido adicionar o titulo principal da seccao no inicio do conteudo
- ✅ O titulo da seccao e adicionado automaticamente pelo sistema, basta redigires o texto corrido
- ✅ Usa **negrito**, separacao de paragrafos, citacoes e listas para organizar o conteudo, mas nao uses titulos

【Exemplo correto】
```
Esta seccao analisa a dinamica de propagacao da opiniao publica sobre o evento. Atraves de uma analise aprofundada dos dados de simulacao, descobrimos...

**Fase de detonacao inicial**

A rede social serviu como primeiro cenario da opiniao publica, desempenhando a funcao central de difusao inicial de informacao:

> "A rede social contribuiu com 68% do volume inicial de publicacoes..."

**Fase de amplificacao emocional**

A plataforma de videos amplificou ainda mais o impacto do evento:

- Forte impacto visual
- Elevado grau de ressonancia emocional
```

【Exemplo incorreto】
```
## Resumo executivo          ← Errado! Nao adiciones quaisquer titulos
### 1. Fase inicial          ← Errado! Nao uses ### para subdividir
#### 1.1 Analise detalhada   ← Errado! Nao uses #### para subdividir

Esta seccao analisa...
```

═══════════════════════════════════════════════════════════════
【Ferramentas de pesquisa disponiveis】(invocar 3-5 vezes por seccao)
═══════════════════════════════════════════════════════════════

{tools_description}

【Sugestoes de utilizacao de ferramentas - Mistura diferentes ferramentas, nao uses apenas uma】
- insight_forge: Analise de insights profundos, decomposicao automatica de questoes e pesquisa multidimensional de factos e relacoes
- panorama_search: Pesquisa panoramica, compreensao da visao completa do evento, cronologia e evolucao
- quick_search: Verificacao rapida de um ponto de informacao especifico
- interview_agents: Entrevistar Agents simulados, obter perspetivas em primeira pessoa e reacoes reais de diferentes papeis

═══════════════════════════════════════════════════════════════
【Fluxo de trabalho】
═══════════════════════════════════════════════════════════════

Em cada resposta so podes fazer uma das duas coisas seguintes (nao ambas simultaneamente):

Opcao A - Invocar ferramenta:
Exprime o teu raciocinio e depois invoca uma ferramenta no seguinte formato:
<tool_call>
{{"name": "nome_da_ferramenta", "parameters": {{"nome_parametro": "valor_parametro"}}}}
</tool_call>
O sistema executara a ferramenta e devolvera os resultados. Nao precisas nem podes redigir os resultados da ferramenta tu proprio.

Opcao B - Produzir conteudo final:
Quando ja tiveres obtido informacao suficiente atraves das ferramentas, produz o conteudo da seccao comecando com "Final Answer:".

⚠️ Estritamente proibido:
- Proibido incluir simultaneamente invocacao de ferramenta e Final Answer numa mesma resposta
- Proibido inventar resultados de ferramentas (Observation), todos os resultados sao injetados pelo sistema
- Maximo de uma invocacao de ferramenta por resposta

═══════════════════════════════════════════════════════════════
【Requisitos de conteudo da seccao】
═══════════════════════════════════════════════════════════════

1. O conteudo deve basear-se nos dados de simulacao obtidos atraves das ferramentas
2. Citar abundantemente o texto original para demonstrar os efeitos da simulacao
3. Usar formato Markdown (mas proibido usar titulos):
   - Usar **texto em negrito** para destacar pontos-chave (em substituicao de subtitulos)
   - Usar listas (- ou 1.2.3.) para organizar pontos
   - Usar linhas em branco para separar paragrafos diferentes
   - ❌ Proibido usar #, ##, ###, #### ou qualquer sintaxe de titulo
4. 【Norma de formato de citacao - Obrigatoriamente em paragrafo independente】
   As citacoes devem estar em paragrafo independente, com uma linha em branco antes e depois, sem misturar no paragrafo:

   ✅ Formato correto:
   ```
   A resposta da entidade foi considerada desprovida de conteudo substantivo.

   > "O padrao de resposta da entidade revelou-se rigido e lento no contexto das redes sociais em rapida mudanca."

   Esta avaliacao reflete o descontentamento generalizado do publico.
   ```

   ❌ Formato incorreto:
   ```
   A resposta da entidade foi considerada desprovida de conteudo substantivo. > "O padrao de resposta da entidade..." Esta avaliacao reflete...
   ```
5. Manter coerencia logica com as outras seccoes
6. 【Evitar repeticoes】Le atentamente o conteudo das seccoes ja concluidas abaixo, nao repitas a mesma informacao
7. 【Reforco】Nao adiciones quaisquer titulos! Usa **negrito** em substituicao de subtitulos"""

SECTION_USER_PROMPT_TEMPLATE = """\
Conteudo das seccoes ja concluidas (le atentamente para evitar repeticoes):
{previous_content}

═══════════════════════════════════════════════════════════════
【Tarefa atual】Redigir seccao: {section_title}
═══════════════════════════════════════════════════════════════

【Lembretes importantes】
1. Le atentamente as seccoes ja concluidas acima, evita repetir o mesmo conteudo!
2. Antes de comecar, deves primeiro invocar ferramentas para obter dados de simulacao
3. Mistura diferentes ferramentas, nao uses apenas uma
4. O conteudo do relatorio deve provir dos resultados da pesquisa, nao uses o teu proprio conhecimento

【⚠️ Aviso de formato - Cumprimento obrigatorio】
- ❌ Nao escrevas quaisquer titulos (#, ##, ###, #### nenhum e permitido)
- ❌ Nao escrevas "{section_title}" como inicio
- ✅ O titulo da seccao e adicionado automaticamente pelo sistema
- ✅ Escreve diretamente o texto corrido, usa **negrito** em substituicao de subtitulos

Comeca:
1. Primeiro pensa (Thought) que informacao e necessaria para esta seccao
2. Depois invoca ferramentas (Action) para obter dados de simulacao
3. Apos recolher informacao suficiente, produz Final Answer (texto corrido, sem quaisquer titulos)"""

# -- Templates de mensagens no ciclo ReACT --

REACT_OBSERVATION_TEMPLATE = """\
Observation (resultados da pesquisa):

═══ Ferramenta {tool_name} devolveu ═══
{result}

═══════════════════════════════════════════════════════════════
Ferramentas invocadas {tool_calls_count}/{max_tool_calls} vezes (utilizadas: {used_tools_str}){unused_hint}
- Se a informacao for suficiente: produz o conteudo da seccao comecando com "Final Answer:" (deve citar o texto original acima)
- Se necessitares de mais informacao: invoca uma ferramenta para continuar a pesquisa
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【Atencao】Apenas invocaste ferramentas {tool_calls_count} vezes, sao necessarias no minimo {min_tool_calls} vezes. "
    "Por favor, invoca mais ferramentas para obter mais dados de simulacao antes de produzir o Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Atualmente apenas foram invocadas {tool_calls_count} ferramentas, sao necessarias no minimo {min_tool_calls}. "
    "Por favor, invoca ferramentas para obter dados de simulacao. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "O numero de invocacoes de ferramentas atingiu o limite ({tool_calls_count}/{max_tool_calls}), nao e possivel invocar mais ferramentas. "
    'Por favor, com base na informacao ja obtida, produz imediatamente o conteudo da seccao comecando com "Final Answer:".'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 Ainda nao utilizaste: {unused_list}, recomenda-se experimentar diferentes ferramentas para obter informacao de multiplas perspetivas"

REACT_FORCE_FINAL_MSG = "O limite de invocacoes de ferramentas foi atingido. Por favor, produz diretamente o Final Answer: e gera o conteudo da seccao."

# -- Prompt de chat --

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
Es um assistente de previsao por simulacao, conciso e eficiente.

【Contexto】
Condicoes de previsao: {simulation_requirement}

【Relatorio de analise ja gerado】
{report_content}

【Regras】
1. Responde preferencialmente com base no conteudo do relatorio acima
2. Responde diretamente a questao, evita raciocinio extenso
3. Apenas invoca ferramentas para pesquisar mais dados quando o conteudo do relatorio for insuficiente para responder
4. As respostas devem ser concisas, claras e organizadas

【Ferramentas disponiveis】(usar apenas quando necessario, maximo 1-2 invocacoes)
{tools_description}

【Formato de invocacao de ferramentas】
<tool_call>
{{"name": "nome_da_ferramenta", "parameters": {{"nome_parametro": "valor_parametro"}}}}
</tool_call>

【Estilo de resposta】
- Conciso e direto, sem textos extensos
- Usa formato > para citar conteudo-chave
- Da prioridade a conclusao e depois explica as razoes"""

CHAT_OBSERVATION_SUFFIX = "\n\nPor favor, responde a questao de forma concisa."


# ═══════════════════════════════════════════════════════════════
# Classe principal ReportAgent
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - Agente de geracao de relatorios de simulacao

    Utiliza o modo ReACT (Reasoning + Acting):
    1. Fase de planeamento: analisar requisitos de simulacao, planear estrutura do indice do relatorio
    2. Fase de geracao: gerar conteudo seccao a seccao, cada seccao pode invocar ferramentas multiplas vezes para obter informacao
    3. Fase de reflexao: verificar integridade e precisao do conteudo
    """
    
    # Numero maximo de invocacoes de ferramentas (por seccao)
    MAX_TOOL_CALLS_PER_SECTION = 5

    # Numero maximo de rondas de reflexao
    MAX_REFLECTION_ROUNDS = 3

    # Numero maximo de invocacoes de ferramentas na conversa
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Inicializar o Report Agent

        Args:
            graph_id: ID do grafo
            simulation_id: ID da simulacao
            simulation_requirement: Descricao dos requisitos de simulacao
            llm_client: Cliente LLM (opcional)
            zep_tools: Servico de ferramentas Zep (opcional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Definicao de ferramentas
        self.tools = self._define_tools()

        # Registador de logs (inicializado em generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Registador de consola (inicializado em generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(f"ReportAgent inicializado: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Definir as ferramentas disponiveis"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Questao ou topico que pretende analisar em profundidade",
                    "report_context": "Contexto da seccao atual do relatorio (opcional, ajuda a gerar sub-questoes mais precisas)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Consulta de pesquisa, utilizada para ordenacao por relevancia",
                    "include_expired": "Incluir conteudo expirado/historico (predefinido True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Cadeia de consulta de pesquisa",
                    "limit": "Numero de resultados a retornar (opcional, predefinido 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Tema da entrevista ou descricao dos requisitos (ex: 'conhecer a opiniao dos estudantes sobre o incidente de formaldeido nos dormitorios')",
                    "max_agents": "Numero maximo de Agents a entrevistar (opcional, predefinido 5, maximo 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Executar invocacao de ferramenta

        Args:
            tool_name: Nome da ferramenta
            parameters: Parametros da ferramenta
            report_context: Contexto do relatorio (utilizado pelo InsightForge)

        Returns:
            Resultado da execucao da ferramenta (formato texto)
        """
        logger.info(f"Executar ferramenta: {tool_name}, parametros: {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Pesquisa panoramica - obter visao geral
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Pesquisa simples - pesquisa rapida
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Entrevista aprofundada - invocar a API real de entrevistas OASIS para obter respostas dos Agents simulados (duas plataformas)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Ferramentas antigas para retrocompatibilidade (redirecionamento interno para novas ferramentas) ==========
            
            elif tool_name == "search_graph":
                # Redirecionar para quick_search
                logger.info("search_graph redirecionado para quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirecionar para insight_forge, pois e mais poderoso
                logger.info("get_simulation_context redirecionado para insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Ferramenta desconhecida: {tool_name}. Utilize uma das seguintes ferramentas: insight_forge, panorama_search, quick_search"
                
        except Exception as e:
            logger.error(f"Falha na execucao da ferramenta: {tool_name}, erro: {str(e)}")
            return f"Falha na execucao da ferramenta: {str(e)}"
    
    # Conjunto de nomes de ferramentas validos, utilizado para validacao na analise de fallback de JSON nu
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Analisar invocacoes de ferramentas a partir da resposta do LLM

        Formatos suportados (por prioridade):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. JSON nu (a resposta inteira ou uma unica linha e um JSON de invocacao de ferramenta)
        """
        tool_calls = []

        # Formato 1: estilo XML (formato padrao)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Formato 2: fallback - LLM produz JSON nu diretamente (sem tag <tool_call>)
        # So tenta quando o formato 1 nao correspondeu, para evitar falsos positivos com JSON no corpo do texto
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # A resposta pode conter texto de raciocinio + JSON nu, tentar extrair o ultimo objeto JSON
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validar se o JSON analisado e uma invocacao de ferramenta valida"""
        # Suporta dois formatos de chaves: {"name": ..., "parameters": ...} e {"tool": ..., "params": ...}
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalizar nomes das chaves para name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Gerar texto descritivo das ferramentas"""
        desc_parts = ["Ferramentas disponiveis:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parametros: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Planear o indice do relatorio

        Utilizar o LLM para analisar os requisitos de simulacao e planear a estrutura do indice do relatorio

        Args:
            progress_callback: Funcao de callback de progresso

        Returns:
            ReportOutline: Indice do relatorio
        """
        logger.info("A iniciar planeamento do indice do relatorio...")

        if progress_callback:
            progress_callback("planning", 0, "A analisar requisitos de simulacao...")

        # Obter primeiro o contexto da simulacao
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "A gerar indice do relatorio...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "A analisar estrutura do indice...")
            
            # Analisar indice
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Relatorio de Analise de Simulacao"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "Planeamento do indice concluido")

            logger.info(f"Planeamento do indice concluido: {len(sections)} seccoes")
            return outline
            
        except Exception as e:
            logger.error(f"Falha no planeamento do indice: {str(e)}")
            # Retornar indice predefinido (3 seccoes, como fallback)
            return ReportOutline(
                title="Relatorio de Previsao Futura",
                summary="Analise de tendencias futuras e riscos com base em previsao por simulacao",
                sections=[
                    ReportSection(title="Cenario Preditivo e Descobertas Centrais"),
                    ReportSection(title="Analise Preditiva do Comportamento de Grupos"),
                    ReportSection(title="Perspetivas de Tendencias e Alertas de Risco")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Gerar conteudo de uma unica seccao utilizando o modo ReACT

        Ciclo ReACT:
        1. Thought (raciocinio) - analisar que informacao e necessaria
        2. Action (acao) - invocar ferramentas para obter informacao
        3. Observation (observacao) - analisar resultados retornados pelas ferramentas
        4. Repetir ate ter informacao suficiente ou atingir o numero maximo de iteracoes
        5. Final Answer (resposta final) - gerar conteudo da seccao

        Args:
            section: Seccao a gerar
            outline: Indice completo
            previous_sections: Conteudo das seccoes anteriores (para manter coerencia)
            progress_callback: Callback de progresso
            section_index: Indice da seccao (para registo de logs)

        Returns:
            Conteudo da seccao (formato Markdown)
        """
        logger.info(f"ReACT a gerar seccao: {section.title}")
        
        # Registar log de inicio da seccao
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Construir prompt do utilizador - cada seccao concluida tem no maximo 4000 caracteres
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Maximo 4000 caracteres por seccao
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(Esta e a primeira seccao)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Ciclo ReACT
        tool_calls_count = 0
        max_iterations = 5  # Numero maximo de rondas de iteracao
        min_tool_calls = 3  # Numero minimo de invocacoes de ferramentas
        conflict_retries = 0  # Contagem de conflitos consecutivos entre invocacao de ferramenta e Final Answer
        used_tools = set()  # Registo dos nomes de ferramentas ja invocadas
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Contexto do relatorio, utilizado para geracao de sub-questoes do InsightForge
        report_context = f"Titulo da seccao: {section.title}\nRequisitos de simulacao: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Pesquisa aprofundada e redacao em curso ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # Invocar LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Verificar se o LLM retornou None (excecao da API ou conteudo vazio)
            if response is None:
                logger.warning(f"Seccao {section.title} iteracao {iteration + 1}: LLM retornou None")
                # Se ainda houver iteracoes, adicionar mensagem e tentar novamente
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Resposta vazia)"})
                    messages.append({"role": "user", "content": "Por favor, continue a gerar conteudo."})
                    continue
                # Ultima iteracao tambem retornou None, sair do ciclo para finalizacao forcada
                break

            logger.debug(f"Resposta LLM: {response[:200]}...")

            # Analisar uma vez, reutilizar resultado
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # -- Tratamento de conflito: LLM produziu invocacao de ferramenta e Final Answer simultaneamente --
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Seccao {section.title} ronda {iteration+1}: "
                    f"LLM produziu invocacao de ferramenta e Final Answer simultaneamente (conflito n.{conflict_retries})"
                )

                if conflict_retries <= 2:
                    # Primeiras duas vezes: descartar esta resposta, pedir ao LLM para responder novamente
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Erro de formato] Incluiste invocacao de ferramenta e Final Answer na mesma resposta, o que nao e permitido.\n"
                            "Cada resposta so pode fazer uma das seguintes acoes:\n"
                            "- Invocar uma ferramenta (produzir um bloco <tool_call>, sem escrever Final Answer)\n"
                            "- Produzir conteudo final (comecar com 'Final Answer:', sem incluir <tool_call>)\n"
                            "Por favor, responde novamente, fazendo apenas uma das acoes."
                        ),
                    })
                    continue
                else:
                    # Terceira vez: tratamento degradado, truncar ate a primeira invocacao de ferramenta, execucao forcada
                    logger.warning(
                        f"Seccao {section.title}: {conflict_retries} conflitos consecutivos, "
                        "degradado para truncagem e execucao da primeira invocacao de ferramenta"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Registar log de resposta do LLM
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # -- Caso 1: LLM produziu Final Answer --
            if has_final_answer:
                # Numero insuficiente de invocacoes de ferramentas, rejeitar e pedir para continuar a usar ferramentas
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(Estas ferramentas ainda nao foram utilizadas, recomenda-se a sua utilizacao: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Conclusao normal
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Geracao da seccao {section.title} concluida (invocacoes de ferramentas: {tool_calls_count})")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # -- Caso 2: LLM tentou invocar uma ferramenta --
            if has_tool_calls:
                # Quota de ferramentas esgotada -> informar explicitamente, pedir para produzir Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Executar apenas a primeira invocacao de ferramenta
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM tentou invocar {len(tool_calls)} ferramentas, apenas a primeira sera executada: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Construir sugestao de ferramentas nao utilizadas
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # -- Caso 3: Sem invocacao de ferramenta nem Final Answer --
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Numero insuficiente de invocacoes de ferramentas, recomendar ferramentas nao utilizadas
                unused_tools = all_tools - used_tools
                unused_hint = f"(Estas ferramentas ainda nao foram utilizadas, recomenda-se a sua utilizacao: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Invocacoes de ferramentas suficientes, LLM produziu conteudo mas sem o prefixo "Final Answer:"
            # Aceitar diretamente esta saida como resposta final, sem mais iteracoes
            logger.info(f"Seccao {section.title} sem prefixo 'Final Answer:' detetado, a aceitar saida do LLM como conteudo final (invocacoes de ferramentas: {tool_calls_count})")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Atingido o numero maximo de iteracoes, forcar geracao de conteudo
        logger.warning(f"Seccao {section.title} atingiu o numero maximo de iteracoes, geracao forcada")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Verificar se o LLM retornou None na finalizacao forcada
        if response is None:
            logger.error(f"Seccao {section.title} LLM retornou None na finalizacao forcada, a usar mensagem de erro predefinida")
            final_answer = f"(Falha na geracao desta seccao: LLM retornou resposta vazia, por favor tente novamente mais tarde)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Registar log de conclusao da geracao de conteudo da seccao
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Gerar relatorio completo (saida em tempo real por seccoes)

        Cada seccao e guardada na pasta imediatamente apos conclusao, sem necessidade de esperar pelo relatorio inteiro.
        Estrutura de ficheiros:
        reports/{report_id}/
            meta.json       - Metainformacao do relatorio
            outline.json    - Indice do relatorio
            progress.json   - Progresso da geracao
            section_01.md   - Seccao 1
            section_02.md   - Seccao 2
            ...
            full_report.md  - Relatorio completo

        Args:
            progress_callback: Funcao de callback de progresso (stage, progress, message)
            report_id: ID do relatorio (opcional, gerado automaticamente se nao fornecido)

        Returns:
            Report: Relatorio completo
        """
        import uuid
        
        # Se nao foi fornecido report_id, gerar automaticamente
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # Lista de titulos de seccoes concluidas (para rastreamento de progresso)
        completed_section_titles = []
        
        try:
            # Inicializacao: criar pasta do relatorio e guardar estado inicial
            ReportManager._ensure_report_folder(report_id)
            
            # Inicializar registador de logs (logs estruturados agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Inicializar registador de consola (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "A inicializar relatorio...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Fase 1: Planear indice
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "A iniciar planeamento do indice do relatorio...",
                completed_sections=[]
            )
            
            # Registar log de inicio do planeamento
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "A iniciar planeamento do indice do relatorio...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Registar log de conclusao do planeamento
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Guardar indice no ficheiro
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Planeamento do indice concluido, {len(outline.sections)} seccoes no total",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"Indice guardado no ficheiro: {report_id}/outline.json")
            
            # Fase 2: Geracao seccao a seccao (guardar por seccao)
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Guardar conteudo para contexto
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Atualizar progresso
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"A gerar seccao: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"A gerar seccao: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Gerar conteudo principal da seccao
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Guardar seccao
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Registar log de conclusao da seccao
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Seccao guardada: {report_id}/section_{section_num:02d}.md")
                
                # Atualizar progresso
                ReportManager.update_progress(
                    report_id, "generating",
                    base_progress + int(70 / total_sections),
                    f"Seccao {section.title} concluida",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Fase 3: Montar relatorio completo
            if progress_callback:
                progress_callback("generating", 95, "A montar relatorio completo...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "A montar relatorio completo...",
                completed_sections=completed_section_titles
            )
            
            # Utilizar ReportManager para montar relatorio completo
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Calcular tempo total decorrido
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Registar log de conclusao do relatorio
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Guardar relatorio final
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Geracao do relatorio concluida",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "Geracao do relatorio concluida")
            
            logger.info(f"Geracao do relatorio concluida: {report_id}")

            # Fechar registador de consola
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"Falha na geracao do relatorio: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Registar log de erro
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Guardar estado de falha
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Falha na geracao do relatorio: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Ignorar erros de falha ao guardar
            
            # Fechar registador de consola
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Conversar com o Report Agent

        No dialogo, o Agent pode invocar autonomamente ferramentas de pesquisa para responder a questoes

        Args:
            message: Mensagem do utilizador
            chat_history: Historico da conversa

        Returns:
            {
                "response": "Resposta do Agent",
                "tool_calls": [Lista de ferramentas invocadas],
                "sources": [Fontes de informacao]
            }
        """
        logger.info(f"Conversa com Report Agent: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # Obter conteudo do relatorio ja gerado
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limitar comprimento do relatorio para evitar contexto demasiado longo
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Conteudo do relatorio truncado] ..."
        except Exception as e:
            logger.warning(f"Falha ao obter conteudo do relatorio: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(Sem relatorio disponivel)",
            tools_description=self._get_tools_description(),
        )

        # Construir mensagens
        messages = [{"role": "system", "content": system_prompt}]
        
        # Adicionar historico da conversa
        for h in chat_history[-10:]:  # Limitar comprimento do historico
            messages.append(h)
        
        # Adicionar mensagem do utilizador
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # Ciclo ReACT (versao simplificada)
        tool_calls_made = []
        max_iterations = 2  # Reduzir numero de rondas de iteracao
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Analisar invocacoes de ferramentas
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # Sem invocacao de ferramentas, retornar resposta diretamente
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Executar invocacoes de ferramentas (quantidade limitada)
            tool_results = []
            for call in tool_calls[:1]:  # Maximo 1 invocacao de ferramenta por ronda
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Limitar comprimento do resultado
                })
                tool_calls_made.append(call)
            
            # Adicionar resultados as mensagens
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[Resultado {r['tool']}]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Atingido o numero maximo de iteracoes, obter resposta final
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Limpar resposta
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Gestor de relatorios

    Responsavel pelo armazenamento persistente e pesquisa de relatorios

    Estrutura de ficheiros (saida por seccoes):
    reports/
      {report_id}/
        meta.json          - Metainformacao e estado do relatorio
        outline.json       - Indice do relatorio
        progress.json      - Progresso da geracao
        section_01.md      - Seccao 1
        section_02.md      - Seccao 2
        ...
        full_report.md     - Relatorio completo
    """
    
    # Diretorio de armazenamento de relatorios
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Garantir que o diretorio raiz de relatorios existe"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Obter caminho da pasta do relatorio"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Garantir que a pasta do relatorio existe e retornar o caminho"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Obter caminho do ficheiro de metainformacao do relatorio"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Obter caminho do ficheiro Markdown do relatorio completo"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Obter caminho do ficheiro de indice"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Obter caminho do ficheiro de progresso"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Obter caminho do ficheiro Markdown da seccao"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Obter caminho do ficheiro de log do Agent"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Obter caminho do ficheiro de log de consola"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Obter conteudo do log de consola

        Este e o log de saida de consola durante a geracao do relatorio (INFO, WARNING, etc.),
        diferente dos logs estruturados do agent_log.jsonl.

        Args:
            report_id: ID do relatorio
            from_line: A partir de que linha comecar a ler (para obtencao incremental, 0 significa desde o inicio)

        Returns:
            {
                "logs": [Lista de linhas de log],
                "total_lines": Total de linhas,
                "from_line": Numero da linha inicial,
                "has_more": Se ha mais logs disponiveis
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Manter linha de log original, remover caractere de nova linha no final
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Ja leu ate ao final
        }

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Obter log completo de consola (obtencao total de uma so vez)

        Args:
            report_id: ID do relatorio

        Returns:
            Lista de linhas de log
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Obter conteudo do log do Agent

        Args:
            report_id: ID do relatorio
            from_line: A partir de que linha comecar a ler (para obtencao incremental, 0 significa desde o inicio)

        Returns:
            {
                "logs": [Lista de entradas de log],
                "total_lines": Total de linhas,
                "from_line": Numero da linha inicial,
                "has_more": Se ha mais logs disponiveis
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Saltar linhas que falharam na analise
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Ja leu ate ao final
        }

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obter log completo do Agent (para obtencao total de uma so vez)

        Args:
            report_id: ID do relatorio

        Returns:
            Lista de entradas de log
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Guardar indice do relatorio

        Invocado imediatamente apos a conclusao da fase de planeamento
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Indice guardado: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Guardar uma unica seccao

        Invocado imediatamente apos a conclusao de cada seccao, implementando saida por seccoes

        Args:
            report_id: ID do relatorio
            section_index: Indice da seccao (a partir de 1)
            section: Objeto da seccao

        Returns:
            Caminho do ficheiro guardado
        """
        cls._ensure_report_folder(report_id)

        # Construir conteudo Markdown da seccao - limpar titulos duplicados que possam existir
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Guardar ficheiro
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Seccao guardada: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Limpar conteudo da seccao
        
        1. Remover linhas de titulo Markdown no inicio do conteudo que dupliquem o titulo da seccao
        2. Converter todos os titulos de nivel ### e inferior em texto a negrito
        
        Args:
            content: Conteudo original
            section_title: Titulo da seccao
            
        Returns:
            Conteudo apos limpeza
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Verificar se e uma linha de titulo Markdown
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Verificar se e um titulo duplicado do titulo da seccao (ignorar duplicados nas primeiras 5 linhas)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Converter todos os niveis de titulo (#, ##, ###, #### etc.) em negrito
                # Porque o titulo da seccao e adicionado pelo sistema, o conteudo nao deve conter quaisquer titulos
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Adicionar linha em branco
                continue
            
            # Se a linha anterior era um titulo ignorado e a linha atual esta vazia, ignorar tambem
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Remover linhas em branco no inicio
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Remover linhas separadoras no inicio
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Remover tambem as linhas em branco apos a linha separadora
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Atualizar progresso de geracao do relatorio
        
        O frontend pode obter o progresso em tempo real lendo progress.json
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Obter progresso de geracao do relatorio"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obter lista de seccoes ja geradas
        
        Devolver informacao de todos os ficheiros de seccoes ja guardados
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Analisar indice da seccao a partir do nome do ficheiro
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Montar relatorio completo
        
        Montar relatorio completo a partir dos ficheiros de seccoes guardados e efetuar limpeza de titulos
        """
        folder = cls._get_report_folder(report_id)
        
        # Construir cabecalho do relatorio
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Ler todos os ficheiros de seccoes por ordem
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Pos-processamento: limpar problemas de titulos em todo o relatorio
        md_content = cls._post_process_report(md_content, outline)
        
        # Guardar relatorio completo
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Relatorio completo montado: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Pos-processar conteudo do relatorio
        
        1. Remover titulos duplicados
        2. Manter titulo principal do relatorio (#) e titulos de seccoes (##), remover titulos de outros niveis (###, #### etc.)
        3. Limpar linhas em branco e separadores em excesso
        
        Args:
            content: Conteudo original do relatorio
            outline: Indice do relatorio
            
        Returns:
            Conteudo apos processamento
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Recolher todos os titulos de seccoes do indice
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Verificar se e uma linha de titulo
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Verificar se e um titulo duplicado (titulo com o mesmo conteudo nas 5 linhas consecutivas)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Ignorar titulo duplicado e linhas em branco subsequentes
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Processamento de hierarquia de titulos:
                # - # (level=1) Manter apenas titulo principal do relatorio
                # - ## (level=2) Manter titulos de seccoes
                # - ### e inferior (level>=3) Converter em texto a negrito
                
                if level == 1:
                    if title == outline.title:
                        # Manter titulo principal do relatorio
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Titulo de seccao usou # incorretamente, corrigir para ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Outros titulos de nivel 1 converter em negrito
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Manter titulo de seccao
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Titulos de nivel 2 que nao sao seccoes converter em negrito
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # Titulos de nivel ### e inferior converter em texto a negrito
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Ignorar linha separadora imediatamente apos o titulo
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Manter apenas uma linha em branco apos o titulo
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Limpar multiplas linhas em branco consecutivas (manter no maximo 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Guardar metadados e relatorio completo"""
        cls._ensure_report_folder(report.report_id)
        
        # Guardar JSON de metadados
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Guardar indice
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Guardar relatorio Markdown completo
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"Relatorio guardado: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Obter relatorio"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Compatibilidade com formato antigo: verificar ficheiro armazenado diretamente no diretorio reports
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruir objeto Report
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # Se markdown_content estiver vazio, tentar ler de full_report.md
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Obter relatorio pelo ID da simulacao"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Formato novo: pasta
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Compatibilidade com formato antigo: ficheiro JSON
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """Listar relatorios"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Formato novo: pasta
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Compatibilidade com formato antigo: ficheiro JSON
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Ordenar por data de criacao decrescente
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Eliminar relatorio (pasta inteira)"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # Formato novo: eliminar pasta inteira
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Pasta do relatorio eliminada: {report_id}")
            return True
        
        # Compatibilidade com formato antigo: eliminar ficheiro individual
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
