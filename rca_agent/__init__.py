"""rca_agent — LLM-core Root Cause Analysis agent.

Top-level package. Implementation modules live under subpackages:
  * ``rca_agent.contracts`` — frozen integration contracts (Pydantic + Protocols)
  * ``rca_agent.providers`` — DataProvider backends (parquet, clickhouse) + loader
  * ``rca_agent.llm``      — DeepSeek thinking-mode streaming client
  * ``rca_agent.memory``   — agent memory store
  * ``rca_agent.context``  — context manager (reasoning_content echo + compress)
  * ``rca_agent.tools``    — agent tools wrapping DataProvider + MemoryStore
  * ``rca_agent.agent``    — RCA agent core (ReAct loop)
  * ``rca_agent.server``   — FastAPI SSE server
  * ``rca_agent.store``    — MySQL persistence
  * ``rca_agent.observability`` — OpenTelemetry instrumentation
  * ``rca_agent.eval``     — benchmark / evaluation runner
"""

__version__ = "0.1.0"
