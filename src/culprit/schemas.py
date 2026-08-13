"""Canonical trace/span/step domain model. Every workstream imports from here,
so nothing in this module changes shape once other workstreams start coding
against it.

Three decisions here are easy to "fix" by mistake and must not be:

1. `Span.payload` is a plain union (`LlmPayload | ToolPayload | ... | None`),
   not a `Field(discriminator=...)` discriminated union. Pydantic's
   discriminated unions need the tag *inside* the member model, but the tag
   (`SpanKind`) already lives on the sibling `kind` field. Duplicating it onto
   every payload just to satisfy the discriminator would be redundant state
   that can drift from `kind`. Resolution by `kind` happens in `normalize.py`.

2. `Span.attributes` is always retained, never pruned here. A span whose
   vocabulary we cannot interpret today (`normalize_error` set, `payload`
   `None`) still carries every raw attribute, so a later vocabulary module can
   backfill it without re-ingesting. Retention-policy pruning is a storage
   concern (`spans.attributes_pruned` in Postgres), not a schema concern.

3. `Trace` deliberately has no `task_embedding` field. It is a 384-float
   storage concern that would bloat every API response and log line if it
   rode along on the Pydantic model; it exists only as a Postgres column,
   populated and queried by the L2 workstream directly against the database.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SpanKind(StrEnum):
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    AGENT = "agent"
    CHAIN = "chain"
    EMBEDDING = "embedding"
    GUARDRAIL = "guardrail"
    UNKNOWN = "unknown"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class Message(BaseModel):
    role: str
    content: str


class ToolCallRequest(BaseModel):
    call_id: str
    tool_name: str
    arguments: dict[str, Any]


class RetrievedDoc(BaseModel):
    doc_id: str
    text: str
    score: float | None = None


class LlmPayload(BaseModel):
    provider: str
    model: str
    request_messages: list[Message]
    response_messages: list[Message]
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    temperature: float | None = None
    response_format_schema: dict[str, Any] | None = None


class ToolPayload(BaseModel):
    tool_name: str
    call_id: str
    arguments_json: str
    arguments: dict[str, Any]
    # result_len is stored rather than derived from result_text because the
    # empty_tool_result detector (L1) must still fire after a retention
    # policy prunes result_text but keeps the scalar fields.
    result_text: str
    result_len: int
    is_error: bool
    error_message: str | None = None


class RetrievalPayload(BaseModel):
    query: str
    documents: list[RetrievedDoc]
    top_k: int


class AgentPayload(BaseModel):
    agent_name: str
    role: str
    input_text: str
    output_text: str
    delegated_to: str | None = None


class Span(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    status: SpanStatus
    status_message: str | None
    start_ns: int
    end_ns: int
    vocabulary: str
    # Raw attributes are never discarded here: normalization stays reversible
    # until a retention policy explicitly prunes them (see module docstring).
    attributes: dict[str, Any]
    payload: LlmPayload | ToolPayload | RetrievalPayload | AgentPayload | None
    normalize_error: str | None = None


class Trace(BaseModel):
    trace_id: str
    source: str
    outcome: Outcome
    # agent_key/task_key/task_goal/framework/root_span_id are resolved during
    # normalization and can legitimately be unset while ingest_error explains
    # why (a partially-ingested trace is still a valid Trace, not a crash).
    agent_key: str | None = None
    task_key: str | None = None
    task_goal: str | None = None
    framework: str | None = None
    root_span_id: str | None = None
    span_count: int
    step_count: int
    started_at: datetime | None = None
    ended_at: datetime | None = None
    ingested_at: datetime | None = None
    # Orphan spans are re-parented to root, never dropped, and their ids are
    # recorded here as metadata["orphan_span_ids"] (see linearize.py).
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingest_error: str | None = None


class Step(BaseModel):
    trace_id: str
    step_index: int
    span_id: str
    kind: SpanKind
    actor: str
    depth: int
    tree_path: str
    signature: str
    summary: str
    start_ns: int
    end_ns: int
    duration_ms: float
    # Ids of CHAIN/EMBEDDING/GUARDRAIL/UNKNOWN spans folded into this step by
    # linearize.py, so framework noise never inflates step_index (see plan
    # "Linearization, three rules that all matter", rule 2).
    collapsed_span_ids: list[str] = Field(default_factory=list)
