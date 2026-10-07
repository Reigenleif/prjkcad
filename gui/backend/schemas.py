from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str
    thought: Optional[str] = None
    dualseq_tuples: Optional[List[Any]] = None


class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[ChatMessage]] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 768


class ChatResponse(BaseModel):
    assistant_message: str
    thought: str
    dualseq_tuples: List[List[Any]]
    render_result: Optional[Dict[str, Any]] = None
    tree_data: Optional[Dict[str, Any]] = None
    raw_output: str
    error: Optional[str] = None


class RenderRequest(BaseModel):
    dualseq_tuples: List[List[Any]]
    name: Optional[str] = "cad_model"


class RenderResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    stl_url: Optional[str] = None
    png_url: Optional[str] = None
    tree_data: Dict[str, Any]
    properties: Dict[str, Any]
    dualseq_tuples: Optional[List[List[Any]]] = None


class SampleItem(BaseModel):
    uid: str
    label: str
    category: str
    prompt: str
    split: str
    note: Optional[str] = ""
    gt_cmds: List[str] = Field(default_factory=list)
    gt_args: List[Dict[str, float]] = Field(default_factory=list)


class ExportRequest(BaseModel):
    dualseq_tuples: List[List[Any]]
    format: str = "stl"
    filename: Optional[str] = "cad_model"
