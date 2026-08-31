"""
Schemas Pydantic para os endpoints da API.
"""

from pydantic import BaseModel
from typing import Optional

class Prompt(BaseModel):
    image: str
    size: Optional[int] = None
    quality: Optional[int] = None
    verify_crops: bool = True
    include_token_usage: bool = False  # ← novo; padrão False para não quebrar clientes atuais

class PromptSys(BaseModel):
    """Payload com prompt de sistema customizado."""
    image: str
    prompt: str
