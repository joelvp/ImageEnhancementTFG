from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LlamaResponse:
    tasks: Optional[List[str]] = None
    new_background_image: Optional[str] = None
    response_text: Optional[str] = None
    error_message: Optional[str] = None
