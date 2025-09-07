from dataclasses import dataclass
from typing import List

@dataclass
class Query:
    question: str
    target_responses: List[str]


@dataclass
class LLMOutput:
    question: str
    target_responses: List[str]
    llm: str
