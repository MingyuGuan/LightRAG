from dataclasses import dataclass
from typing import override, Dict
from pydantic import BaseModel, Field

from evaluation.llm import InstructorClient
from evaluation.prompts import get_prompt
from evaluation.types import LLMOutput

class Metric(BaseModel):
    name: str

    async def compute(
        self,
        question: str,
        reference_answer: str,
        llm_answer: str,
        llm: InstructorClient,
    ) -> 'Metric':
        pass

    async def evaluate_metric(
        self, llm_output: LLMOutput, llm: InstructorClient
    ) -> Dict[str, 'Metric']:
        return {
            f"response_{idx}": await self.compute(llm_output.question, target_response, llm_output.llm, llm)
            for idx, target_response in enumerate(llm_output.target_responses)
        }

class JudgeLLMScore(BaseModel):
    comprehensiveness: int = Field(le=100)
    diversity: int = Field(le=100)
    empowerment: int = Field(le=100)

class JudgeLLM(Metric):

    @override
    async def compute(
        self,
        question: str,
        reference_answer: str,
        llm_answer: str,
        llm: InstructorClient,
    ) -> JudgeLLMScore:
        prompt = get_prompt(question, reference_answer, llm_answer)
        return await llm(prompt, JudgeLLMScore)

class MetricRegistry:
    def __init__(self):
        self.metrics = {}

    def register(self, metric: Metric):
        self.metrics[metric.name] = metric

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics
