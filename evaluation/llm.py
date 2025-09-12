import instructor
from openai import AsyncOpenAI
from typing import TypeVar

from evaluation.config import ModelProxy
from evaluation.prompts import SYSTEM_PROMPT

T = TypeVar("T")

# Judge LLM client
class InstructorClient:
    def __init__(self, model_name: str = "openai/gpt-4o-mini", model_proxy: ModelProxy = None):
        self.model_name = model_name
        self.client = instructor.from_openai(AsyncOpenAI(
            api_key=model_proxy.api_key,
            base_url=model_proxy.base_url
        ))

    async def __call__(self, prompt: str, response_model: T, *args, **kwds) -> T:
        return await self.client.chat.completions.create(
            model=self.model_name,
            response_model=response_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
