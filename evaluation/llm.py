import instructor
from openai import AsyncOpenAI
from typing import TypeVar
import os

from evaluation.prompts import SYSTEM_PROMPT

T = TypeVar("T")

# Judge LLM client
class InstructorClient:
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name
        self.client = instructor.from_openai(AsyncOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("LITELLM_PROXY_URL")
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
