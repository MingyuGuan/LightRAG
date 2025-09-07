import instructor
from typing import TypeVar

from evaluation.prompts import SYSTEM_PROMPT

T = TypeVar("T")

class InstructorClient:
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name
        self.client = instructor.from_provider(model_name, async_client=True)

    async def __call__(self, prompt: str, response_model: T, *args, **kwds) -> T:
        return await self.client.chat.completions.create(
            response_model=response_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
