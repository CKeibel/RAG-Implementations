import json

from app.config import settings
from app.services.service_mixin import ServiceMixin
from openai import AsyncOpenAI


class VLLMService(ServiceMixin):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(base_url=settings.VLLM_URL, api_key="vllm")
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        base64_images: list[str] | None = None,
    ) -> dict:

        user_message_content = [{"type": "text", "text": user_prompt}]
        if base64_images:
            for img in base64_images:
                user_message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                )

        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message_content},
                ],
                max_tokens=self.max_tokens,
                temperature=self.max_tokens,
                response_format={"type": "json_object"},
            )

            if response.choices[0].message.content:
                return json.loads(response.choices[0].message.content)

        except Exception as e:
            self._handle_error(e)
