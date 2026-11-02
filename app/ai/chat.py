
import asyncio
import json
import logging
import random

from collections.abc import AsyncIterator

from app.ai.rag.pipeline import rag_pipeline
from app.ai.models import models

from app.ai.prompts.gemini_prompt import (
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
)


logger = logging.getLogger("uvicorn.error")

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything about that in the uploaded documents."
)


class ChatService:

    @staticmethod
    async def retry_with_jitter(
        func,
        *args,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
        **kwargs
    ):

        for attempt in range(1, max_retries + 1):

            try:
                return await func(*args, **kwargs)

            except Exception:

                if attempt >= max_retries:
                    logger.exception(
                        "Model request failed after all retries."
                    )
                    raise

                backoff = min(
                    max_delay,
                    base_delay * (2 ** (attempt - 1))
                )

                sleep_time = random.uniform(0, backoff)

                logger.warning(
                    f"Retry {attempt}/{max_retries} "
                    f"in {sleep_time:.2f} seconds."
                )

                await asyncio.sleep(sleep_time)


    @staticmethod
    async def _open_stream(
        model_func,
        prompt: str
    ) -> tuple[str, AsyncIterator[str]]:

        stream = model_func(prompt)

        first_token = await anext(stream)

        return first_token, stream


    @staticmethod
    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"


    @staticmethod
    async def chat(user_prompt: str) -> AsyncIterator[str]:

        try:

            # 1. Retrieve relevant document chunks
            retrieval = await rag_pipeline.retrieve(user_prompt)

            if not retrieval["results"]:

                yield ChatService._sse({
                    "type": "token",
                    "content": NO_CONTEXT_MESSAGE,
                })

                yield ChatService._sse({
                    "type": "sources",
                    "sources": [],
                })

                yield ChatService._sse({
                    "type": "done",
                })

                return


            # 2. Build the RAG prompt
            context, sources = rag_pipeline.build_context(
                retrieval["results"]
            )

            prompt = RAG_PROMPT_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                context=context,
                question=user_prompt,
            )


            # 3. Try available LLM providers
            fallback_chain = models.get_fallback_chain()

            for model_name, model_func in fallback_chain:

                logger.info(
                    f"Trying LLM provider: {model_name}"
                )

                try:

                    first_token, stream = (
                        await ChatService.retry_with_jitter(
                            ChatService._open_stream,
                            model_func,
                            prompt,
                        )
                    )

                except Exception:

                    logger.exception(
                        f"{model_name} failed. "
                        "Trying next provider."
                    )

                    continue


                # 4. Notify selected provider
                yield ChatService._sse({
                    "type": "model",
                    "name": model_name,
                })


                # 5. Stream the first token
                yield ChatService._sse({
                    "type": "token",
                    "content": first_token,
                })


                # 6. Stream remaining tokens
                try:

                    async for token in stream:

                        yield ChatService._sse({
                            "type": "token",
                            "content": token,
                        })


                except Exception:

                    logger.exception(
                        f"{model_name} failed during streaming."
                    )

                    yield ChatService._sse({
                        "type": "error",
                        "message": "The model stream was interrupted.",
                    })

                    return


                # 7. Send document sources
                yield ChatService._sse({
                    "type": "sources",
                    "sources": sources,
                })


                # 8. Finish stream
                yield ChatService._sse({
                    "type": "done",
                })

                return


            # All model providers failed
            yield ChatService._sse({
                "type": "error",
                "message": (
                    "All LLM providers failed. "
                    "Check API credentials and provider availability."
                ),
            })

        except Exception:

            logger.exception("Chat request failed.")

            yield ChatService._sse({
                "type": "error",
                "message": "An error occurred while processing your question.",
            })


chat_service = ChatService()