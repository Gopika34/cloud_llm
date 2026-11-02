import logging
from typing import TypeVar, Callable, Awaitable

logger = logging.getLogger("uvicorn.error")

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:

    @staticmethod
    async def process_in_batches(
        items: list[T],
        process_batch: Callable[[list[T]], Awaitable[list[R]]],
        batch_size: int = 10
    ) -> list[R]:
        
        if not items:
            return []

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        all_results = []

        total_items = len(items)
        total_batches = (total_items + batch_size - 1) // batch_size

        for start in range(0, total_items, batch_size):

            batch_number = (start // batch_size) + 1
            batch = items[start:start + batch_size]

            logger.info(
                f"Processing batch {batch_number}/{total_batches} "
                f"({len(batch)} items)"
            )

            batch_results = await process_batch(batch)

            if len(batch_results) != len(batch):
                raise ValueError(
                    f"Batch {batch_number}: expected {len(batch)} results, "
                    f"but received {len(batch_results)}."
                )

            all_results.extend(batch_results)

        logger.info(
            f"Batch processing completed: {total_items} items "
            f"across {total_batches} batches."
        )

        return all_results