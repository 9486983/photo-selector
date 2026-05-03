from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class _WorkItem:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: asyncio.Future


class GpuCpuScheduler:
    def __init__(self, gpu_workers: int = 1, cpu_workers: int = 4) -> None:
        self._gpu_workers = max(0, gpu_workers)
        self._cpu_workers = max(1, cpu_workers)
        self._gpu_queue: asyncio.Queue[_WorkItem] = asyncio.Queue()
        self._cpu_queue: asyncio.Queue[_WorkItem] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []
        self._started = False

    @property
    def queue_stats(self) -> dict[str, int]:
        return {
            "gpu_queue": self._gpu_queue.qsize(),
            "cpu_queue": self._cpu_queue.qsize(),
            "gpu_workers": self._gpu_workers,
            "cpu_workers": self._cpu_workers,
            "total_gpu_memory_mb": int(ModelRegistry.total_gpu_memory_mb()),
        }

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for _ in range(self._gpu_workers):
            self._worker_tasks.append(asyncio.create_task(self._worker(self._gpu_queue)))
        for _ in range(self._cpu_workers):
            self._worker_tasks.append(asyncio.create_task(self._worker(self._cpu_queue)))
        logger.info(
            "Scheduler started: %d GPU workers, %d CPU workers",
            self._gpu_workers,
            self._cpu_workers,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("Scheduler stopped")

    async def submit(self, requires_gpu: bool, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self._started:
            await self.start()

        if requires_gpu and self._gpu_workers == 0:
            logger.debug("GPU not available, routing %s to CPU queue", getattr(fn, "__name__", "?"))
            requires_gpu = False

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = _WorkItem(fn=fn, args=args, kwargs=kwargs, future=future)
        queue = self._gpu_queue if requires_gpu else self._cpu_queue
        await queue.put(item)
        return await future

    async def _worker(self, queue: asyncio.Queue[_WorkItem]) -> None:
        while True:
            item = await queue.get()
            try:
                result = await asyncio.to_thread(item.fn, *item.args, **item.kwargs)
                item.future.set_result(result)
            except Exception as ex:
                item.future.set_exception(ex)
            finally:
                queue.task_done()
