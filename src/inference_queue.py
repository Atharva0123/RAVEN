"""Bounded asynchronous micro-batching queue for serialized model execution."""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class AsyncInferenceQueue:
    """Collect short request bursts and execute them sequentially off-loop.

    This admission queue reduces event-loop blocking and controls concurrent
    access to stateful tracker/model objects. Each queued image is still run
    through the existing frame pipeline independently.
    """

    def __init__(self, processor: Callable, max_batch_size=None, max_wait_ms=None):
        self.processor = processor
        self.max_batch_size = max_batch_size or int(os.getenv("RAVEN_QUEUE_BATCH_SIZE", "8"))
        self.max_wait = (max_wait_ms if max_wait_ms is not None else
                         int(os.getenv("RAVEN_QUEUE_WAIT_MS", "8"))) / 1000
        self.queue = None
        self.worker = None

    async def start(self):
        if self.worker is None or self.worker.done():
            self.queue = asyncio.Queue(maxsize=int(os.getenv("RAVEN_QUEUE_CAPACITY", "256")))
            self.worker = asyncio.create_task(self._run(), name="raven-inference-queue")

    async def submit(self, payload):
        await self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        try:
            self.queue.put_nowait((payload, future))
        except asyncio.QueueFull as exc:
            raise RuntimeError("Inference queue is full; retry shortly") from exc
        return await future

    async def close(self):
        if self.worker is not None:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
            self.worker = None

    async def _run(self):
        while True:
            first = await self.queue.get()
            batch = [first]
            if self.max_wait:
                await asyncio.sleep(self.max_wait)
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                outputs = await asyncio.to_thread(self._process_batch, batch)
                for (_, future), output in zip(batch, outputs):
                    if not future.cancelled():
                        if isinstance(output, Exception):
                            future.set_exception(output)
                        else:
                            future.set_result(output)
            except Exception as exc:
                LOGGER.exception("Inference queue batch failed")
                for _, future in batch:
                    if not future.cancelled():
                        future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()

    def _process_batch(self, batch):
        outputs = []
        for payload, _ in batch:
            try:
                outputs.append(self.processor(payload))
            except Exception as exc:
                outputs.append(exc)
        return outputs
