"""Bounded per-event-loop microbatching; no prediction/result cache."""
import asyncio


class MicroBatcher:
    def __init__(self,runtime,max_batch=8,wait_seconds=.0005,max_queue=256):
        self.runtime=runtime;self.max_batch=max_batch;self.wait_seconds=wait_seconds
        self.loop=asyncio.get_running_loop();self.queue=asyncio.Queue(maxsize=max_queue)
        self.worker=self.loop.create_task(self._run())

    async def submit(self,text):
        future=self.loop.create_future()
        self.queue.put_nowait((text,future))
        return await future

    async def _run(self):
        try:
            while True:
                items=[await self.queue.get()]
                await asyncio.sleep(self.wait_seconds)
                while len(items)<self.max_batch and not self.queue.empty():items.append(self.queue.get_nowait())
                try:
                    def infer():return self.runtime.logits(self.runtime.tokenize([text for text,_ in items]))
                    outputs=await asyncio.to_thread(infer)
                    for (_,future),output in zip(items,outputs):
                        if not future.done():future.set_result(output)
                except Exception as error:
                    for _,future in items:
                        if not future.done():future.set_exception(error)
        except asyncio.CancelledError:
            while not self.queue.empty():
                _,future=self.queue.get_nowait()
                if not future.done():future.cancel()
            raise
