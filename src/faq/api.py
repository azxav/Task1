import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .service import SearchService, ServiceUnavailable
from .text.normalization import TextValidationError


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(service=None):
    holder = {"service": service}
    admission = asyncio.Semaphore(2)

    @asynccontextmanager
    async def lifespan(app):
        if holder["service"] is None:
            holder["service"] = await run_in_threadpool(SearchService, Settings.load())
        yield

    application = FastAPI(title="Uzbek semantic FAQ", version="0.1.0", lifespan=lifespan)

    @application.get("/health")
    async def health():
        return {"status": "alive"}

    @application.get("/ready")
    async def ready():
        try:
            await run_in_threadpool(holder["service"].assert_ready)
            return {"status": "ready"}
        except ServiceUnavailable:
            raise HTTPException(503, "Search service is not ready") from None

    @application.post("/search")
    async def search(request: SearchRequest):
        try:
            await asyncio.wait_for(admission.acquire(), timeout=0.2)
        except TimeoutError:
            raise HTTPException(503, "Search is busy; retry shortly") from None
        try:
            return await run_in_threadpool(holder["service"].search, request.question)
        except TextValidationError as exc:
            raise HTTPException(422, str(exc)) from None
        except ServiceUnavailable:
            raise HTTPException(503, "Search temporarily unavailable; retry later") from None
        finally:
            admission.release()

    return application


app = create_app()
