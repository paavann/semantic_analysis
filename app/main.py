from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import routes_relevance
from app.services.scorer import TopicRelevanceScorer

@asynccontextmanager
async def lifespan(app: FastAPI):
    scorer = TopicRelevanceScorer()
    app.state.scorer = scorer
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return { "message": "hello world!" }


app.include_router(routes_relevance.router)