from fastapi import FastAPI
from api import routes_relevance
from services.scorer import TopicRelevanceScorer
from contextlib import asynccontextmanager

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