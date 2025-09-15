from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import traceback

from app.api import routes_relevance
from app.services.scorer import TopicRelevanceScorer

@asynccontextmanager
async def lifespan(app: FastAPI):
    scorer = TopicRelevanceScorer()
    app.state.scorer = scorer
    yield



app = FastAPI(lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("validation error: ")
    print(exc.errors())
    print(exc.body)
    print(traceback.format_exc())

    return JSONResponse(
        status_code=422,
        content={ "detail": exc.errors(), "body": exc.body }
    )

@app.get("/")
async def root():
    return { "message": "api for semantic and sensitivity analysis. accepts a topic and .txt file: '/model/bi-encoder/'" }


app.include_router(routes_relevance.router)