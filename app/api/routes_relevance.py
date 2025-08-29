from fastapi import APIRouter, UploadFile, File, Request


router = APIRouter()

@router.post("/model/bi-encoder/")
async def relevance_analysis(request: Request, topic: str, file: UploadFile = File(...)):
    scorer = request.app.state.scorer
    content = (await file.read()).decode("utf-8")
    result = scorer.score_relevance(content, topic)

    return result.__dict__