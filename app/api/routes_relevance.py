from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Form
import traceback


router = APIRouter()

@router.post("/model/bi-encoder/")
async def relevance_analysis(request: Request, topic: str = Form(...), file: UploadFile = File(...)):
    try:
        scorer = request.app.state.scorer
        content = (await file.read()).decode("utf-8")
        result = scorer.score_relevance(content, topic)

        return result.__dict__
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)

        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": e.__class__.__name__,
                "traceback": error_trace.splitlines()[-5:],
            }
        )