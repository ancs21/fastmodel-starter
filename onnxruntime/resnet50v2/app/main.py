from typing import Optional
from starlette.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi import FastAPI, Depends, Security, HTTPException
from fastapi.security.api_key import APIKey, APIKeyQuery, APIKeyHeader

from app.predictor import Predictor

API_KEY = "1234567asdfgh"
API_KEY_NAME = "access_token"

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header),
):

    if api_key_query == API_KEY:
        return api_key_query
    elif api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Unauthorized"
        )
        
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
model = Predictor('app/resnet50-v2-7.onnx')

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI."}

@app.get('/predict')
async def predict(image_url: Optional[str] = None, api_key: APIKey = Depends(get_api_key)):
    if image_url is None:
        return JSONResponse({"error": "Miss image_url query"}, status_code=400)
    try:
        data = model.predict(image_url)
        return JSONResponse({"output": data}, status_code=200) 

    except Exception as e:
        return JSONResponse({
            "error": "Unexpected Error: could not run inference on model"},
            status_code=500)