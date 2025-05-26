from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import gpt_client
import llama_client
from gru import get_prediction as predict_with_gru
import bert
from ml_classic import *
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

@app.get("/")
async def hello_world():
    return {"message": "Hello, world!"}

@app.post("/predict/{model_name}")
async def predict_text(model_name: str, payload: TextPayload):
    text = payload.text
    
    if model_name == "gpt-4o":
        result = gpt_client.get_classification(text)
    elif model_name == "llama-3.1":
        result = llama_client.get_classification(text)
    elif model_name == "logistic_regression":
        result = predict_with_logistic_regression(text)
    elif model_name == "naive_bayes":
        result = predict_with_naive_bayes(text)
    elif model_name == "svm":
        result = predict_with_svm(text)
    elif model_name == "gru":
        result = predict_with_gru(text)
    elif model_name in bert.models:
        result = bert.get_prediction(text, model_name)
    else:
        result = "Unknown model"

    return {"result": result, "text":text}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)