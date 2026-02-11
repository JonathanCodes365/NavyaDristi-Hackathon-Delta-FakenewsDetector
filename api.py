from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from verify_mvp import verify_article  # import your existing code

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend to access API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerifyInput(BaseModel):
    title: str
    text: str = ""

@app.post("/verify")
def verify(input_data: VerifyInput):
    result = verify_article(input_data.title, input_data.text)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}