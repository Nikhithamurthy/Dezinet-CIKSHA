from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Dezinet CIKSHA")


class Health(BaseModel):
    status: str


@app.get("/health")
async def health() -> Health:
    return Health(status="ok")
