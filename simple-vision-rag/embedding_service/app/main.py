from app.lifespan import lifespan
from app.router import embedding_router
from fastapi import FastAPI

app = FastAPI(lifespan=lifespan)
app.include_router(embedding_router)
