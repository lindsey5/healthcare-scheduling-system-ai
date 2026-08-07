import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from agent.agent import initialize_agent
from routes.ai_agent_routes import agent_router

app = FastAPI()

allowed_origins = [
    "http://localhost:5173"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_agent()

app.include_router(agent_router)

@app.middleware("http")
async def log_request(request, call_next):
    print(">>> Origin:", request.headers.get("origin"))
    response = await call_next(request)
    print("<<< Access-Control-Allow-Origin:", response.headers.get("access-control-allow-origin"))
    return response

@app.get("/")
async def root():
    return JSONResponse(content={"hi"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)