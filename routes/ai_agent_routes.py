from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import uuid
from agent.agent import get_chat_bot_agent

agent_router = APIRouter()

@agent_router.post("/api/chat")
async def chat(req: Request):
    try:
        body = await req.json()

        thread_id = body.get("thread_id") or str(uuid.uuid4())
        user_message = body.get("message")

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message is required"},
            )

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        agent = get_chat_bot_agent()

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ]
            },
            config=config,
        )

        ai_message = result["messages"][-1]

        return JSONResponse(
            status_code=200,
            content={
                "thread_id": thread_id,
                "message": ai_message.content,
            },
        )

    except Exception as e:
        print("Error in /api/chat:", e)

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(e),
            },
        )