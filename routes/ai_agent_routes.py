from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import uuid
import re
from agent.agent import get_chat_bot_agent

agent_router = APIRouter()


def extract_text(message):
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))

        return "".join(texts)

    return ""


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

        text = extract_text(ai_message)

        text = re.sub(r"```[a-zA-Z]*\n?", "", text)
        text = text.replace("```", "")
        return JSONResponse(
            status_code=200,
            content={
                "thread_id": thread_id,
                "message": text,
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