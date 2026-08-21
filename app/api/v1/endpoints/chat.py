"""智能问答端点 — POST /api/v1/chat, POST /api/v1/chat/stream

/chat:        意图识别 → 路由 (rag_qa / direct / multi_hop) → 回答
/chat/stream: 同链路, SSE 真流式 (token 级, 事件见 ChatService.stream)
SSE 事件序列: data: {"type":"intent",...} → 若干 data: {"type":"answer_delta",...}
              → data: {"type":"sources",...} → data: {"type":"done",...}
出错时插入 data: {"type":"error",...}; 均用 json.dumps 序列化, 不拼字符串。
"""

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service
from app.core.security import verify_api_key
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"], dependencies=[Depends(verify_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """智能问答主接口 (非流式, 带限流/超时降级/多轮记忆)"""
    result = await chat_service.chat(req.question, session_id=req.session_id)
    return ChatResponse(**result)


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """智能问答 (SSE 真流式): 事件带 type 字段, 结束发 data: {"type":"done"}"""

    async def event_source():
        async for event in chat_service.stream(req.question, session_id=req.session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
