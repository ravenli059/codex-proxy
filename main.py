import json
import logging
import time
import traceback
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxy")

app = FastAPI()

# --- 从 modelconfig.json 加载配置 ---
CONFIG_PATH = Path(__file__).parent / "modelconfig.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = json.load(f)

MODEL_CONFIG = _config["model"]
BACKEND_URL = MODEL_CONFIG["baseurl"].rstrip("/")
BACKEND_API_KEY = MODEL_CONFIG["apikey"]
DEFAULT_MODEL = MODEL_CONFIG["modelname"]
# ------------------------------------


def _backend_headers() -> dict:
    return {
        "Authorization": f"Bearer {BACKEND_API_KEY}",
        "Content-Type": "application/json",
    }


@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request):
    """Codex 调用 GET /models 获取模型元数据"""
    logger.info("GET %s", request.url.path)
    return JSONResponse(content={
        "models": [
            {
                "slug": DEFAULT_MODEL,
                "display_name": DEFAULT_MODEL,
                "description": "Local model via proxy",
                "context_window": 32768,
                "max_context_window": 32768,
                "supported_reasoning_levels": ["low", "medium", "high"],
                "supported_in_api": True,
                "supports_reasoning_summaries": False,
                "supports_parallel_tool_calls": True,
                "visibility": "list",
                "input_modalities": ["text"],
            }
        ]
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", DEFAULT_MODEL)
    stream = body.get("stream", False)
    payload = {**body, "model": model}

    if stream:
        return StreamingResponse(_stream_chat(payload, model), media_type="text/event-stream")
    else:
        return await _non_stream_chat(payload, model)


@app.post("/v1/responses")
@app.post("/responses")
async def responses_api(request: Request):
    body = await request.json()
    logger.info("POST /responses model=%s stream=%s", body.get("model"), body.get("stream"))
    logger.info("Raw input: %s", json.dumps(body.get("input", []), ensure_ascii=False)[:2000])
    model = body.get("model", DEFAULT_MODEL)
    stream = body.get("stream", False)
    messages = _convert_responses_input(body.get("input", []))
    logger.info("Converted messages: %s", json.dumps(messages, ensure_ascii=False)[:2000])
    payload = {"model": model, "messages": messages, "stream": stream}

    if stream:
        return StreamingResponse(_stream_responses(payload, model), media_type="text/event-stream")
    else:
        return await _non_stream_responses(payload, model)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    model = body.get("model", DEFAULT_MODEL)
    stream = body.get("stream", False)

    messages = []
    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        messages.append({"role": role, "content": content})

    payload = {"model": model, "messages": messages, "stream": stream}

    if stream:
        return StreamingResponse(_stream_anthropic(payload, model), media_type="text/event-stream")
    else:
        return await _non_stream_anthropic(payload, model)


# --- 流式生成器：client 在内部创建，生命周期与生成器一致 ---

async def _stream_chat(payload: dict, model: str):
    payload = {**payload, "stream": True}
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{BACKEND_URL}/chat/completions", json=payload, headers=_backend_headers()) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield f"data: {json.dumps({'error': {'message': body.decode(), 'type': 'proxy_error'}})}\n\n"
                return
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n"
            yield "data: [DONE]\n\n"


async def _stream_responses(payload: dict, model: str):
    payload = {**payload, "stream": True}
    resp_id = f"resp_{int(time.time())}"
    created_at = int(time.time())
    output_text = ""
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{BACKEND_URL}/chat/completions", json=payload, headers=_backend_headers()) as resp:
                yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': {'id': resp_id, 'object': 'response', 'status': 'in_progress', 'model': model, 'created_at': created_at, 'output': []}})}\n\n"

                if resp.status_code != 200:
                    body = await resp.aread()
                    logger.error("Backend error %d: %s", resp.status_code, body.decode())
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'proxy_error', 'message': body.decode()}})}\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        text = chunk["choices"][0].get("delta", {}).get("content", "")
                        if text:
                            output_text += text
                            yield f"event: response.output_item.delta\ndata: {json.dumps({'type': 'response.output_item.delta', 'delta': {'type': 'content_delta', 'content_index': 0, 'delta': {'type': 'output_text_delta', 'text': text}}})}\n\n"
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    except Exception as e:
        logger.error("Stream error: %s\n%s", e, traceback.format_exc())

    completed_at = int(time.time())
    yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': {'type': 'message', 'id': f'msg_{resp_id}', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': output_text}]}})}\n\n"
    yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': {'id': resp_id, 'object': 'response', 'status': 'completed', 'model': model, 'created_at': created_at, 'completed_at': completed_at, 'output': [{'type': 'message', 'id': f'msg_{resp_id}', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': output_text}]}], 'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}}})}\n\n"


async def _stream_anthropic(payload: dict, model: str):
    payload = {**payload, "stream": True}
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{BACKEND_URL}/chat/completions", json=payload, headers=_backend_headers()) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'proxy_error', 'message': body.decode()}})}\n\n"
                return
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    yield "event: message_stop\ndata: {}\n\n"
                    return
                try:
                    chunk = json.loads(data_str)
                    text = chunk["choices"][0].get("delta", {}).get("content", "")
                    if text:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


# --- 非流式端点 ---

async def _non_stream_chat(payload: dict, model: str):
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{BACKEND_URL}/chat/completions", json=payload, headers=_backend_headers())
    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    return JSONResponse(content=resp.json())


async def _non_stream_responses(payload: dict, model: str):
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{BACKEND_URL}/chat/completions", json=payload, headers=_backend_headers())
    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    data = resp.json()
    text = data["choices"][0]["message"].get("content", "")
    created_at = int(time.time())
    resp_id = f"resp_{created_at}"
    return JSONResponse(content={
        "id": resp_id,
        "object": "response",
        "status": "completed",
        "model": model,
        "created_at": created_at,
        "completed_at": created_at,
        "output": [{"type": "message", "id": f"msg_{resp_id}", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text}]}],
        "usage": {
            "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": data.get("usage", {}).get("total_tokens", 0),
        },
    })


async def _non_stream_anthropic(payload: dict, model: str):
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{BACKEND_URL}/chat/completions", json=payload, headers=_backend_headers())
    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    data = resp.json()
    text = data["choices"][0]["message"].get("content", "")
    return JSONResponse(content={
        "id": "msg_proxy_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "usage": {
            "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
        },
    })


# --- 工具函数 ---

def _convert_responses_input(input_data) -> list:
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    # OpenAI Responses API 角色 -> Chat Completions 角色
    ROLE_MAP = {"developer": "system"}

    messages = []
    for item in input_data:
        item_type = item.get("type", "message")
        role = ROLE_MAP.get(item.get("role", "user"), item.get("role", "user"))

        if item_type == "message":
            content = item.get("content", "")
            if isinstance(content, list):
                content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            messages.append({"role": role, "content": content})
        elif item_type == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": item.get("call_id", ""), "type": "function", "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "")}}],
            })
        elif item_type == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""), "content": item.get("output", "")})

    return messages


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def catch_all(request: Request, path: str):
    logger.warning("404: %s %s", request.method, path)
    return JSONResponse(
        content={"error": "Available endpoints: /v1/chat/completions, /v1/responses, /v1/messages, /v1/models"},
        status_code=404,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9101)
