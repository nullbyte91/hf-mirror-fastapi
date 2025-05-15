import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
import httpx
import aiofiles

app = FastAPI(title="HF Mirror FastAPI")

# Directory for cached files
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face authentication token
TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
client = httpx.AsyncClient()

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return JSONResponse({"message": "HF Mirror FastAPI is running 🚀"})

@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
async def mirror(full_path: str, request: Request):
    if not full_path or full_path.endswith("/"):
        raise HTTPException(status_code=404, detail="Not Found")

    url = f"https://huggingface.co/{full_path}"
    cache_path = CACHE_DIR / full_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.is_file():
        if request.method == "HEAD":
            return Response(status_code=200)
        return FileResponse(str(cache_path))

    headers = {}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code)

    async with aiofiles.open(cache_path, "wb") as f:
        await f.write(resp.content)

    if request.method == "HEAD":
        return Response(status_code=200)
    return Response(content=resp.content, media_type=resp.headers.get("content-type"))
