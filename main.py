import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
import httpx
import aiofiles

app = FastAPI(title="HF Mirror FastAPI")

CACHE_DIR = Path(os.getenv("CACHE_DIR", "/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

    # ✅ If file exists in cache
    if cache_path.is_file():
        if request.method == "HEAD":
            return Response(status_code=200)
        return FileResponse(str(cache_path))

    # ✅ If file doesn't exist in cache, download on GET, skip download on HEAD
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    
    if request.method == "HEAD":
        # Try remote HEAD to check if it exists
        resp = await client.head(url, headers=headers)
        return Response(status_code=resp.status_code)

    # Download and cache the file
    resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code)

    async with aiofiles.open(cache_path, "wb") as f:
        await f.write(resp.content)

    return Response(content=resp.content, media_type=resp.headers.get("content-type"))
