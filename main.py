import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import FileResponse
import httpx
import aiofiles

app = FastAPI()
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
client = httpx.AsyncClient()

@app.get("/{full_path:path}")
async def mirror(full_path: str, request: Request):
    # Determine source URL
    url = f"https://huggingface.co/{full_path}"
    cache_path = CACHE_DIR / full_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Serve from cache if exists
    if cache_path.exists():
        return FileResponse(str(cache_path))

    # Fetch from Hugging Face
    headers = {}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code)

    # Cache and return
    async with aiofiles.open(cache_path, "wb") as f:
        await f.write(resp.content)
    return Response(content=resp.content, media_type=resp.headers.get("content-type"))
