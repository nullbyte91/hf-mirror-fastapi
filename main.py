import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import httpx
import aiofiles

app = FastAPI(title="HF Mirror FastAPI")

# Directory for cached files
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face authentication token
TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

@app.on_event("startup")
async def startup_event():
    app.state.client = httpx.AsyncClient(timeout=60.0)

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()

@app.get("/")
async def root():
    return JSONResponse({"message": "HF Mirror FastAPI is running 🚀"})

@app.get("/api/{api_path:path}")
async def proxy_hf_api(api_path: str, request: Request):
    url = f"https://huggingface.co/api/{api_path}"
    headers = {}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"]
    async with app.state.client.stream("GET", url, headers=headers) as r:
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code)
        return StreamingResponse(r.aiter_raw(), status_code=r.status_code, media_type=r.headers.get("content-type"))

@app.get("/{full_path:path}")
async def mirror(full_path: str):
    if not full_path or full_path.endswith("/"):
        raise HTTPException(status_code=404, detail="Not Found")

    # Build cache path
    cache_path = CACHE_DIR / full_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.is_file():
        return FileResponse(str(cache_path))

    # Proxy download from Hugging Face
    hf_url = f"https://huggingface.co/{full_path}"
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

    async with app.state.client.stream("GET", hf_url, headers=headers) as r:
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code)

        # Save streamed content to cache
        async with aiofiles.open(cache_path, "wb") as f:
            async for chunk in r.aiter_bytes():
                await f.write(chunk)

        return FileResponse(str(cache_path), media_type=r.headers.get("content-type"))
