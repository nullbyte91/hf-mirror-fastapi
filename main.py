import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
import httpx
import aiofiles

app = FastAPI(title="HF Mirror FastAPI")

# Directory to cache files on disk
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# (Optional) Hugging Face API token for private repos
TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

@app.on_event("startup")
async def startup_event():
    # Create a single shared HTTPX client
    app.state.client = httpx.AsyncClient(timeout=60.0)

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up the HTTPX client
    await app.state.client.aclose()

@app.api_route("/", methods=["GET", "HEAD"])
async def health(request: Request):
    """
    Health check endpoint.
    """
    return JSONResponse({"status": "ok"})

@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
async def mirror(full_path: str, request: Request):
    """
    Mirror any Hugging Face Hub file under /{repo_id}/resolve/{revision}/{filename}.
    Caches files in CACHE_DIR.
    Supports both GET (fetch & cache) and HEAD (probe existence).
    """
    if not full_path or full_path.endswith("/"):
        raise HTTPException(status_code=404, detail="Not Found")

    # Construct upstream URL
    upstream_url = f"https://huggingface.co/{full_path}"
    # Map to local cache path
    cache_path = CACHE_DIR / full_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Serve from cache if already downloaded
    if cache_path.is_file():
        if request.method == "HEAD":
            return Response(status_code=200)
        return FileResponse(str(cache_path))

    # Prepare auth headers if needed
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

    # Handle HEAD: probe upstream without downloading
    if request.method == "HEAD":
        resp = await app.state.client.head(upstream_url, headers=headers)
        return Response(status_code=resp.status_code)

    # Handle GET: fetch, cache, and return
    resp = await app.state.client.get(upstream_url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code)

    # Write content to cache
    async with aiofiles.open(cache_path, "wb") as f:
        await f.write(resp.content)

    return FileResponse(
        str(cache_path),
        media_type=resp.headers.get("content-type", "application/octet-stream")
    )
