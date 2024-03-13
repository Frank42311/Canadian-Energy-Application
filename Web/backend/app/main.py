from pathlib import Path

from starlette.responses import HTMLResponse

from models import QueryItem
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api_handlers import router
app = FastAPI()

app.include_router(router)

# Serve static files from the 'frontend' directory
app.mount("/static", StaticFiles(directory="./frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main HTML file to the client. This endpoint is the entry point of the web application,
    where the frontend interface is loaded.

    Returns:
    HTMLResponse: The HTML content of the main page loaded from a file, along with a 200 OK status code.
    """
    with open(Path('./frontend/frontend.html'), 'r', encoding='utf-8') as html_file:
        return HTMLResponse(content=html_file.read(), status_code=200)

# @app.post("/ask")
# async def ask_query(item: QueryItem):
#     return await ask_gpt(item)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)