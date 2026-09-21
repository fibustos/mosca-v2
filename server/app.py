"""FastAPI application and static web interface."""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import STATIC_DIRECTORY
from server.websocket_manager import simulation_websocket

app = FastAPI(title="Mosca v2: Closed-loop 3D")
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY, html=True), name="static")
app.mount("/css", StaticFiles(directory=STATIC_DIRECTORY / "css"), name="css")
app.mount("/js", StaticFiles(directory=STATIC_DIRECTORY / "js"), name="js")


@app.get("/", include_in_schema=False)
async def web_interface() -> FileResponse:
    """Serve the interactive web interface."""
    return FileResponse(STATIC_DIRECTORY / "index.html")


app.add_api_websocket_route("/ws/simulation", simulation_websocket)
