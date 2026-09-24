"""FastAPI application and static web interface."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import PROFILES_DIRECTORY, STATIC_DIRECTORY
from server.websocket_manager import (
    active_debug_state,
    load_profile_selection_into_active_sessions,
    reset_active_brains_for_diagnostics,
    simulation_websocket,
)

app = FastAPI(title="Mosca v2: Closed-loop 3D")
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY, html=True), name="static")
app.mount("/css", StaticFiles(directory=STATIC_DIRECTORY / "css"), name="css")
app.mount("/js", StaticFiles(directory=STATIC_DIRECTORY / "js"), name="js")


@app.get("/", include_in_schema=False)
async def web_interface() -> FileResponse:
    """Serve the interactive web interface."""
    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/api/profiles/custom")
async def list_custom_profiles() -> list[str]:
    """List saved custom profile names without their JSON extension."""
    directory = PROFILES_DIRECTORY / "custom"
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json") if path.is_file())


@app.post("/api/profiles/custom/load/{name}")
async def load_custom_profile(name: str) -> dict[str, str | int]:
    """Load one saved profile into all connected, live simulation sessions."""
    try:
        loaded_sessions = load_profile_selection_into_active_sessions(f"custom:{name}")
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"name": name, "loaded_sessions": loaded_sessions}


@app.post("/api/profiles/load/{selection}")
async def load_profile_selection(selection: str) -> dict[str, str | int]:
    """Load a validated ``preset:`` or ``custom:`` profile selection."""
    try:
        loaded_sessions = load_profile_selection_into_active_sessions(selection)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"selection": selection, "loaded_sessions": loaded_sessions}


@app.get("/api/debug/state")
async def debug_state() -> dict[str, object]:
    """Return the diagnostic state of the active WebSocket simulation."""
    try:
        state = active_debug_state()
        if state is None:
            return {
                "status": "error",
                "message": "No active simulation session.",
                "fly_state": "RESTING",
                "motor_outputs": {"v_forward": 0.0, "yaw_rate": 0.0, "thrust": 0.0},
            }
        return state
    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
            "fly_state": "RESTING",
            "motor_outputs": {"v_forward": 0.0, "yaw_rate": 0.0, "thrust": 0.0},
        }


@app.post("/api/debug/reset_brain")
async def reset_brain() -> dict[str, int | str]:
    """Restore basal brain weights and restart every live session in exploration."""
    reset_sessions = reset_active_brains_for_diagnostics()
    if not reset_sessions:
        raise HTTPException(status_code=404, detail="No active simulation session.")
    return {
        "message": "Pesos sinápticos basales restaurados; exploración reiniciada.",
        "reset_sessions": reset_sessions,
    }


app.add_api_websocket_route("/ws/simulation", simulation_websocket)
