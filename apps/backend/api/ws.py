"""WebSocket endpoint for video frames."""
from fastapi import WebSocket

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Handle video frames
    pass
