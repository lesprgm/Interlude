import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import socketio

# Initialize Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
# Create FastAPI app
app = FastAPI()

# Wrap FastAPI app with Socket.IO ASGI app
# This makes the Socket.IO server accessible at /socket.io/ (default)
app_with_sio = socketio.ASGIApp(sio, app)

# HTML for the root endpoint (for testing browser access)
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>BridgeSpeak Backend</title>
</head>
<body>
    <h1>BridgeSpeak Backend is Running!</h1>
    <p>This is the root endpoint. Socket.IO and other APIs will be here.</p>
</body>
</html>
"""

@app.get("/")
async def read_root():
    return HTMLResponse(content=html_content)

# test endpoint for FastAPI (optional, but good for quick check)
@app.get("/hello")
async def hello_world():
    return {"message": "Hello from FastAPI!"}

# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    await sio.emit('message', f'Welcome, {sid}!', room=sid)

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def message(sid, data):
    print(f"Message from {sid}: {data}")
    # Echo message back for testing
    await sio.emit('message', f'Server received: {data}', room=sid)


if __name__ == "__main__":
    # When running with uvicorn directly, you use the app_with_sio
    # Remember to bind to 0.0.0.0 to be accessible externally
    uvicorn.run(app_with_sio, host="0.0.0.0", port=8000)