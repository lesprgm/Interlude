import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import socketio

# Initialize Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
# Create FastAPI app
app = FastAPI()

# Wrap FastAPI app with Socket.IO ASGI app
app_with_sio = socketio.ASGIApp(sio, app)

# HTML for the root endpoint
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Interlude Backend</title>
</head>
<body>
    <h1>Interlude Backend is Running!</h1>
    <p>Socket.IO and WebRTC signaling server for real-time communication.</p>
</body>
</html>
"""

@app.get("/")
async def read_root():
    return HTMLResponse(content=html_content)

@app.get("/hello")
async def hello_world():
    return {"message": "Hello from Interlude API!"}

# Status endpoint to monitor rooms and users
@app.get("/status")
async def get_status():
    return {
        "status": "active",
        "active_rooms": len(rooms),
        "active_users": len(users),
        "rooms_detail": {room_id: list(room_users) for room_id, room_users in rooms.items()}
    }

# Store active rooms and users
rooms = {}
users = {}

# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ):
    users[sid] = {
        'id': sid,
        'room': None,
        'connected': True
    }
    await sio.emit('message', f'Welcome, {sid}!', room=sid)

@sio.event
async def disconnect(sid, data=None):
    # Handle room cleanup
    user = users.get(sid)
    if user and user['room']:
        room_id = user['room']
        if room_id in rooms:
            rooms[room_id].discard(sid)
            
            # Notify other users in the room
            await sio.emit('user-left', {'userId': sid}, room=room_id, skip_sid=sid)
            
            # Clean up empty room
            if room_id in rooms and len(rooms[room_id]) == 0:
                del rooms[room_id]
    
    # Remove user
    if sid in users:
        del users[sid]

@sio.event
async def join_room(sid, data):
    room_id = data['roomId']
    
    # Leave previous room if any
    user = users.get(sid)
    if user and user['room']:
        await sio.leave_room(sid, user['room'])
    
    # Join new room
    await sio.enter_room(sid, room_id)
    users[sid]['room'] = room_id
    
    # Initialize room if it doesn't exist
    if room_id not in rooms:
        rooms[room_id] = set()
    
    room = rooms[room_id]
    room.add(sid)
    
    # If this is the second user, initiate connection
    if len(room) == 2:
        room_users = list(room)
        other_user = room_users[0] if room_users[1] == sid else room_users[1]
        
        # Tell the other user that someone joined
        await sio.emit('user-joined', {'userId': sid}, room=other_user)
        await sio.emit('user-ready', {'userId': other_user}, room=sid)
    elif len(room) > 2:
        await sio.emit('room-full', room=sid)

# WebRTC Signaling Events  
@sio.on('webrtc-offer')
async def webrtc_offer(sid, data):
    to_sid = data.get('to')
    if to_sid:
        await sio.emit('webrtc-offer', {
            'offer': data['offer'],
            'from': sid
        }, room=to_sid)

@sio.on('webrtc-answer')
async def webrtc_answer(sid, data):
    to_sid = data.get('to')
    if to_sid:
        await sio.emit('webrtc-answer', {
            'answer': data['answer'],
            'from': sid
        }, room=to_sid)

@sio.on('webrtc-ice-candidate')
async def webrtc_ice_candidate(sid, data):
    to_sid = data.get('to')
    if to_sid:
        await sio.emit('webrtc-ice-candidate', {
            'candidate': data['candidate'],
            'from': sid
        }, room=to_sid)

@sio.event
async def message(sid, data):
    # Echo message back
    await sio.emit('message', f'Server received: {data}', room=sid)

if __name__ == "__main__":
    # When running with uvicorn directly, you use the app_with_sio
    # Remember to bind to 0.0.0.0 to be accessible externally
    uvicorn.run(app_with_sio, host="0.0.0.0", port=8000)
