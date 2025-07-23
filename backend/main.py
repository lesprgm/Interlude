import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import socketio
import asyncio
import logging
from google.cloud import speech_v1p1beta1 as speech
# from google.oauth2 import service_account # Not needed if GOOGLE_APPLICATION_CREDENTIALS env var is set
# import base64 # No longer needed as audio is handled as binary directly

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- Google Cloud Speech-to-Text Setup ---
speech_client = speech.SpeechClient()

# Dictionary to hold active STT streaming requests per SID
# Each entry will be a tuple: (stream_generator, response_iterator)
active_stt_streams = {}

# STT configuration (adjust as needed for your audio)
# This assumes 16000 Hz, mono, LINEAR16 (PCM) audio
STT_CONFIG = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US",
    enable_automatic_punctuation=True,
)
STREAMING_CONFIG = speech.StreamingRecognitionConfig(
    config=STT_CONFIG,
    interim_results=True, # Essential for real-time updates
)

# --- Socket.IO Event Handlers ---
@sio.event
async def connect(sid, environ):
    users[sid] = {
        'id': sid,
        'room': None,
        'connected': True
    }
    logger.info(f"Connect: {sid}")
    await sio.emit('message', f'Welcome, {sid}!', room=sid)

@sio.event
async def disconnect(sid, data=None):
    logger.info(f"Disconnect: {sid}")
    # Close STT stream if active
    if sid in active_stt_streams:
        # Close the generator to signal end of stream to GCP
        active_stt_streams[sid][0].close()
        del active_stt_streams[sid]
        logger.info(f"Closed STT stream for {sid}")

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
    
    logger.info(f"User {sid} joined room {room_id}. Current room size: {len(room)}")

    # If this is the second user, initiate connection
    if len(room) == 2:
        room_users = list(room)
        other_user = room_users[0] if room_users[1] == sid else room_users[1]
        
        # Tell the other user that someone joined
        await sio.emit('user-joined', {'userId': sid}, room=other_user)
        await sio.emit('user-ready', {'userId': other_user}, room=sid)
    elif len(room) > 2:
        await sio.emit('room-full', room=sid)

# --- STT Audio Streaming Event Handlers (Consolidated and Corrected) ---
@sio.on('start_audio_stream') # Note: Event name is 'start_audio_stream' (snake_case)
async def start_audio_stream(sid, data=None): # data is optional, might contain audio config
    logger.info(f"Starting audio stream for {sid}")
    if sid in active_stt_streams:
        logger.warning(f"STT stream already active for {sid}, closing existing one.")
        # Close the generator to signal end of stream to GCP
        active_stt_streams[sid][0].close() 
        del active_stt_streams[sid]

    # Create a new audio content generator for this session
    # This generator will yield AudioContent objects as chunks come in
    audio_generator = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in iter(lambda: None, None))
    
    # Start the streaming recognition call
    responses = speech_client.streaming_recognize(STREAMING_CONFIG, audio_generator)
    
    active_stt_streams[sid] = (audio_generator, responses) # Store generator and iterator

    # Start a background task to consume responses from GCP STT
    asyncio.create_task(handle_stt_responses(sid, responses))
    logger.info(f"STT streaming recognition initiated for {sid}")

@sio.on('send_audio_chunk') # Note: Event name is 'send_audio_chunk' (snake_case)
async def send_audio_chunk(sid, audio_data):
    if sid not in active_stt_streams:
        logger.warning(f"Received audio chunk for {sid} but no active STT stream. Ignoring.")
        return
    
    try:
        # Send the audio data to the generator.
        # The `_write` method is internal but used for feeding data to gRPC streams.
        active_stt_streams[sid][0]._write(audio_data)
        # logger.debug(f"Sent {len(audio_data)} bytes for {sid}") # Uncomment for detailed chunk logging
    except Exception as e:
        logger.error(f"Error sending audio chunk for {sid}: {e}", exc_info=True)
        # Consider closing stream if error occurs
        if sid in active_stt_streams:
            active_stt_streams[sid][0].close()
            del active_stt_streams[sid]

@sio.on('end_audio_stream') # Note: Event name is 'end_audio_stream' (snake_case)
async def end_audio_stream(sid, data=None): # data is optional
    logger.info(f"Ending audio stream for {sid}")
    if sid in active_stt_streams:
        active_stt_streams[sid][0].close() # Signal end of stream to GCP
        del active_stt_streams[sid]
        logger.info(f"STT stream for {sid} explicitly ended.")

async def handle_stt_responses(sid, responses):
    """Background task to process responses from GCP STT and relay to peer."""
    try:
        user_info = users.get(sid)
        if not user_info or not user_info['room']:
            logger.warning(f"User {sid} not in a room, cannot relay transcript.")
            return

        room_id = user_info['room']
        current_room_sids = rooms.get(room_id)

        if not current_room_sids or len(current_room_sids) != 2:
            logger.warning(f"Room {room_id} does not have exactly two users, cannot relay transcript for {sid}.")
            return

        # Find the other user in the room
        other_user_sid = next(iter(s for s in current_room_sids if s != sid), None)

        if not other_user_sid:
            logger.warning(f"Could not find peer for {sid} in room {room_id}, cannot relay transcript.")
            return

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            
            if result.is_final:
                logger.info(f"Final Transcript for {sid} (relaying to {other_user_sid}): {transcript}")
                await sio.emit('transcribed_text', {'text': transcript, 'isFinal': True}, room=other_user_sid)
            else:
                logger.info(f"Partial Transcript for {sid} (relaying to {other_user_sid}): {transcript}")
                await sio.emit('transcribed_text', {'text': transcript, 'isFinal': False}, room=other_user_sid)
    except Exception as e:
        logger.error(f"Error processing STT responses for {sid}: {e}", exc_info=True)
    finally:
        if sid in active_stt_streams:
            active_stt_streams[sid][0].close()
            del active_stt_streams[sid]
            logger.info(f"STT response handler for {sid} finished/cleaned up.")


# WebRTC Signaling Events (Remain unchanged)
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
    await sio.emit('message', f'Server received: {data}', room=sid)

if __name__ == "__main__":
    uvicorn.run(app_with_sio, host="0.0.0.0", port=8000)
