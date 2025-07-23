import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import socketio
import asyncio
import logging
# Import SpeechAsyncClient for asynchronous operations
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import SpeechAsyncClient # Import the async client
from asyncio import Queue # Import Queue

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
# Instantiate the asynchronous SpeechClient
speech_client = SpeechAsyncClient()

# Dictionary to hold active STT streaming requests per SID
# Each entry will be a tuple: (audio_queue, response_iterator)
active_stt_streams = {}

# STT configuration (adjust as needed for your audio)
# This assumes 16000 Hz, mono, LINEAR16 (PCM) audio
STT_CONFIG = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, # <<< CHANGED: Expect WebM/Opus encoding
    sample_rate_hertz=48000, # <<< ADJUSTED: Opus typically uses 48000 Hz sample rate
    language_code="en-US",
    enable_automatic_punctuation=True,
)
STREAMING_CONFIG = speech.StreamingRecognitionConfig(
    config=STT_CONFIG,
    interim_results=True, # Essential for real-time updates
)

# --- Async Generator for STT Audio Input ---
async def generate_audio_requests(audio_queue: Queue, streaming_config: speech.StreamingRecognitionConfig):
    """
    An async generator that yields StreamingRecognizeRequest objects.
    It sends the streaming_config as the first request, then reads audio chunks
    from an asyncio.Queue for subsequent requests.
    """
    # Send the configuration as the first request
    yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)

    while True:
        try:
            # Get audio chunk from the queue
            chunk = await audio_queue.get()
            if chunk is None: # Signal to close the stream
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)
        except asyncio.CancelledError:
            logger.info("Audio request generator cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in audio request generator: {e}", exc_info=True)
            break

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
        # Put None into the queue to signal the generator to stop
        await active_stt_streams[sid][0].put(None) 
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

# --- STT Audio Streaming Event Handlers ---
@sio.on('start_audio_stream')
async def start_audio_stream(sid, data=None):
    logger.info(f"Starting audio stream for {sid} (Version: 2025-07-23_17:00_OpusConfig)") # Updated version check
    if sid in active_stt_streams:
        logger.warning(f"STT stream already active for {sid}, closing existing one.")
        await active_stt_streams[sid][0].put(None) # Signal old generator to stop
        del active_stt_streams[sid]

    # Create a new Queue for this session's audio chunks
    audio_queue = Queue()
    
    # Create the async generator that will read from the queue and send config first
    audio_requests_generator = generate_audio_requests(audio_queue, STREAMING_CONFIG)

    # Start the streaming recognition call using the async client
    responses = await speech_client.streaming_recognize(requests=audio_requests_generator)
    
    # Store the audio_queue and responses iterator
    active_stt_streams[sid] = (audio_queue, responses)

    # Start a background task to consume responses from GCP STT
    asyncio.create_task(handle_stt_responses(sid, responses))
    logger.info(f"STT streaming recognition initiated for {sid}")

@sio.on('send_audio_chunk')
async def send_audio_chunk(sid, audio_data):
    if sid not in active_stt_streams:
        logger.warning(f"Received audio chunk for {sid} but no active STT stream. Ignoring.")
        return
    
    try:
        # Put the audio data into the queue
        await active_stt_streams[sid][0].put(audio_data)
        # logger.debug(f"Sent {len(audio_data)} bytes for {sid}") # Uncomment for detailed chunk logging
    except Exception as e:
        logger.error(f"Error sending audio chunk for {sid}: {e}", exc_info=True)
        # Consider closing stream if error occurs
        if sid in active_stt_streams:
            await active_stt_streams[sid][0].put(None) # Signal generator to stop
            del active_stt_streams[sid]

@sio.on('end_audio_stream')
async def end_audio_stream(sid, data=None):
    logger.info(f"Ending audio stream for {sid}")
    if sid in active_stt_streams:
        await active_stt_streams[sid][0].put(None) # Signal the generator to stop
        del active_stt_streams[sid]
        logger.info(f"STT stream for {sid} explicitly ended.")

async def handle_stt_responses(sid, responses):
    """Background task to process responses from GCP STT and relay to peer."""
    try:
        # Use async for to iterate over the async responses iterator
        async for response in responses: 
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            
            user_info = users.get(sid)
            room_id = user_info['room'] if user_info else None
            
            other_user_sid = None
            # Check for exactly two users in the room before attempting to find the other user
            if room_id and room_id in rooms and len(rooms[room_id]) == 2:
                current_room_sids = rooms.get(room_id)
                other_user_sid = next(iter(s for s in current_room_sids if s != sid), None)

            if other_user_sid: # Only emit if there's a peer to send to
                if result.is_final:
                    logger.info(f"Final Transcript for {sid} (relaying to {other_user_sid}): {transcript}")
                    await sio.emit('transcribed_text', {'text': transcript, 'isFinal': True}, room=other_user_sid)
                else:
                    logger.info(f"Partial Transcript for {sid} (relaying to {other_user_sid}): {transcript}")
                    await sio.emit('transcribed_text', {'text': transcript, 'isFinal': False}, room=other_user_sid)
            else:
                # Log that transcription is happening but not relayed yet
                logger.info(f"Transcript for {sid} (not yet relayed): {transcript} (Room size: {len(rooms.get(room_id, [])) if room_id else 'N/A'})")

    except Exception as e:
        logger.error(f"Error processing STT responses for {sid}: {e}", exc_info=True)
    finally: # This finally block is now for graceful task shutdown, not stream cleanup
        # Ensure the audio queue is signaled to stop if the response handler exits unexpectedly
        if sid in active_stt_streams:
            try:
                await active_stt_streams[sid][0].put(None)
            except Exception as q_e:
                logger.warning(f"Error signaling audio queue to stop for {sid}: {q_e}")
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
