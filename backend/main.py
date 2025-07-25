import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import socketio
import asyncio
import logging
import numpy as np # Added for numerical operations in ML model
import tensorflow as tf # Added for the ML model placeholder

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

# --- ASL Keypoint Buffering and Model Placeholder ---
# Store keypoint buffers for each session
# Each entry will be a list of keypoint_data dictionaries for a short sequence
keypoint_buffers = {}
# Define a maximum buffer size (e.g., 30 frames for 1 second at 30fps, adjust as needed)
MAX_KEYPOINT_BUFFER_SIZE = 30

class ASLRecognizer:
    def __init__(self):
        self.model = None
        # Define your 5-10 chosen isolated words/short phrases + UNKNOWN
        self.labels = ["HELLO", "YES", "NO", "THANK_YOU", "I_LOVE_YOU", "GO", "STOP", "EAT", "DRINK", "HOW_ARE_YOU", "UNKNOWN"]
        
        # Define the input shape for a single frame
        # (33 pose landmarks + 21 left hand + 21 right hand) * 3 coordinates (x,y,z)
        # Assuming x,y,z for all landmarks. Adjust if you include visibility or more/fewer landmarks.
        self.frame_features = (33 + 21 + 21) * 3 

        # Define the sequence length (number of frames in one sign/phrase)
        self.sequence_length = MAX_KEYPOINT_BUFFER_SIZE 

        try:
            # Create a very simple 1D CNN for sequence processing, optimized for CPU
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.sequence_length, self.frame_features)), # Input shape for sequence
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'), # Small 1D Conv layer
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), # Small hidden layer
                tf.keras.layers.Dropout(0.3), # Dropout for regularization
                tf.keras.layers.Dense(len(self.labels), activation='softmax') # Output layer for classification
            ])
            # No need to compile or train for this placeholder, just define structure
            logger.info("ASL Recognizer initialized. CPU-optimized sequence model placeholder created.")
        except Exception as e:
            logger.warning(f"Could not create ASL model placeholder: {e}")

    def preprocess_keypoints(self, keypoint_sequence_buffer):
        # This function takes a list of keypoint_data (frames)
        processed_sequence = []
        for frame_data in keypoint_sequence_buffer:
            flat_features = []
            # Extract pose landmarks
            if frame_data.get('pose'):
                for landmark in frame_data['pose']:
                    flat_features.extend([landmark['x'], landmark['y'], landmark['z']])
            # Extract left hand landmarks
            if frame_data.get('leftHand'):
                for landmark in frame_data['leftHand']:
                    flat_features.extend([landmark['x'], landmark['y'], landmark['z']])
            # Extract right hand landmarks
            if frame_data.get('rightHand'):
                for landmark in frame_data['rightHand']:
                    flat_features.extend([landmark['x'], landmark['y'], landmark['z']])

            # Pad or truncate individual frame features to match self.frame_features
            if len(flat_features) < self.frame_features:
                flat_features.extend([0.0] * (self.frame_features - len(flat_features)))
            elif len(flat_features) > self.frame_features:
                flat_features = flat_features[:self.frame_features]
            
            processed_sequence.append(flat_features)

        # Pad the sequence if it's shorter than expected
        while len(processed_sequence) < self.sequence_length:
            processed_sequence.append([0.0] * self.frame_features) # Pad with zeros
        
        # Truncate the sequence if it's longer than expected (take the most recent frames)
        if len(processed_sequence) > self.sequence_length:
            processed_sequence = processed_sequence[-self.sequence_length:]

        # Convert to NumPy array, add batch dimension (1 for single sequence)
        return np.array([processed_sequence], dtype=np.float32)

    def predict(self, keypoint_sequence_buffer):
        # Ensure the buffer is not empty before processing
        if not keypoint_sequence_buffer:
            return "No valid keypoints", 0.0

        processed_input = self.preprocess_keypoints(keypoint_sequence_buffer)
        
        # Check if preprocessing resulted in valid input
        if processed_input is None or processed_input.shape[1] == 0:
            return "Error: Invalid processed keypoint data", 0.0

        # For the placeholder, we still use random for now
        import random
        predicted_index = random.randint(0, len(self.labels) - 1)
        predicted_label = self.labels[predicted_index]
        confidence = random.uniform(0.5, 0.99)
        return predicted_label, confidence

# Instantiate the ASLRecognizer globally
asl_recognizer = ASLRecognizer()


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

    # Clear keypoint buffer for disconnected user
    if sid in keypoint_buffers:
        del keypoint_buffers[sid]
        logger.info(f"Cleared keypoint buffer for {sid}")

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
    # Extract user role from data
    user_role = data.get('userRole', 'hearing') # Default to 'hearing' if not provided

    # Leave previous room if any
    user = users.get(sid)
    if user and user['room']:
        await sio.leave_room(sid, user['room'])
    
    # Join new room
    await sio.enter_room(sid, room_id)
    users[sid]['room'] = room_id
    users[sid]['role'] = user_role # Store the user's role
    
    # Initialize room if it doesn't exist
    if room_id not in rooms:
        rooms[room_id] = set()
    
    room = rooms[room_id]
    room.add(sid)
    
    logger.info(f"User {sid} (Role: {user_role}) joined room {room_id}. Current room size: {len(room)}")

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

            # Only relay transcript if the other user is 'deaf'
            if other_user_sid:
                other_user_info = users.get(other_user_sid)
                if other_user_info and other_user_info.get('role') == 'deaf':
                    if result.is_final:
                        logger.info(f"Final Transcript for {sid} (relaying to DEAF user {other_user_sid}): {transcript}")
                        await sio.emit('transcribed_text', {'text': transcript, 'isFinal': True}, room=other_user_sid)
                    else:
                        logger.info(f"Partial Transcript for {sid} (relaying to DEAF user {other_user_sid}): {transcript}")
                        await sio.emit('transcribed_text', {'text': transcript, 'isFinal': False}, room=other_user_sid)
                else:
                    logger.info(f"Transcript for {sid} (not relayed to {other_user_sid} - not deaf or role missing): {transcript}")
            else:
                # Log that transcription is happening but not relayed yet
                logger.info(f"Transcript for {sid} (not yet relayed - no peer or room size not 2): {transcript} (Room size: {len(rooms.get(room_id, [])) if room_id else 'N/A'})")

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

# --- ASL Keypoint Streaming Event Handler ---
@sio.on('asl_keypoints')
async def handle_asl_keypoints(sid, keypoint_data):
    if sid not in keypoint_buffers:
        keypoint_buffers[sid] = []

    keypoint_buffers[sid].append(keypoint_data)

    # Keep buffer size manageable (e.g., last MAX_KEYPOINT_BUFFER_SIZE frames)
    if len(keypoint_buffers[sid]) > MAX_KEYPOINT_BUFFER_SIZE:
        keypoint_buffers[sid] = keypoint_buffers[sid][-MAX_KEYPOINT_BUFFER_SIZE:]

    # Log a sample of the data to avoid overwhelming the console
    if keypoint_data and keypoint_data.get('pose') and len(keypoint_data['pose']) > 0:
        logger.info(f"Received ASL keypoints from {sid}. Buffer size: {len(keypoint_buffers[sid])}. Sample pose landmark: {keypoint_data['pose'][0]}")
    else:
        logger.info(f"Received ASL keypoints from {sid}. Buffer size: {len(keypoint_buffers[sid])}. No pose data or empty.")

    # Now, call the model with the current buffer (sequence of frames)
    # This is where the model inference will happen in Task B4
    # For now, it will use the placeholder model
    predicted_label, confidence = asl_recognizer.predict(keypoint_buffers[sid])
    logger.info(f"ASL Prediction for {sid}: '{predicted_label}' with confidence {confidence:.2f}")


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
