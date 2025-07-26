import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import socketio
import asyncio
import logging
import numpy as np
import tensorflow as tf # Required for building the LSTM model structure
import random # For simulating predictions
import os # NEW: For accessing environment variables

# Import SpeechAsyncClient for asynchronous operations (STT remains Google Cloud)
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import SpeechAsyncClient

# NEW: Import httpx for making asynchronous HTTP requests to Eleven Labs
import httpx

# Removed: from google.cloud import texttospeech_v1beta1 as texttospeech
from asyncio import Queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
# Create FastAPI app
app = FastAPI()

# Wrap FastAPI app with Socket.IO ASGI app
app_with_sio = socketio.ASGIApp(sio, app)

# HTML for the root endpoint (unchanged)
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
    # This uses the internal sio.manager.rooms structure which is fine for status.
    return {
        "status": "active",
        "active_rooms": list(sio.manager.rooms['/'].keys()),
        "connected_users": len(sio.manager.connected_sockets) # Total connected clients
    }

# Dictionary to store audio queues for each user (for STT)
audio_queues = {}
# Dictionary to store speech recognition streams for each user (for STT)
speech_streams = {}
# Dictionary to store ASL keypoint buffers for each user
keypoint_buffers = {}

# --- ASL Recognition Model Placeholder ---
class ASLRecognizer:
    def __init__(self):
        # Define the specific ASL signs we want to recognize
        self.actions = ['HELLO', 'GOODBYE', 'YES', 'NO', 'THANK YOU', 'UNKNOWN'] # Added UNKNOWN for clarity
        self.sequence_length = 30  # Number of frames (keypoint sets) to consider for one sign
        
        # Calculate the number of features per frame based on MediaPipe Holistic output
        # Pose: 33 landmarks * 4 (x,y,z,visibility) = 132
        # Left Hand: 21 landmarks * 3 (x,y,z) = 63
        # Right Hand: 21 landmarks * 3 (x,y,z) = 63
        self.num_features = 132 + 63 + 63 # Total 258 features per frame

        # Placeholder for a real TensorFlow/Keras LSTM model
        try:
            self.model = self._build_dummy_model()
            logger.info("ASLRecognizer dummy model built successfully.")
        except Exception as e:
            logger.error(f"Failed to build dummy ASL model: {e}")
            self.model = None # Ensure model is None if building fails

    def _build_dummy_model(self):
        """Builds a dummy Keras Sequential model with LSTM layers for shape compatibility."""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.sequence_length, self.num_features)),
            tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
            tf.keras.layers.LSTM(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.actions), activation='softmax') # Output for each action
        ])
        # The model is not compiled or trained, it's just for structural representation.
        return model

    def preprocess_keypoints(self, keypoint_data):
        """
        Flattens and normalizes keypoint data from a single frame into a 1D numpy array.
        Ensures consistent feature order and handles missing data.
        """
        features = []

        # Process Pose landmarks (33 * 4 = 132 features)
        if keypoint_data and keypoint_data.get('pose'):
            for lm in keypoint_data['pose']:
                features.extend([lm['x'], lm['y'], lm['z'], lm['visibility']])
        else:
            features.extend([0.0] * (33 * 4)) # Pad with zeros if no pose data

        # Process Left Hand landmarks (21 * 3 = 63 features)
        if keypoint_data and keypoint_data.get('leftHand'):
            for lm in keypoint_data['leftHand']:
                features.extend([lm['x'], lm['y'], lm['z']])
        else:
            features.extend([0.0] * (21 * 3)) # Pad with zeros if no left hand data

        # Process Right Hand landmarks (21 * 3 = 63 features)
        if keypoint_data and keypoint_data.get('rightHand'):
            for lm in keypoint_data['rightHand']:
                features.extend([lm['x'], lm['y'], lm['z']])
        else:
            features.extend([0.0] * (21 * 3)) # Pad with zeros if no right hand data
        
        # Ensure the feature vector has the expected length
        if len(features) != self.num_features:
            logger.warning(f"Feature vector length mismatch: Expected {self.num_features}, got {len(features)}. Adjusting.")
            # Adjust or raise error as needed, for now, pad/truncate to expected size
            if len(features) > self.num_features:
                features = features[:self.num_features]
            else:
                features.extend([0.0] * (self.num_features - len(features)))

        return np.array(features, dtype=np.float32)

    def predict(self, keypoint_sequence):
        """
        Simulates ASL sign prediction from a sequence of keypoints.
        In a real scenario, this would perform inference using the loaded model.
        """
        if not self.model:
            return 'MODEL_ERROR', 0.0

        # If not enough frames for a full sequence, return 'UNKNOWN'
        if len(keypoint_sequence) < self.sequence_length:
            return 'UNKNOWN', 0.0 # Confidence 0.0 for UNKNOWN

        # Take the most recent 'sequence_length' frames
        sequence_to_predict = keypoint_sequence[-self.sequence_length:]
        
        # Convert list of dicts to a NumPy array of shape (sequence_length, num_features)
        processed_sequence = np.array([self.preprocess_keypoints(frame_data) for frame_data in sequence_to_predict])
        
        # Reshape for model input: (1, sequence_length, num_features)
        model_input = np.expand_dims(processed_sequence, axis=0)

        # --- Simulate Model Prediction ---
        # In a real model, you'd do:
        # predictions = self.model.predict(model_input)[0]
        # predicted_index = np.argmax(predictions)
        # confidence = predictions[predicted_index]
        # predicted_label = self.actions[predicted_index]

        # For simulation, generate random probabilities
        simulated_predictions = np.random.rand(len(self.actions))
        simulated_predictions = simulated_predictions / simulated_predictions.sum() # Normalize to sum to 1

        predicted_index = np.argmax(simulated_predictions)
        predicted_label = self.actions[predicted_index]
        confidence = simulated_predictions[predicted_index]

        # Simulate 'UNKNOWN' more often if confidence is low, or based on random chance
        if confidence < 0.6 or np.random.rand() < 0.3: # 30% chance of 'UNKNOWN' or if confidence is low
            predicted_label = 'UNKNOWN'
            confidence = 0.0 # Set confidence to 0 for UNKNOWN

        logger.info(f"Simulated prediction: {predicted_label} with confidence {confidence:.2f}")

        return predicted_label, float(confidence) # Return confidence as a standard float

# Initialize the ASL Recognizer
asl_recognizer = ASLRecognizer()

# --- Socket.IO Event Handlers ---
@sio.on('connect')
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

@sio.on('disconnect')
async def disconnect(sid, reason):
    logger.info(f"Client disconnected: {sid} (Reason: {reason})")
    
    session = await sio.get_session(sid)
    room_id = session.get('room_id')
    user_role = session.get('user_role')

    if sid in speech_streams:
        if sid in audio_queues:
            await audio_queues[sid].put(None)
            del audio_queues[sid]
        del speech_streams[sid]

    if sid in keypoint_buffers:
        del keypoint_buffers[sid]
    
    if room_id:
        room_clients_before_disconnect = sio.manager.rooms['/'].get(room_id, set())
        
        await sio.leave_room(sid, room_id)
        logger.info(f"Client {sid} left room {room_id}.")

        remaining_users_in_room = sio.manager.rooms['/'].get(room_id, set())
        if remaining_users_in_room:
            other_sid = list(remaining_users_in_room)[0]
            await sio.emit('user-left', {'userId': sid}, room=other_sid)
            logger.info(f"Notified {other_sid} that {sid} left room {room_id}.")
        else:
            logger.info(f"Room {room_id} is now empty.")
    

@sio.on('join_room')
async def join_room(sid, data):
    room_id = data.get('roomId')
    user_role = data.get('userRole')
    
    if not room_id or not user_role:
        await sio.emit('error', {'message': 'Room ID and user role are required'}, room=sid)
        logger.warning(f"Client {sid} tried to join room without roomId or userRole.")
        return

    current_room_participants = sio.manager.rooms['/'].get(room_id, set())
    
    if len(current_room_participants) >= 2:
        await sio.emit('room-full', room=sid)
        logger.warning(f"Room {room_id} is full. User {sid} cannot join.")
        return

    await sio.enter_room(sid, room_id)
    
    session = await sio.get_session(sid)
    session['room_id'] = room_id
    session['user_role'] = user_role
    await sio.save_session(sid, session)

    all_participants_in_room = sio.manager.rooms['/'].get(room_id, set())
    current_room_size = len(all_participants_in_room)
    logger.info(f"User {sid} (Role: {user_role}) joined room {room_id}. Current room size: {current_room_size}")

    other_sids_in_room = [client_sid for client_sid in all_participants_in_room if client_sid != sid]
    
    if other_sids_in_room:
        for other_sid in other_sids_in_room:
            await sio.emit('user-joined', {'userId': sid}, room=other_sid)
            logger.info(f"Notified {other_sid} about new user {sid} joining room {room_id}.")
        await sio.emit('user-ready', {'userId': other_sids_in_room[0]}, room=sid)
        logger.info(f"Notified {sid} about existing user {other_sids_in_room[0]} in room {room_id}.")
    else:
        await sio.emit('user-ready', {'userId': sid}, room=sid)
        logger.info(f"User {sid} is the first in room {room_id}.")


# --- Speech-to-Text (STT) Event Handlers ---
@sio.on('start_audio_stream')
async def start_audio_stream_handler(sid, data=None):
    logger.info(f"Received start_audio_stream from {sid}")
    session = await sio.get_session(sid)
    user_role = session.get('user_role')

    if user_role != 'hearing':
        logger.warning(f"User {sid} (role: {user_role}) attempted to start audio stream but is not a 'hearing' user.")
        await sio.emit('audio-stream-error', {'error': 'Only hearing users can stream audio for STT'}, room=sid)
        return

    if sid in speech_streams and speech_streams[sid] is not None:
        logger.warning(f"Speech stream already active for {sid}. Skipping start.")
        await sio.emit('audio-stream-started', {'status': 'success'}, room=sid)
        return

    sio.start_background_task(run_stt_stream, sid)
    await sio.emit('audio-stream-started', {'status': 'success'}, room=sid)

async def run_stt_stream(sid):
    """
    Handles the actual Google Cloud Speech-to-Text streaming for a given SID.
    This function should be run as a background task.
    """
    logger.info(f"Starting background STT stream for {sid}")
    session = await sio.get_session(sid)
    room_id = session.get('room_id')

    try:
        client = SpeechAsyncClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="default"
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        if sid not in audio_queues:
            audio_queues[sid] = asyncio.Queue()
        audio_queue = audio_queues[sid]

        async def request_generator():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    logger.info(f"Received None chunk for {sid}, stopping STT request generator.")
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        responses = await client.streaming_recognize(request_generator())
        speech_streams[sid] = responses

        async for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            transcript = result.alternatives[0].transcript
            is_final = result.is_final

            logger.info(f"STT Result for {sid}: {'(Final)' if is_final else '(Interim)'} {transcript}")
            
            if room_id:
                room_clients = sio.manager.rooms['/'].get(room_id, set())
                for client_sid in room_clients:
                    client_session = await sio.get_session(client_sid)
                    if client_session.get('user_role') == 'deaf' and client_sid != sid:
                        await sio.emit('transcribed_text', {'text': transcript, 'isFinal': is_final}, room=client_sid)
                        break

    except Exception as e:
        logger.error(f"Error in background STT stream for {sid}: {e}", exc_info=True)
        await sio.emit('audio-stream-error', {'error': str(e)}, room=sid)
    finally:
        logger.info(f"Background STT stream for {sid} finished/stopped.")
        if sid in audio_queues:
            if not audio_queues[sid].empty():
                try:
                    await audio_queues[sid].put(None)
                except Exception:
                    pass
            del audio_queues[sid]
        if sid in speech_streams:
            del speech_streams[sid]


@sio.on('send_audio_chunk')
async def send_audio_chunk(sid, data):
    if sid in audio_queues:
        await audio_queues[sid].put(data)
    else:
        logger.warning(f"Audio queue not found for {sid}. Chunk discarded. Has start_audio_stream been called?")

@sio.on('end_audio_stream')
async def end_audio_stream(sid):
    logger.info(f"Received end_audio_stream from {sid}")
    if sid in audio_queues:
        await audio_queues[sid].put(None)
        del audio_queues[sid]
    await sio.emit('audio-stream-stopped', {'status': 'success'}, room=sid)

# --- NEW: Eleven Labs Text-to-Speech (TTS) Function ---
async def synthesize_speech_elevenlabs(text_to_synthesize: str) -> bytes:
    """
    Synthesizes speech from the given text using Eleven Labs API.
    Returns the raw audio content as bytes (MP3 format).
    """
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    if not ELEVENLABS_API_KEY:
        logger.error("ELEVENLABS_API_KEY environment variable not set.")
        return b''

    # You can find voice_id in Eleven Labs dashboard -> VoiceLab -> Add Voice -> Use Pre-made Voice
    # Example: 'Rachel' voice_id
    VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # Replace with your preferred voice ID

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text_to_synthesize,
        "model_id": "eleven_monolingual_v1", # Or "eleven_multilingual_v2" if needed
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=10.0) # Added timeout
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            logger.info(f"Successfully synthesized speech for text: '{text_to_synthesize}' via Eleven Labs.")
            return response.content # Returns MP3 audio bytes

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error synthesizing speech with Eleven Labs for '{text_to_synthesize}': {e.response.status_code} - {e.response.text}", exc_info=True)
        return b''
    except httpx.RequestError as e:
        logger.error(f"Network error synthesizing speech with Eleven Labs for '{text_to_synthesize}': {e}", exc_info=True)
        return b''
    except Exception as e:
        logger.error(f"Unexpected error synthesizing speech with Eleven Labs for '{text_to_synthesize}': {e}", exc_info=True)
        return b''

# --- ASL Keypoint Event Handler ---
@sio.on('asl_keypoints')
async def asl_keypoints(sid, keypoint_data):
    session = await sio.get_session(sid)
    user_role = session.get('user_role')

    if user_role != 'deaf':
        # Silently ignore if not a deaf user, as this is a continuous stream and not an error
        return 

    # Initialize buffer if not exists
    if sid not in keypoint_buffers:
        keypoint_buffers[sid] = []

    # Add current frame's keypoints to the buffer
    keypoint_buffers[sid].append(keypoint_data)

    # Keep the buffer limited to the sequence length required by the model
    if len(keypoint_buffers[sid]) > asl_recognizer.sequence_length:
        keypoint_buffers[sid].pop(0) # Remove the oldest frame

    # Only attempt prediction if we have enough frames for a full sequence
    if len(keypoint_buffers[sid]) < asl_recognizer.sequence_length:
        # Not enough data for a full sign, send UNKNOWN or CLEAR if previous was a sign
        last_prediction_info = session.get('last_asl_prediction', {'label': None, 'confidence': 0.0})
        if last_prediction_info['label'] not in ['UNKNOWN', 'CLEAR', None]:
            logger.info(f"ASL Prediction for {sid}: Not enough data, sending CLEAR.")
            room_id = session.get('room_id')
            if room_id:
                room_clients = sio.manager.rooms['/'].get(room_id, set())
                for client_sid in room_clients:
                    client_session = await sio.get_session(client_sid)
                    if client_session.get('user_role') == 'hearing' and client_sid != sid:
                        await sio.emit('asl_prediction', {'sign': 'CLEAR', 'confidence': 0.0}, room=client_sid)
                        # Optionally, send a silent audio chunk or a specific signal to stop TTS playback on frontend
                        # await sio.emit('synthesized_audio_chunk', b'', room=client_sid)
                        break
            session = await sio.get_session(sid)
            session['last_asl_prediction'] = {'label': 'CLEAR', 'confidence': 0.0}
            await sio.save_session(sid, session)
        return # Exit early if not enough data

    # Now, call the model with the current buffer (sequence of frames)
    predicted_label, confidence = asl_recognizer.predict(keypoint_buffers[sid])
    
    last_prediction_info = session.get('last_asl_prediction', {'label': None, 'confidence': 0.0})
    
    should_emit = False
    if predicted_label != 'UNKNOWN' and confidence > 0.5:
        if predicted_label != last_prediction_info['label'] or confidence > last_prediction_info['confidence'] + 0.1:
            should_emit = True
    elif predicted_label == 'UNKNOWN' and last_prediction_info['label'] not in ['UNKNOWN', 'CLEAR', None]:
        predicted_label = 'CLEAR' # Use 'CLEAR' for frontend signal
        confidence = 0.0
        should_emit = True

    if should_emit:
        logger.info(f"ASL Prediction for {sid}: '{predicted_label}' with confidence {confidence:.2f}")
        
        room_id = session.get('room_id')
        if room_id:
            room_clients = sio.manager.rooms['/'].get(room_id, set())
            for client_sid in room_clients:
                client_session = await sio.get_session(client_sid)
                if client_session.get('user_role') == 'hearing' and client_sid != sid:
                    # Emit the text prediction to the hearing user
                    await sio.emit('asl_prediction', {
                        'sign': predicted_label,
                        'confidence': confidence
                    }, room=client_sid)

                    # NEW: If a valid sign is predicted (not UNKNOWN/CLEAR), synthesize speech and send it
                    if predicted_label not in ['UNKNOWN', 'CLEAR']:
                        # Run TTS in a background task to avoid blocking the ASL keypoint processing
                        sio.start_background_task(send_synthesized_speech, client_sid, predicted_label)
                    break
        
        session = await sio.get_session(sid)
        session['last_asl_prediction'] = {'label': predicted_label, 'confidence': confidence}
        await sio.save_session(sid, session)

# NEW: Background task to synthesize and send speech
async def send_synthesized_speech(target_sid: str, text: str):
    """Synthesizes speech and sends it to the target SID."""
    logger.info(f"Attempting to synthesize speech for '{text}' and send to {target_sid}")
    # Call the Eleven Labs synthesis function
    audio_content = await synthesize_speech_elevenlabs(text)
    if audio_content:
        # Send the raw audio bytes. Frontend will need to play this.
        # IMPORTANT: Eleven Labs returns MP3. Frontend will need to handle MP3 playback.
        await sio.emit('synthesized_audio_chunk', audio_content, room=target_sid)
        logger.info(f"Sent {len(audio_content)} bytes of synthesized audio for '{text}' to {target_sid}")
    else:
        logger.warning(f"No audio content synthesized for '{text}'. Not sending.")


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

if __name__ == "__main__":
    # IMPORTANT: Use 0.0.0.0 for host to make it accessible from outside the container/VM
    # Use the port your backend is configured to listen on (e.g., 8000)
    # This command is correct and routes all traffic (HTTP and Socket.IO) through app_with_sio
    uvicorn.run(app_with_sio, host="0.0.0.0", port=8000)
