import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import socketio
import asyncio
import logging
import numpy as np
import tensorflow as tf # Required for building the LSTM model structure
import random # For simulating predictions

# Import SpeechAsyncClient for asynchronous operations
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import SpeechAsyncClient
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

# Dictionary to store audio queues for each user
audio_queues = {}
# Dictionary to store speech recognition streams for each user
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
async def disconnect(sid, reason): # FIX: Added 'reason' argument to match Socket.IO signature
    logger.info(f"Client disconnected: {sid} (Reason: {reason})")
    
    # Retrieve room_id and user_role from session before cleanup
    session = await sio.get_session(sid)
    room_id = session.get('room_id')
    user_role = session.get('user_role')

    # Clean up STT stream if exists
    if sid in speech_streams:
        # Signal the generator to stop by putting None into its queue
        if sid in audio_queues:
            await audio_queues[sid].put(None)
            del audio_queues[sid]
        # The async for loop in run_stt_stream will then exit gracefully
        del speech_streams[sid]

    # Clean up keypoint buffer
    if sid in keypoint_buffers:
        del keypoint_buffers[sid]
    
    # Explicitly leave the room
    if room_id:
        # Get current participants in the room before leaving to notify others
        # FIX: Use sio.manager.rooms for robust participant check
        room_clients_before_disconnect = sio.manager.rooms['/'].get(room_id, set())
        
        await sio.leave_room(sid, room_id) # AWAIT this call
        logger.info(f"Client {sid} left room {room_id}.")

        # Notify remaining user in the room that a user left
        # Get participants *after* the current user has left
        # FIX: Use sio.manager.rooms for robust participant check
        remaining_users_in_room = sio.manager.rooms['/'].get(room_id, set())
        if remaining_users_in_room:
            # There should be only one other user if it's a 2-person room
            other_sid = list(remaining_users_in_room)[0] # Get the only remaining SID
            await sio.emit('user-left', {'userId': sid}, room=other_sid)
            logger.info(f"Notified {other_sid} that {sid} left room {room_id}.")
        else:
            # If the room becomes empty, Socket.IO manager will clean it up internally,
            # but we can log for clarity.
            logger.info(f"Room {room_id} is now empty.")
    
    # FIX: Removed await sio.delete_session(sid) as it's not a valid method
    # Socket.IO handles session cleanup automatically on disconnect.


@sio.on('join_room')
async def join_room(sid, data):
    room_id = data.get('roomId')
    user_role = data.get('userRole')
    
    if not room_id or not user_role: # Ensure user_role is also provided
        await sio.emit('error', {'message': 'Room ID and user role are required'}, room=sid)
        logger.warning(f"Client {sid} tried to join room without roomId or userRole.")
        return

    # Get current room participants using the correct method
    # FIX: Use sio.manager.rooms to get participants
    current_room_participants = sio.manager.rooms['/'].get(room_id, set())
    
    if len(current_room_participants) >= 2:
        await sio.emit('room-full', room=sid)
        logger.warning(f"Room {room_id} is full. User {sid} cannot join.")
        return

    # Join the room - AWAIT this call
    await sio.enter_room(sid, room_id)
    
    # Store user role and room in the session for later retrieval
    # FIX: Correct session management: get, modify, then save
    session = await sio.get_session(sid)
    session['room_id'] = room_id
    session['user_role'] = user_role
    await sio.save_session(sid, session) # AWAIT this call

    # Get participants *after* the current user has joined, using the correct method
    # FIX: Use sio.manager.rooms to get participants
    all_participants_in_room = sio.manager.rooms['/'].get(room_id, set())
    current_room_size = len(all_participants_in_room)
    logger.info(f"User {sid} (Role: {user_role}) joined room {room_id}. Current room size: {current_room_size}")

    # Notify other users in the room
    other_sids_in_room = [client_sid for client_sid in all_participants_in_room if client_sid != sid]
    
    if other_sids_in_room:
        # If there's another user, tell them a new user joined
        for other_sid in other_sids_in_room:
            await sio.emit('user-joined', {'userId': sid}, room=other_sid)
            logger.info(f"Notified {other_sid} about new user {sid} joining room {room_id}.")
        # And tell the new user about the existing user (the initiator)
        await sio.emit('user-ready', {'userId': other_sids_in_room[0]}, room=sid)
        logger.info(f"Notified {sid} about existing user {other_sids_in_room[0]} in room {room_id}.")
    else:
        # If no other users, just confirm join to the current user
        await sio.emit('user-ready', {'userId': sid}, room=sid) # Send own ID to confirm join
        logger.info(f"User {sid} is the first in room {room_id}.")


# --- Speech-to-Text (STT) Event Handlers ---
@sio.on('start_audio_stream')
async def start_audio_stream_handler(sid, data=None): # Renamed to avoid confusion with the async function below
    logger.info(f"Received start_audio_stream from {sid}")
    session = await sio.get_session(sid)
    user_role = session.get('user_role') # Get role from session

    if user_role != 'hearing':
        logger.warning(f"User {sid} (role: {user_role}) attempted to start audio stream but is not a 'hearing' user.")
        await sio.emit('audio-stream-error', {'error': 'Only hearing users can stream audio for STT'}, room=sid)
        return

    if sid in speech_streams and speech_streams[sid] is not None:
        logger.warning(f"Speech stream already active for {sid}. Skipping start.")
        await sio.emit('audio-stream-started', {'status': 'success'}, room=sid)
        return

    # IMPORTANT FIX: Run the actual STT processing in a background task
    # This prevents the Socket.IO event handler from blocking while waiting for STT responses.
    sio.start_background_task(run_stt_stream, sid)
    await sio.emit('audio-stream-started', {'status': 'success'}, room=sid)

async def run_stt_stream(sid):
    """
    Handles the actual Google Cloud Speech-to-Text streaming for a given SID.
    This function should be run as a background task.
    """
    logger.info(f"Starting background STT stream for {sid}")
    session = await sio.get_session(sid) # Re-fetch session to ensure latest data
    room_id = session.get('room_id')

    try:
        client = SpeechAsyncClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000, # Ensure this matches frontend MediaRecorder
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="default" # "video" or "phone_call" might be better depending on audio characteristics
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True # Get interim results for real-time display
        )

        # Create a new audio queue for this session if it doesn't exist
        if sid not in audio_queues:
            audio_queues[sid] = asyncio.Queue()
        audio_queue = audio_queues[sid]

        # FIX: The first request must contain the streaming_config.
        # Subsequent requests only contain audio_content.
        async def request_generator():
            # Yield the first request with the config
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            
            # Then yield subsequent audio chunks
            while True:
                chunk = await audio_queue.get()
                if chunk is None: # Signal to close the stream
                    logger.info(f"Received None chunk for {sid}, stopping STT request generator.")
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        # FIX: AWAIT the client.streaming_recognize() call
        responses = await client.streaming_recognize(request_generator())
        speech_streams[sid] = responses # Store the response iterator for cleanup

        async for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            transcript = result.alternatives[0].transcript
            is_final = result.is_final

            logger.info(f"STT Result for {sid}: {'(Final)' if is_final else '(Interim)'} {transcript}")
            
            # Emit transcribed text to the room (for the deaf user)
            if room_id:
                # Find the 'deaf' user in the room using the correct method
                room_clients = sio.manager.rooms['/'].get(room_id, set())
                for client_sid in room_clients:
                    client_session = await sio.get_session(client_sid)
                    if client_session.get('user_role') == 'deaf' and client_sid != sid:
                        await sio.emit('transcribed_text', {'text': transcript, 'isFinal': is_final}, room=client_sid)
                        break # Assume only one deaf user per room for now

    except Exception as e:
        logger.error(f"Error in background STT stream for {sid}: {e}", exc_info=True)
        await sio.emit('audio-stream-error', {'error': str(e)}, room=sid)
    finally:
        logger.info(f"Background STT stream for {sid} finished/stopped.")
        # Ensure cleanup in case of error or normal termination
        if sid in audio_queues:
            # If the stream stopped due to an error, ensure the queue is cleared
            # and potentially put None if it wasn't already to unblock generator
            if not audio_queues[sid].empty():
                try:
                    await audio_queues[sid].put(None)
                except Exception:
                    pass # Ignore if queue is already closed/deleted
            del audio_queues[sid]
        if sid in speech_streams:
            del speech_streams[sid]


@sio.on('send_audio_chunk')
async def send_audio_chunk(sid, data):
    # logger.info(f"Received audio chunk from {sid}, size: {len(data)} bytes")
    if sid in audio_queues:
        await audio_queues[sid].put(data)
        # Removed 'audio-chunk-received' emit to reduce unnecessary network traffic
        # await sio.emit('audio-chunk-received', {'status': 'success'}, room=sid)
    else:
        logger.warning(f"Audio queue not found for {sid}. Chunk discarded. Has start_audio_stream been called?")

@sio.on('end_audio_stream')
async def end_audio_stream(sid):
    logger.info(f"Received end_audio_stream from {sid}")
    if sid in audio_queues:
        await audio_queues[sid].put(None) # Signal to close the generator
        del audio_queues[sid]
    # The speech_streams entry will be cleaned up by the run_stt_stream background task's finally block
    await sio.emit('audio-stream-stopped', {'status': 'success'}, room=sid)

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
        if last_prediction_info['label'] not in ['UNKNOWN', 'CLEAR', None]: # Check if last was a real sign
            logger.info(f"ASL Prediction for {sid}: Not enough data, sending CLEAR.")
            room_id = session.get('room_id')
            if room_id:
                # FIX: Use sio.manager.rooms for robust participant check
                room_clients = sio.manager.rooms['/'].get(room_id, set())
                for client_sid in room_clients:
                    client_session = await sio.get_session(client_sid)
                    if client_session.get('user_role') == 'hearing' and client_sid != sid:
                        await sio.emit('asl_prediction', {'sign': 'CLEAR', 'confidence': 0.0}, room=client_sid)
                        break
            # FIX: Correct session management: get, modify, then save
            session = await sio.get_session(sid) # Re-fetch session to ensure latest
            session['last_asl_prediction'] = {'label': 'CLEAR', 'confidence': 0.0}
            await sio.save_session(sid, session)
        return # Exit early if not enough data

    # Now, call the model with the current buffer (sequence of frames)
    predicted_label, confidence = asl_recognizer.predict(keypoint_buffers[sid])
    
    # Optimization: Only send prediction if it's not 'UNKNOWN' and confidence is reasonable,
    # or if it's a new prediction different from the last one sent.
    last_prediction_info = session.get('last_asl_prediction', {'label': None, 'confidence': 0.0})
    
    should_emit = False
    if predicted_label != 'UNKNOWN' and confidence > 0.5:
        if predicted_label != last_prediction_info['label'] or confidence > last_prediction_info['confidence'] + 0.1:
            should_emit = True
    elif predicted_label == 'UNKNOWN' and last_prediction_info['label'] not in ['UNKNOWN', 'CLEAR', None]:
        # If current is UNKNOWN and last was a real sign, send CLEAR signal
        predicted_label = 'CLEAR' # Use 'CLEAR' for frontend signal
        confidence = 0.0
        should_emit = True

    if should_emit:
        logger.info(f"ASL Prediction for {sid}: '{predicted_label}' with confidence {confidence:.2f}")
        
        # Emit the prediction to the hearing user in the same room
        room_id = session.get('room_id')
        if room_id:
            # FIX: Use sio.manager.rooms for robust participant check
            room_clients = sio.manager.rooms['/'].get(room_id, set())
            for client_sid in room_clients:
                client_session = await sio.get_session(client_sid)
                if client_session.get('user_role') == 'hearing' and client_sid != sid:
                    await sio.emit('asl_prediction', {
                        'sign': predicted_label,
                        'confidence': confidence
                    }, room=client_sid)
                    break # Assume one hearing user per room for now
        
        # Update last prediction in session - AWAIT this call
        # FIX: Correct session management: get, modify, then save
        session = await sio.get_session(sid) # Re-fetch session to ensure latest
        session['last_asl_prediction'] = {'label': predicted_label, 'confidence': confidence}
        await sio.save_session(sid, session)


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
