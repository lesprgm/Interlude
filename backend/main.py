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
    return {
        "status": "active",
        "active_rooms": list(sio.manager.rooms['/'].keys()),
        "connected_users": len(sio.manager.connected_sockets)
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
        self.actions = ['HELLO', 'GOODBYE', 'YES', 'NO', 'THANK YOU']
        self.sequence_length = 30  # Number of frames (keypoint sets) to consider for one sign
        
        # Calculate the number of features per frame based on MediaPipe Holistic output
        # Pose: 33 landmarks * 4 (x,y,z,visibility) = 132
        # Left Hand: 21 landmarks * 3 (x,y,z) = 63
        # Right Hand: 21 landmarks * 3 (x,y,z) = 63
        self.num_features = 132 + 63 + 63 # Total 258 features per frame

        # Placeholder for a real TensorFlow/Keras LSTM model
        # This model is not trained but has the correct input/output shape
        # In a real scenario, you would load pre-trained weights here:
        # self.model = tf.keras.models.load_model('path/to/your/trained_asl_model.h5')
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
            logger.warning(f"Feature vector length mismatch: Expected {self.num_features}, got {len(features)}")
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

        if len(keypoint_sequence) < self.sequence_length:
            return 'UNKNOWN', 0.84 # Not enough data for a full sign

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

        logger.info(f"Simulated prediction: {predicted_label} with confidence {confidence:.2f}")

        return predicted_label, float(confidence) # Return confidence as a standard float

# Initialize the ASL Recognizer
asl_recognizer = ASLRecognizer()

# --- Socket.IO Event Handlers ---
@sio.on('connect')
async def connect(sid, environ):
    logger.info(f"Connect: {sid}")

@sio.on('disconnect')
async def disconnect(sid):
    logger.info(f"Disconnect: {sid}")
    # Clean up resources associated with the disconnected user
    if sid in audio_queues:
        del audio_queues[sid]
    if sid in speech_streams:
        # Close the speech stream if it exists
        if speech_streams[sid]:
            await speech_streams[sid].close()
        del speech_streams[sid]
    if sid in keypoint_buffers:
        del keypoint_buffers[sid]
    
    # Notify other users in the room that a user has left
    for room_id in sio.manager.rooms['/'].keys():
        if sid in sio.manager.rooms['/'][room_id]:
            # Get other user's sid in the room
            other_sids = [s for s in sio.manager.rooms['/'][room_id] if s != sid]
            for other_sid in other_sids:
                await sio.emit('user-left', {'userId': sid}, room=other_sid)
            break # User can only be in one room

@sio.on('join_room')
async def join_room(sid, data):
    room_id = data.get('roomId')
    user_role = data.get('userRole')
    
    if not room_id:
        await sio.emit('error', {'message': 'Room ID is required'}, room=sid)
        return

    # Defensive check: Ensure the room entry exists in the manager's rooms dictionary
    # before attempting to access or add to it. This explicitly handles cases where
    # `sio.enter_room` might internally struggle if the key is not present initially.
    if room_id not in sio.manager.rooms['/']:
        sio.manager.rooms['/'][room_id] = set() # Initialize with an empty set
        logger.info(f"Created new room entry for {room_id} in sio.manager.rooms.")

    # Check current room size (max 2 users)
    # Now we can directly access it since we ensured it exists
    room_clients_before_join = sio.manager.rooms['/'][room_id]
    
    if len(room_clients_before_join) >= 2:
        await sio.emit('room-full', room=sid)
        logger.warning(f"Room {room_id} is full. User {sid} cannot join.")
        return

    # Join the room
    sio.enter_room(sid, room_id)
    
    # After entering the room, get the current number of participants
    current_room_size = len(sio.get_participants(room=room_id)) # More reliable way to get current room size
    # Corrected logger.info line to use current_room_size
    logger.info(f"User {sid} (Role: {user_role}) joined room {room_id}. Current room size: {current_room_size}")

    # Store user role and room for later use
    sio.update_session(sid, {'room_id': room_id, 'user_role': user_role})

    # Notify other users in the room
    # Get participants *after* the current user has joined
    other_sids_in_room = [client_sid for client_sid in sio.get_participants(room=room_id) if client_sid != sid]
    if other_sids_in_room:
        # If there's another user, tell them a new user joined
        for other_sid in other_sids_in_room:
            await sio.emit('user-joined', {'userId': sid}, room=other_sid)
        # And tell the new user about the existing user
        await sio.emit('user-ready', {'userId': other_sids_in_room[0]}, room=sid)
    else:
        # If no other users, just confirm join
        await sio.emit('user-ready', {'userId': sid}, room=sid) # Send own ID to confirm join

# --- Speech-to-Text (STT) Event Handlers ---
@sio.on('start_audio_stream')
async def start_audio_stream(sid, data=None): # Accept data argument
    logger.info(f"Received start_audio_stream from {sid}")
    session = await sio.get_session(sid)
    # Prefer role from event data if provided, fallback to session
    user_role = data.get('userRole') if data and 'userRole' in data else session.get('user_role')

    if user_role != 'hearing':
        logger.warning(f"User {sid} (role: {user_role}) attempted to start audio stream but is not a 'hearing' user.")
        await sio.emit('audio-stream-error', {'error': 'Only hearing users can stream audio for STT'}, room=sid)
        return

    if sid in speech_streams and speech_streams[sid] is not None:
        logger.warning(f"Speech stream already active for {sid}")
        await sio.emit('audio-stream-started', {'status': 'success'}, room=sid)
        return

    try:
        # Initialize the Google Cloud Speech-to-Text client
        client = SpeechAsyncClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000, # Adjust if your MediaRecorder uses a different rate
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="default" # or "video", "phone_call" for specific use cases
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True # Get interim results for real-time display
        )

        # Create a new audio queue for this session
        audio_queue = asyncio.Queue()
        audio_queues[sid] = audio_queue

        async def request_generator():
            while True:
                chunk = await audio_queue.get()
                if chunk is None: # Signal to close the stream
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        # Start the streaming recognition
        responses = await client.streaming_recognize(config=streaming_config, requests=request_generator())
        speech_streams[sid] = responses # Store the response iterator

        await sio.emit('audio-stream-started', {'status': 'success'}, room=sid)

        # Process responses from STT API
        async for response in responses:
            if not response.results:
                continue

            # The transcript is in the first result alternative.
            result = response.results[0]
            transcript = result.alternatives[0].transcript
            is_final = result.is_final

            logger.info(f"STT Result for {sid}: {'(Final)' if is_final else '(Interim)'} {transcript}")
            
            # Emit transcribed text to the room (for the deaf user)
            room_id = session.get('room_id')
            if room_id:
                # Find the 'deaf' user in the room
                room_clients = sio.manager.rooms['/'].get(room_id, set())
                for client_sid in room_clients:
                    client_session = await sio.get_session(client_sid)
                    if client_session.get('user_role') == 'deaf' and client_sid != sid:
                        await sio.emit('transcribed_text', {'text': transcript, 'isFinal': is_final}, room=client_sid)
                        break # Assume only one deaf user per room for now

        logger.info(f"Speech stream for {sid} finished processing.")

    except Exception as e:
        logger.error(f"Error in start_audio_stream for {sid}: {e}", exc_info=True)
        await sio.emit('audio-stream-error', {'error': str(e)}, room=sid)
        # Clean up in case of error
        if sid in audio_queues:
            del audio_queues[sid]
        if sid in speech_streams:
            if speech_streams[sid]:
                await speech_streams[sid].close()
            del speech_streams[sid]


@sio.on('send_audio_chunk')
async def send_audio_chunk(sid, data):
    # logger.info(f"Received audio chunk from {sid}, size: {len(data)} bytes")
    if sid in audio_queues:
        await audio_queues[sid].put(data)
        await sio.emit('audio-chunk-received', {'status': 'success'}, room=sid)
    else:
        logger.warning(f"Audio queue not found for {sid}. Chunk discarded.")

@sio.on('end_audio_stream')
async def end_audio_stream(sid):
    logger.info(f"Received end_audio_stream from {sid}")
    if sid in audio_queues:
        await audio_queues[sid].put(None) # Signal to close the generator
        del audio_queues[sid]
    if sid in speech_streams:
        # The async generator will stop upon receiving None,
        # so we just need to remove the stream from our dict.
        # The client.streaming_recognize context manager handles closing.
        del speech_streams[sid]
    await sio.emit('audio-stream-stopped', {'status': 'success'}, room=sid)

# --- ASL Keypoint Event Handler ---
@sio.on('asl_keypoints')
async def asl_keypoints(sid, keypoint_data):
    session = await sio.get_session(sid)
    user_role = session.get('user_role')

    if user_role != 'deaf':
        # logger.warning(f"User {sid} (role: {user_role}) attempted to send ASL keypoints but is not a 'deaf' user.")
        return # Silently ignore if not a deaf user, as this is a continuous stream

    # Initialize buffer if not exists
    if sid not in keypoint_buffers:
        keypoint_buffers[sid] = []

    # Add current frame's keypoints to the buffer
    keypoint_buffers[sid].append(keypoint_data)

    # Keep the buffer limited to the sequence length required by the model
    if len(keypoint_buffers[sid]) > asl_recognizer.sequence_length:
        keypoint_buffers[sid].pop(0) # Remove the oldest frame

    # Log sample data for verification (optional, can be removed for production)
    # if keypoint_data.get('pose') and len(keypoint_data['pose']) > 0:
    #     logger.info(f"Received ASL keypoints from {sid}. Buffer size: {len(keypoint_buffers[sid])}. Sample pose landmark: {keypoint_data['pose'][0]}")
    # else:
    #     logger.info(f"Received ASL keypoints from {sid}. Buffer size: {len(keypoint_buffers[sid])}. No pose data or empty.")

    # Now, call the model with the current buffer (sequence of frames)
    predicted_label, confidence = asl_recognizer.predict(keypoint_buffers[sid])
    
    # Only send prediction if it's not 'UNKNOWN' and confidence is reasonable,
    # or if it's a new prediction different from the last one sent.
    # This prevents spamming the frontend with 'UNKNOWN' or redundant predictions.
    last_prediction_info = session.get('last_asl_prediction', {'label': None, 'confidence': 0.0})
    
    if predicted_label != 'UNKNOWN' and confidence > 0.5 and \
       (predicted_label != last_prediction_info['label'] or confidence > last_prediction_info['confidence'] + 0.1): # Send if new sign or significantly more confident
        
        logger.info(f"ASL Prediction for {sid}: '{predicted_label}' with confidence {confidence:.2f}")
        
        # Emit the prediction to the hearing user in the same room
        room_id = session.get('room_id')
        if room_id:
            room_clients = sio.manager.rooms['/'].get(room_id, set())
            for client_sid in room_clients:
                client_session = await sio.get_session(client_sid)
                if client_session.get('user_role') == 'hearing' and client_sid != sid:
                    await sio.emit('asl_prediction', {
                        'sign': predicted_label,
                        'confidence': confidence
                    }, room=client_sid)
                    break # Assume one hearing user per room for now
        
        # Update last prediction in session
        sio.update_session(sid, {'last_asl_prediction': {'label': predicted_label, 'confidence': confidence}})
    elif predicted_label == 'UNKNOWN' and last_prediction_info['label'] != 'UNKNOWN':
        # If it transitions back to UNKNOWN after a sign, clear the last prediction
        sio.update_session(sid, {'last_asl_prediction': {'label': 'UNKNOWN', 'confidence': 0.0}})
        # Optionally, send a 'clear sign' signal to hearing user
        room_id = session.get('room_id')
        if room_id:
            room_clients = sio.manager.rooms['/'].get(room_id, set())
            for client_sid in room_clients:
                client_session = await sio.get_session(client_sid)
                if client_session.get('user_role') == 'hearing' and client_sid != sid:
                    await sio.emit('asl_prediction', {'sign': 'CLEAR', 'confidence': 0.0}, room=client_sid)
                    break

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
    uvicorn.run(app_with_sio, host="0.0.0.0", port=8000)
