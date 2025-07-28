// Interlude - Real-time Communication Aid
// Main application JavaScript

class InterludeApp {
    constructor() {
        this.localVideo = null;
        this.remoteVideo = null;
        this.localStream = null;
        this.remoteStream = null;
        this.peerConnection = null;
        this.socket = null;
        this.isCallActive = false;
        this.isVideoEnabled = true;
        this.isAudioEnabled = true;
        this.currentRoom = null;
        this.remotePeerId = null;
        this.isInitiator = false;
        this.userRole = 'hearing'; // Add user role property
        
        // Audio streaming for STT
        this.mediaRecorder = null;
        this.isAudioStreaming = false;
        this.audioChunks = [];
        
        // MediaPipe Holistic for ASL recognition
        this.holistic = null;
        this.localCanvas = null;
        this.localCanvasCtx = null;
        
        // TTS Audio Playback for hearing users
        this.audioContext = null;
        this.audioQueue = [];
        this.isPlayingAudio = false;
        
        // ASL Data Collection for training
        this.isRecording = false;
        this.currentRecording = [];
        this.collectedData = [];
        this.recordingFrameCount = 0;
        
        // WebRTC Configuration with STUN servers
        this.rtcConfiguration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };
        
        this.init();
    }

    init() {
        // Get DOM elements
        this.localVideo = document.getElementById('localVideo');
        this.remoteVideo = document.getElementById('remoteVideo');
        this.startCallBtn = document.getElementById('startCallBtn');
        this.endCallBtn = document.getElementById('endCallBtn');
        this.toggleVideoBtn = document.getElementById('toggleVideoBtn');
        this.toggleAudioBtn = document.getElementById('toggleAudioBtn');
        this.statusMessage = document.getElementById('statusMessage');
        this.speechToAslStatus = document.getElementById('speechToAslStatus');
        this.aslToSpeechStatus = document.getElementById('aslToSpeechStatus');
        this.roomIdInput = document.getElementById('roomIdInput');
        this.joinRoomBtn = document.getElementById('joinRoomBtn');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.connectionText = document.getElementById('connectionText');
        this.remoteVideoPlaceholder = document.getElementById('remoteVideoPlaceholder');
        
        // Role selection elements
        this.roleHearingRadio = document.getElementById('roleHearing');
        this.roleDeafRadio = document.getElementById('roleDeaf');
        
        // MediaPipe canvas elements
        this.localCanvas = document.getElementById('localCanvas');
        this.localCanvasCtx = this.localCanvas.getContext('2d');
        
        // Settings elements
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsModal = document.getElementById('settingsModal');
        this.closeSettingsBtn = document.getElementById('closeSettingsBtn');
        this.saveSettingsBtn = document.getElementById('saveSettingsBtn');
        this.themeToggle = document.getElementById('themeToggle');
        this.usernameInput = document.getElementById('usernameInput');
        this.videoQualitySelect = document.getElementById('videoQuality');
        this.defaultMuteToggle = document.getElementById('defaultMute');
        this.defaultVideoOffToggle = document.getElementById('defaultVideoOff');

        // ASL Data Collection elements
        this.aslDataCollection = document.getElementById('aslDataCollection');
        this.gestureSelect = document.getElementById('gestureSelect');
        this.startRecordingBtn = document.getElementById('startRecordingBtn');
        this.stopRecordingBtn = document.getElementById('stopRecordingBtn');
        this.downloadDataBtn = document.getElementById('downloadDataBtn');
        this.recordingStatus = document.getElementById('recordingStatus');
        this.frameCount = document.getElementById('frameCount');
        this.datasetCount = document.getElementById('datasetCount');

        this.initializeSocket();
        this.bindEventListeners();
        this.checkBrowserCompatibility();
        this.initializeUI();
    }

    bindEventListeners() {
        this.startCallBtn.addEventListener('click', () => this.startCall());
        this.endCallBtn.addEventListener('click', () => this.endCall());
        this.toggleVideoBtn.addEventListener('click', () => this.toggleVideo());
        this.toggleAudioBtn.addEventListener('click', () => this.toggleAudio());
        this.joinRoomBtn.addEventListener('click', () => this.joinRoom());
        this.roomIdInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.joinRoom();
        });
        
        // Role selection event listeners
        this.roleHearingRadio.addEventListener('change', () => {
            if (this.roleHearingRadio.checked) {
                this.userRole = 'hearing';
            }
        });
        this.roleDeafRadio.addEventListener('change', () => {
            if (this.roleDeafRadio.checked) {
                this.userRole = 'deaf';
            }
        });
        
        // Settings event listeners
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        this.saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) this.closeSettings();
        });
        this.themeToggle.addEventListener('change', () => this.toggleTheme());
        
        // ASL Data Collection event listeners
        this.startRecordingBtn.addEventListener('click', () => this.startRecording());
        this.stopRecordingBtn.addEventListener('click', () => this.stopRecording());
        this.downloadDataBtn.addEventListener('click', () => this.downloadCollectedData());
        
        // Add event listeners for toggle switch labels/sliders to make them clickable
        this.addToggleSwitchListeners();
    }

    initializeSocket() {
        try {
            // Connect to the backend signaling server
            this.socket = io('http://34.61.230.193:8000'); // Ensure this is your VM's public IP
            
            // Socket event handlers
            this.socket.on('connect', () => {
                this.connectionStatus.className = 'status-dot connected';
                this.connectionText.textContent = 'Connected';
                this.updateStatus('Connected to signaling server', 'success');
            });

            this.socket.on('disconnect', () => {
                this.connectionStatus.className = 'status-dot';
                this.connectionText.textContent = 'Disconnected';
                this.updateStatus('Disconnected from signaling server', 'error');
            });

            // WebRTC Signaling Events
            this.socket.on('user-joined', (data) => {
                this.remotePeerId = data.userId;
                this.isInitiator = true;
                this.updateStatus(`User joined. Preparing to initiate call...`, 'info');
                
                // Only create offer if we already have local media and peer connection
                if (this.localStream && this.peerConnection) {
                    setTimeout(() => this.createOffer(), 500);
                }
            });

            this.socket.on('user-ready', (data) => {
                this.remotePeerId = data.userId;
                this.isInitiator = false;
                this.updateStatus(`Connected to room. Waiting for call invitation...`, 'info');
            });

            this.socket.on('webrtc-offer', async (data) => {
                await this.handleOffer(data.offer, data.from);
            });

            this.socket.on('webrtc-answer', async (data) => {
                await this.handleAnswer(data.answer);
            });

            this.socket.on('webrtc-ice-candidate', async (data) => {
                await this.handleIceCandidate(data.candidate);
            });

            this.socket.on('user-left', (data) => {
                this.handleUserLeft(data.userId);
            });

            this.socket.on('room-full', () => {
                this.updateStatus('Room is full. Maximum 2 users per room.', 'error');
            });

            this.socket.on('error', (error) => {
                console.error('Socket.IO error:', error);
                this.updateStatus(`Connection error: ${error}`, 'error');
            });

            // STT Audio Streaming Event Handlers
            this.socket.on('audio-stream-started', (data) => {
                if (data.status === 'success') {
                    this.speechToAslStatus.textContent = 'Streaming audio for analysis...';
                    this.updateStatus('Audio streaming started for speech recognition', 'success');
                }
            });

            this.socket.on('audio-chunk-received', (data) => {
                // Audio chunk processed by backend - could update UI here if needed
            });

            this.socket.on('audio-stream-stopped', (data) => {
                if (data.status === 'success') {
                    this.speechToAslStatus.textContent = 'Speech processing stopped';
                    this.updateStatus('Audio streaming stopped', 'info');
                }
            });

            this.socket.on('audio-stream-error', (data) => {
                console.error('Audio streaming error:', data.error);
                this.updateStatus(`Audio streaming error: ${data.error}`, 'error');
                this.speechToAslStatus.textContent = 'Speech processing error';
                // Stop streaming on error
                this.stopAudioStreaming();
            });

            // Real-time subtitle display handler
            // The backend emits 'transcribed_text', not 'subtitle'
            this.socket.on('transcribed_text', (data) => {
                if (this.userRole === 'deaf') {
                    const subtitleDiv = document.getElementById('subtitleDisplay');
                    if (subtitleDiv && data && typeof data.text === 'string') {
                        subtitleDiv.textContent = data.text;
                        if (data.isFinal) {
                            subtitleDiv.classList.add('final');
                        } else {
                            subtitleDiv.classList.remove('final');
                        }
                    }
                }
            });

            // ASL Prediction Handler - Display recognized signs for hearing users
            this.socket.on('asl_prediction', (data) => {
                // Only display ASL predictions for hearing users
                if (this.userRole === 'hearing') {
                    // Check if it's a clear signal
                    if (data.sign === 'CLEAR') {
                        this.speechToAslStatus.textContent = 'Ready';
                        console.log('ASL prediction cleared');
                    } else {
                        // Display the recognized sign with confidence
                        const confidencePercentage = Math.round(data.confidence * 100);
                        this.speechToAslStatus.textContent = `ASL: ${data.sign} (${confidencePercentage}%)`;
                        console.log(`ASL Prediction received: ${data.sign} with ${confidencePercentage}% confidence`);
                    }
                }
            });

            // TTS Audio Playback Handler - Play synthesized speech for hearing users
            this.socket.on('synthesized_audio_chunk', async (audioData) => {
                // Only play audio for hearing users
                if (this.userRole === 'hearing') {
                    console.log(`TTS Audio received: ${audioData.length} bytes`);
                    await this.playTTSAudio(audioData);
                }
            });

        } catch (error) {
            console.error('Failed to initialize Socket.IO:', error);
            this.updateStatus('Failed to connect to signaling server', 'error');
        }
    }

    joinRoom() {
        const roomId = this.roomIdInput.value.trim();
        if (!roomId) {
            this.updateStatus('Please enter a room ID', 'error');
            return;
        }

        if (!this.socket || !this.socket.connected) {
            this.updateStatus('Not connected to signaling server', 'error');
            return;
        }

        // Get selected role
        const selectedRole = document.querySelector('input[name="userRole"]:checked');
        if (!selectedRole) {
            this.updateStatus('Please select your role (Hearing or Deaf)', 'error');
            return;
        }
        
        this.userRole = selectedRole.value;
        this.currentRoom = roomId;
        this.socket.emit('join_room', { roomId: roomId, userRole: this.userRole });
        this.updateStatus(`Joining room: ${roomId} as ${this.userRole}...`, 'info', true);
        
        // Show/hide ASL data collection panel based on role
        if (this.userRole === 'deaf') {
            this.aslDataCollection.style.display = 'block';
        } else {
            this.aslDataCollection.style.display = 'none';
        }
        
        // Hide room selection modal
        document.getElementById('roomSelection').style.display = 'none';
    }

    checkBrowserCompatibility() {
        const checks = {
            getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
            rtcPeerConnection: !!(window.RTCPeerConnection || window.webkitRTCPeerConnection || window.mozRTCPeerConnection),
            rtcSessionDescription: !!(window.RTCSessionDescription || window.webkitRTCSessionDescription || window.mozRTCSessionDescription),
            rtcIceCandidate: !!(window.RTCIceCandidate || window.webkitRTCIceCandidate || window.mozRTCIceCandidate)
        };

        const unsupported = Object.keys(checks).filter(key => !checks[key]);
        
        if (unsupported.length > 0) {
            this.updateStatus(`Browser missing WebRTC support: ${unsupported.join(', ')}. Please use a modern browser.`, 'error');
            console.error('WebRTC compatibility check failed:', checks);
            return false;
        }
        
        return true;
    }

    async startCall() {
        try {
            // Check if browser supports getUserMedia
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('getUserMedia is not supported by this browser. Please use Chrome, Firefox, Safari, or Edge.');
            }

            this.updateStatus('Starting camera and microphone...', 'info', true);
            
            // Enhanced media constraints for better quality
            let mediaConstraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                },
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            };

            // Request access to camera and microphone
            try {
                this.localStream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
            } catch (constraintError) {
                // If enhanced constraints fail, try basic constraints
                console.warn('Enhanced constraints failed, trying basic:', constraintError);
                mediaConstraints = { video: true, audio: true };
                this.localStream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
            }

            // Display local video stream
            this.localVideo.srcObject = this.localStream;
            this.updateStatus('Local video stream connected successfully', 'success');
            
                            // Set canvas dimensions to match video when metadata is loaded
                this.localVideo.addEventListener('loadedmetadata', () => {
                    this.localCanvas.width = this.localVideo.videoWidth;
                    this.localCanvas.height = this.localVideo.videoHeight;
                });

            // Initialize WebRTC peer connection
            await this.initializePeerConnection();

            // Update UI state
            this.isCallActive = true;
            this.startCallBtn.disabled = true;
            this.endCallBtn.disabled = false;
            this.updateStatus('WebRTC connection ready! Waiting for remote participant...', 'success');

            // If we have a peer connection but no tracks added, add them now
            if (this.peerConnection && this.localStream) {
                this.addLocalTracksToConnection();
            }

            // If we're the initiator and have a remote peer, create an offer
            if (this.isInitiator && this.remotePeerId) {
                setTimeout(() => {
                    this.createOffer();
                }, 1000);
            }

            // Start ASL recognition and speech processing based on user role
            if (this.userRole === 'deaf') {
                this.startAslRecognition();
            } else {
                this.aslToSpeechStatus.textContent = 'ASL recognition skipped: User role is Hearing.';
                this.updateStatus('ASL recognition skipped: User role is Hearing.', 'info');
            }
            
                            if (this.userRole === 'hearing') {
                    this.startSpeechProcessing();
                } else {
                    this.speechToAslStatus.textContent = 'Speech processing skipped: User role is Deaf.';
                    this.updateStatus('Speech processing skipped: User role is Deaf.', 'info');
                }

        } catch (error) {
            console.error('Error starting call:', error);
            
            // Provide specific error messages based on error type
            if (error.name === 'NotAllowedError') {
                this.updateStatus('Camera/microphone access denied. Please allow permissions and try again.', 'error');
            } else if (error.name === 'NotFoundError') {
                this.updateStatus('No camera or microphone found. Please check your devices.', 'error');
            } else if (error.name === 'NotReadableError') {
                this.updateStatus('Camera/microphone is already in use by another application.', 'error');
            } else {
                this.updateStatus(`Failed to start call: ${error.message}`, 'error');
            }
        }
    }

    endCall() {
        try {
            // Stop audio streaming for STT
            this.stopAudioStreaming();
            
            // Stop ASL recognition
            this.stopAslRecognition();
            
            // Stop all media streams
            if (this.localStream) {
                this.localStream.getTracks().forEach(track => track.stop());
                this.localStream = null;
            }

            // Clear video elements
            this.localVideo.srcObject = null;
            this.remoteVideo.srcObject = null;

            // Close peer connection
            if (this.peerConnection) {
                this.peerConnection.close();
                this.peerConnection = null;
            }

            // Update UI state
            this.isCallActive = false;
            this.startCallBtn.disabled = false;
            this.endCallBtn.disabled = true;
            this.updateStatus('Call ended.', 'info');

            // Reset communication status
            this.speechToAslStatus.textContent = 'Ready';
            this.aslToSpeechStatus.textContent = 'Ready';

        } catch (error) {
            console.error('Error ending call:', error);
        }
    }

    toggleVideo() {
        if (this.localStream) {
            const videoTrack = this.localStream.getVideoTracks()[0];
            if (videoTrack) {
                this.isVideoEnabled = !this.isVideoEnabled;
                videoTrack.enabled = this.isVideoEnabled;
                
                // Update button visual state
                if (this.isVideoEnabled) {
                    this.toggleVideoBtn.classList.add('active');
                    this.toggleVideoBtn.title = 'Turn off video';
                } else {
                    this.toggleVideoBtn.classList.remove('active');
                    this.toggleVideoBtn.title = 'Turn on video';
                }
                
                this.updateStatus(this.isVideoEnabled ? 'Video enabled' : 'Video disabled', 'info');
            }
            // If video is disabled, stop ASL recognition as there's no visual input
            if (!this.isVideoEnabled) {
                this.stopAslRecognition();
            } else {
                this.startAslRecognition();
            }
        }
    }

    toggleAudio() {
        if (this.localStream) {
            const audioTrack = this.localStream.getAudioTracks()[0];
            if (audioTrack) {
                this.isAudioEnabled = !this.isAudioEnabled;
                audioTrack.enabled = this.isAudioEnabled;
                
                // Update button visual state
                if (this.isAudioEnabled) {
                    this.toggleAudioBtn.classList.add('active');
                    this.toggleAudioBtn.title = 'Mute audio';
                    
                    // Resume audio streaming if call is active and user is hearing
                    if (this.isCallActive && !this.isAudioStreaming && this.userRole === 'hearing') {
                        this.startSpeechProcessing();
                    }
                } else {
                    this.toggleAudioBtn.classList.remove('active');
                    this.toggleAudioBtn.title = 'Unmute audio';
                    
                    // Stop audio streaming when muted
                    if (this.isAudioStreaming) {
                        this.stopAudioStreaming();
                    }
                }
                
                this.updateStatus(this.isAudioEnabled ? 'Audio enabled' : 'Audio disabled', 'info');
            }
        }
    }

    async initializePeerConnection() {
        try {
            // Create RTCPeerConnection with STUN server configuration
            this.peerConnection = new RTCPeerConnection(this.rtcConfiguration);
            
            // Add local stream tracks to peer connection
            if (this.localStream) {
                this.localStream.getTracks().forEach((track) => {
                    this.peerConnection.addTrack(track, this.localStream);
                });
            }

            // Set up WebRTC event handlers
            this.setupPeerConnectionEventHandlers();
            
            this.updateStatus('WebRTC peer connection established', 'info');
            
        } catch (error) { // Added catch block
            console.error('Error initializing peer connection:', error);
            this.updateStatus(`Failed to initialize peer connection: ${error.message}`, 'error');
            // Optionally, you might want to end the call or disable call buttons here
            this.isCallActive = false;
            this.startCallBtn.disabled = false;
            this.endCallBtn.disabled = true;
        }
    }

    setupPeerConnectionEventHandlers() {
        this.peerConnection.ontrack = (event) => {
            console.log('Remote track received:', event.streams);
            if (this.remoteVideo.srcObject !== event.streams[0]) {
                this.remoteVideo.srcObject = event.streams[0];
                this.remoteStream = event.streams[0];
                this.updateStatus('Remote stream connected.', 'success');
                this.remoteVideoPlaceholder.style.display = 'none'; // Hide placeholder
                this.remoteVideo.style.display = 'block'; // Show remote video
            }
        };

        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                console.log('Sending ICE candidate:', event.candidate);
                this.socket.emit('webrtc-ice-candidate', {
                    to: this.remotePeerId,
                    candidate: event.candidate
                });
            }
        };

        this.peerConnection.oniceconnectionstatechange = () => {
            console.log('ICE connection state:', this.peerConnection.iceConnectionState);
            this.updateStatus(`ICE connection state: ${this.peerConnection.iceConnectionState}`, 'info');
            if (this.peerConnection.iceConnectionState === 'disconnected' || this.peerConnection.iceConnectionState === 'failed') {
                this.updateStatus('WebRTC connection lost. Attempting to reconnect...', 'error');
                // Implement reconnection logic or prompt user to restart call
            } else if (this.peerConnection.iceConnectionState === 'connected') {
                this.updateStatus('WebRTC connection established.', 'success');
            }
        };

        this.peerConnection.onconnectionstatechange = () => {
            console.log('Peer connection state:', this.peerConnection.connectionState);
            this.updateStatus(`Peer connection state: ${this.peerConnection.connectionState}`, 'info');
            if (this.peerConnection.connectionState === 'disconnected' || this.peerConnection.connectionState === 'failed') {
                this.updateStatus('Peer connection lost. Call may have ended.', 'error');
            } else if (this.peerConnection.connectionState === 'connected') {
                this.updateStatus('Peer connection established. Call active.', 'success');
            }
        };
    }

    addLocalTracksToConnection() {
        if (this.localStream && this.peerConnection) {
            this.localStream.getTracks().forEach(track => {
                // Check if track is already added to avoid errors
                const existingSenders = this.peerConnection.getSenders();
                const trackAlreadyAdded = existingSenders.some(sender => sender.track === track);
                if (!trackAlreadyAdded) {
                    this.peerConnection.addTrack(track, this.localStream);
                    console.log('Added local track to peer connection:', track.kind);
                } else {
                    console.log('Track already added:', track.kind);
                }
            });
        }
    }

    async createOffer() {
        try {
            if (!this.peerConnection) {
                this.updateStatus('Peer connection not initialized. Cannot create offer.', 'error');
                return;
            }
            if (!this.remotePeerId) {
                this.updateStatus('No remote peer ID to send offer to.', 'warning');
                return;
            }

            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);
            
            console.log('Sending offer:', offer);
            this.socket.emit('webrtc-offer', {
                to: this.remotePeerId,
                offer: offer
            });
            this.updateStatus('Offer sent to remote peer.', 'info');
        } catch (error) {
            console.error('Error creating offer:', error);
            this.updateStatus(`Failed to create offer: ${error.message}`, 'error');
        }
    }

    async handleOffer(offer, fromSid) {
        try {
            this.remotePeerId = fromSid; // Set remote peer ID from the offer sender
            if (!this.peerConnection) {
                // If peer connection not yet initialized (e.g., if this client is not initiator)
                // Ensure local stream is available before initializing peer connection
                if (!this.localStream) {
                    await this.startCall(); // This will initialize local stream and peer connection
                } else {
                    await this.initializePeerConnection();
                }
            }

            await this.peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
            console.log('Received offer, creating answer...');
            this.updateStatus('Received offer, creating answer...', 'info');

            const answer = await this.peerConnection.createAnswer();
            await this.peerConnection.setLocalDescription(answer);
            
            console.log('Sending answer:', answer);
            this.socket.emit('webrtc-answer', {
                to: this.remotePeerId,
                answer: answer
            });
            this.updateStatus('Answer sent to remote peer.', 'success');
        } catch (error) {
            console.error('Error handling offer:', error);
            this.updateStatus(`Failed to handle offer: ${error.message}`, 'error');
        }
    }

    async handleAnswer(answer) {
        try {
            await this.peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
            console.log('Received answer.');
            this.updateStatus('Received answer. WebRTC connection established.', 'success');
        } catch (error) {
            console.error('Error handling answer:', error);
            this.updateStatus(`Failed to handle answer: ${error.message}`, 'error');
        }
    }

    async handleIceCandidate(candidate) {
        try {
            if (!this.peerConnection || !candidate) {
                console.warn('Peer connection not ready or candidate is null.', candidate);
                return;
            }
            await this.peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
            console.log('Added ICE candidate.');
        } catch (error) {
            console.error('Error adding ICE candidate:', error);
            // Ignore "Failed to set remote answer sdp: Called in wrong state" errors
            // These can happen if candidate arrives before remote description is set
            if (!error.message.includes('wrong state')) {
                this.updateStatus(`Failed to add ICE candidate: ${error.message}`, 'error');
            }
        }
    }

    handleUserLeft(userId) {
        this.updateStatus(`User ${userId} left the room.`, 'info');
        if (this.remotePeerId === userId) {
            this.remotePeerId = null;
            this.remoteVideo.srcObject = null;
            this.remoteVideoPlaceholder.style.display = 'block'; // Show placeholder
            this.remoteVideo.style.display = 'none'; // Hide remote video
            this.updateStatus('Remote user disconnected. Call ended.', 'info');
            this.endCall(); // Automatically end call if remote user leaves
        }
    }

    updateStatus(message, type = 'info', clear = false) {
        if (clear) {
            this.statusMessage.innerHTML = ''; // Clear previous messages
        }
        const p = document.createElement('p');
        p.className = `status-message ${type}`;
        p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        this.statusMessage.prepend(p); // Add new messages to the top
        // Keep only the last 5 messages
        while (this.statusMessage.children.length > 5) {
            this.statusMessage.removeChild(this.statusMessage.lastChild);
        }
    }

    // --- Speech-to-Text (STT) Integration ---
    startSpeechProcessing() {
        if (!this.localStream) {
            this.updateStatus('No local stream available for speech processing.', 'error');
            return;
        }

        if (this.isAudioStreaming) {
            this.updateStatus('Audio streaming already active.', 'info');
            return;
        }

        try {
            const audioTrack = this.localStream.getAudioTracks()[0];
            if (!audioTrack) {
                this.updateStatus('No audio track found in local stream.', 'error');
                return;
            }

            if (audioTrack.readyState !== 'live' || !audioTrack.enabled) {
                this.updateStatus('Audio track is not live or enabled. Cannot start speech processing.', 'error');
                return;
            }

            // Create MediaRecorder from the audio track
            // Use 'audio/webm; codecs=opus' for good compatibility and compression
            this.mediaRecorder = new MediaRecorder(new MediaStream([audioTrack]), {
                mimeType: 'audio/webm; codecs=opus'
            });

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.socket.emit('send_audio_chunk', event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.isAudioStreaming = false;
                this.speechToAslStatus.textContent = 'Speech processing stopped.';
                this.audioChunks = [];
                this.socket.emit('end_audio_stream');
            };

            this.mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
                this.updateStatus(`MediaRecorder error: ${event.error.name}`, 'error');
                this.isAudioStreaming = false;
                this.speechToAslStatus.textContent = 'Speech processing error.';
                // Stop streaming on error
                this.stopAudioStreaming();
            };

            // Inform backend that audio stream is starting
            this.socket.emit('start_audio_stream');
            
            // Start recording and send data in chunks every 500ms
            this.mediaRecorder.start(500); // Changed from 100 to 500ms
            this.isAudioStreaming = true;
            this.speechToAslStatus.textContent = 'Processing speech...';
            this.updateStatus('Audio streaming started for speech recognition', 'success');

        } catch (error) {
            console.error('Error starting speech processing:', error);
            this.updateStatus(`Failed to start speech processing: ${error.message}`, 'error');
            this.isAudioStreaming = false;
        }
    }

    stopAudioStreaming() {
        if (this.isAudioStreaming && this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            console.log('Stopping MediaRecorder...');
            this.mediaRecorder.stop();
            this.isAudioStreaming = false;
            // The onstop event handler will emit 'end_audio_stream' to the backend
        } else {
            console.log('MediaRecorder not active or already stopped.');
        }
    }

    // --- ASL Recognition Integration with MediaPipe Holistic ---
    startAslRecognition() {
                try {
            console.log('Starting ASL recognition...');
            
            // Initialize MediaPipe Holistic
            this.holistic = new window.Holistic({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
                }
            });

            // Set up results callback
            this.holistic.onResults = (results) => {
                // Clear the canvas
                this.localCanvasCtx.clearRect(0, 0, this.localCanvas.width, this.localCanvas.height);
                
                // Draw landmarks using MediaPipe's drawing utilities or fallback
                if (results.poseLandmarks) {
                    this.drawLandmarks(this.localCanvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2, radius: 3});
                    console.log('Pose landmarks detected:', results.poseLandmarks.length, 'points');
                }
                
                if (results.leftHandLandmarks) {
                    this.drawLandmarks(this.localCanvasCtx, results.leftHandLandmarks, {color: '#00FF00', lineWidth: 2, radius: 3});
                    console.log('Left hand landmarks detected:', results.leftHandLandmarks.length, 'points');
                }
                
                if (results.rightHandLandmarks) {
                    this.drawLandmarks(this.localCanvasCtx, results.rightHandLandmarks, {color: '#0000FF', lineWidth: 2, radius: 3});
                    console.log('Right hand landmarks detected:', results.rightHandLandmarks.length, 'points');
                }

                // Add console log to verify keypoints are being captured
                console.log('MediaPipe Results:', {
                    pose: results.poseLandmarks ? results.poseLandmarks.length : 0,
                    leftHand: results.leftHandLandmarks ? results.leftHandLandmarks.length : 0,
                    rightHand: results.rightHandLandmarks ? results.rightHandLandmarks.length : 0
                });

                // Prepare and stream keypoint data to backend
                const keypointData = this.prepareKeypointData(results);
                if (this.socket && this.socket.connected && keypointData) {
                    this.socket.emit('asl_keypoints', keypointData);
                    console.log('Sending ASL keypoints:', keypointData);
                }

                // Store keypoint data if recording for training
                if (this.isRecording && keypointData) {
                    this.currentRecording.push(keypointData);
                    this.recordingFrameCount++;
                    this.frameCount.textContent = `Frames: ${this.recordingFrameCount}`;
                }
            };

            // Enhanced MediaPipe options for improved ASL recognition accuracy
            this.holistic.setOptions({
                modelComplexity: 1,                    // Balance between speed and accuracy
                smoothLandmarks: true,                 // Reduce jitter in hand movements
                enableSegmentation: false,             // Not needed for ASL recognition
                smoothSegmentation: false,
                refineFaceLandmarks: false,            // Focus on hands and pose
                minDetectionConfidence: 0.7,           // Higher threshold for more confident detections
                minTrackingConfidence: 0.5,            // Allow tracking to continue with lower confidence
                staticImageMode: false,                // Enable video mode for better temporal consistency
                maxNumHands: 2,                        // Ensure both hands can be detected
                minHandDetectionConfidence: 0.7,       // Higher hand detection confidence
                minHandPresenceConfidence: 0.5         // Lower presence confidence to maintain tracking
            });

            // Start processing video frames after MediaPipe initialization
            setTimeout(() => {
                this.processVideoFrames();
            }, 1000);
            
            this.aslToSpeechStatus.textContent = 'Processing ASL...';
            this.updateStatus('ASL recognition started', 'success');
            
        } catch (error) {
            console.error('Error starting ASL recognition:', error);
            this.updateStatus(`Failed to start ASL recognition: ${error.message}`, 'error');
            this.aslToSpeechStatus.textContent = 'ASL recognition error';
        }
    }

    stopAslRecognition() {
        try {
            if (this.holistic) {
                this.holistic.close();
                this.holistic = null;
            }
            
            // Clear the canvas
            if (this.localCanvasCtx) {
                this.localCanvasCtx.clearRect(0, 0, this.localCanvas.width, this.localCanvas.height);
            }
``            
            this.aslToSpeechStatus.textContent = 'ASL recognition stopped';
            this.updateStatus('ASL recognition stopped', 'info');
            
        } catch (error) {
            console.error('Error stopping ASL recognition:', error);
            this.updateStatus(`Error stopping ASL recognition: ${error.message}`, 'error');
        }
    }

            // Enhanced video frame processing with quality control and frame rate optimization
        processVideoFrames() {
            let frameCount = 0;
            let lastFrameTime = 0;
            const targetFPS = 15; // Optimized for ASL recognition (balance between accuracy and performance)
            const frameInterval = 1000 / targetFPS;
            
            const sendFrame = async () => {
                const currentTime = performance.now();
                
                if (this.holistic && this.localVideo.readyState >= 2) {
                    try {
                        // Frame rate control for optimal ASL processing
                        if (currentTime - lastFrameTime >= frameInterval) {
                            // Check video quality before processing
                            const videoQuality = this.assessVideoQuality();
                            
                            if (videoQuality.isGoodForASL) {
                                // Send video frame to MediaPipe Holistic for processing
                                await this.holistic.send({image: this.localVideo});
                                lastFrameTime = currentTime;
                                frameCount++;
                                
                                // Log processing stats every 5 seconds
                                if (frameCount % (targetFPS * 5) === 0) {
                                    console.log(`ASL Processing: ${frameCount} frames processed. Quality: ${videoQuality.score}/10`);
                                }
                            }
                        }
                    } catch (error) {
                        console.error('Error sending frame to MediaPipe:', error);
                    }
                }
                
                // Continue processing frames using requestAnimationFrame (manual camera implementation)
                if (this.holistic) {
                    requestAnimationFrame(sendFrame);
                }
            };
            
            // Start the enhanced camera-like frame processing
            console.log('Starting enhanced MediaPipe camera processing with quality control...');
            sendFrame();
        }

        // Assess video quality for ASL recognition suitability
        assessVideoQuality() {
            try {
                const video = this.localVideo;
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                // Sample a small portion of the video for quality assessment
                canvas.width = 160;
                canvas.height = 120;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;
                
                // Calculate brightness and contrast metrics
                let brightness = 0;
                let totalVariance = 0;
                
                for (let i = 0; i < data.length; i += 4) {
                    const luminance = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                    brightness += luminance;
                }
                
                brightness /= (data.length / 4);
                
                // Calculate contrast (standard deviation of luminance)
                for (let i = 0; i < data.length; i += 4) {
                    const luminance = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                    totalVariance += Math.pow(luminance - brightness, 2);
                }
                
                const contrast = Math.sqrt(totalVariance / (data.length / 4));
                
                // Score the quality (0-10 scale)
                let score = 5; // Start with baseline
                
                // Brightness scoring (optimal range: 100-180)
                if (brightness >= 100 && brightness <= 180) {
                    score += 2;
                } else if (brightness >= 80 && brightness <= 200) {
                    score += 1;
                } else if (brightness < 60 || brightness > 220) {
                    score -= 2;
                }
                
                // Contrast scoring (higher is generally better for hand detection)
                if (contrast > 40) {
                    score += 2;
                } else if (contrast > 25) {
                    score += 1;
                } else if (contrast < 15) {
                    score -= 1;
                }
                
                // Video resolution factor
                if (video.videoWidth >= 640 && video.videoHeight >= 480) {
                    score += 1;
                }
                
                return {
                    brightness: Math.round(brightness),
                    contrast: Math.round(contrast),
                    score: Math.max(0, Math.min(10, score)),
                    isGoodForASL: score >= 5,
                    recommendations: this.getQualityRecommendations(brightness, contrast, score)
                };
                
            } catch (error) {
                console.error('Error assessing video quality:', error);
                return {
                    brightness: 0,
                    contrast: 0,
                    score: 5,
                    isGoodForASL: true, // Default to true to avoid blocking
                    recommendations: []
                };
            }
        }

        // Provide quality improvement recommendations
        getQualityRecommendations(brightness, contrast, score) {
            const recommendations = [];
            
            if (brightness < 80) {
                recommendations.push('Increase lighting - add more light sources');
            } else if (brightness > 200) {
                recommendations.push('Reduce lighting - too bright for optimal detection');
            }
            
            if (contrast < 20) {
                recommendations.push('Improve contrast - try a different background');
            }
            
            if (score < 5) {
                recommendations.push('Overall video quality is poor for ASL recognition');
            }
            
            return recommendations;
        }

    // Draw landmarks on canvas (backup method if MediaPipe drawing utils not available)
    drawLandmarks(ctx, landmarks, style = {}) {
        const {color = '#FF0000', lineWidth = 2, radius = 3} = style;
        
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        
        // Draw landmark points
        landmarks.forEach((landmark) => {
            const x = landmark.x * this.localCanvas.width;
            const y = landmark.y * this.localCanvas.height;
            
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

            // Enhanced keypoint data preparation with preprocessing for better ASL recognition
        prepareKeypointData(results) {
            try {
                const keypointData = {
                    timestamp: Date.now(),
                    pose: null,
                    leftHand: null,
                    rightHand: null,
                    // Additional metadata for improved processing
                    quality: {
                        poseVisibility: 0,
                        handDetectionConfidence: 0,
                        frameQuality: 'good'
                    }
                };

                // Process pose landmarks with quality assessment
                if (results.poseLandmarks && results.poseLandmarks.length > 0) {
                    const processedPose = results.poseLandmarks.map((landmark, index) => ({
                        id: index,
                        x: parseFloat(landmark.x.toFixed(4)),
                        y: parseFloat(landmark.y.toFixed(4)),
                        z: parseFloat(landmark.z.toFixed(4)),
                        visibility: parseFloat((landmark.visibility || 0).toFixed(4))
                    }));
                    
                    // Calculate average pose visibility for quality assessment
                    const avgVisibility = processedPose.reduce((sum, lm) => sum + lm.visibility, 0) / processedPose.length;
                    keypointData.quality.poseVisibility = parseFloat(avgVisibility.toFixed(3));
                    
                    // Only include pose landmarks with decent visibility
                    keypointData.pose = processedPose.filter(lm => lm.visibility > 0.5);
                }

                // Process left hand landmarks with normalization
                if (results.leftHandLandmarks && results.leftHandLandmarks.length > 0) {
                    keypointData.leftHand = this.normalizeHandLandmarks(results.leftHandLandmarks, 'left');
                    keypointData.quality.handDetectionConfidence += 0.5;
                }

                // Process right hand landmarks with normalization
                if (results.rightHandLandmarks && results.rightHandLandmarks.length > 0) {
                    keypointData.rightHand = this.normalizeHandLandmarks(results.rightHandLandmarks, 'right');
                    keypointData.quality.handDetectionConfidence += 0.5;
                }

                // Assess frame quality based on available data
                const hasGoodPose = keypointData.pose && keypointData.quality.poseVisibility > 0.7;
                const hasHands = keypointData.leftHand || keypointData.rightHand;
                
                if (!hasGoodPose && !hasHands) {
                    keypointData.quality.frameQuality = 'poor';
                } else if (hasGoodPose && hasHands) {
                    keypointData.quality.frameQuality = 'excellent';
                } else {
                    keypointData.quality.frameQuality = 'good';
                }

                // Only return data if we have meaningful landmarks
                if (keypointData.pose?.length > 0 || keypointData.leftHand || keypointData.rightHand) {
                    return keypointData;
                }

                return null;
            } catch (error) {
                console.error('Error preparing keypoint data:', error);
                return null;
            }
        }

        // Normalize hand landmarks relative to wrist for scale and position invariance
        normalizeHandLandmarks(handLandmarks, handType) {
            if (!handLandmarks || handLandmarks.length === 0) return null;
            
            try {
                // Wrist is landmark 0 in MediaPipe hand model
                const wrist = handLandmarks[0];
                
                return handLandmarks.map((landmark, index) => ({
                    id: index,
                    // Normalize relative to wrist position
                    x: parseFloat((landmark.x - wrist.x).toFixed(4)),
                    y: parseFloat((landmark.y - wrist.y).toFixed(4)),
                    z: parseFloat((landmark.z - wrist.z).toFixed(4)),
                    // Preserve absolute wrist position for context
                    abs_x: index === 0 ? parseFloat(landmark.x.toFixed(4)) : undefined,
                    abs_y: index === 0 ? parseFloat(landmark.y.toFixed(4)) : undefined,
                    handType: handType
                }));
            } catch (error) {
                console.error(`Error normalizing ${handType} hand landmarks:`, error);
                return null;
            }
        }

    // --- TTS Audio Playback Methods ---
    async initializeAudioContext() {
        if (!this.audioContext) {
            try {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                console.log('Audio context initialized for TTS playback');
            } catch (error) {
                console.error('Failed to initialize audio context:', error);
            }
        }
        
        // Resume audio context if suspended (required by browser policies)
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    async playTTSAudio(audioData) {
        try {
            // Initialize audio context if needed
            await this.initializeAudioContext();
            
            if (!this.audioContext) {
                console.error('Audio context not available for TTS playback');
                return;
            }

            // Convert the binary audio data to ArrayBuffer
            let arrayBuffer;
            if (audioData instanceof ArrayBuffer) {
                arrayBuffer = audioData;
            } else if (audioData instanceof Uint8Array) {
                arrayBuffer = audioData.buffer;
            } else {
                // Handle base64 or other formats
                const uint8Array = new Uint8Array(audioData);
                arrayBuffer = uint8Array.buffer;
            }

            // Decode the MP3 audio data
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            // Add to queue and play
            this.audioQueue.push(audioBuffer);
            if (!this.isPlayingAudio) {
                this.playNextAudio();
            }
            
        } catch (error) {
            console.error('Error playing TTS audio:', error);
            
            // Fallback: Use HTML5 audio element for MP3 playback
            this.playTTSAudioFallback(audioData);
        }
    }

    playNextAudio() {
        if (this.audioQueue.length === 0) {
            this.isPlayingAudio = false;
            return;
        }

        this.isPlayingAudio = true;
        const audioBuffer = this.audioQueue.shift();
        
        // Create audio source and play
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        
        // Play next audio when current finishes
        source.onended = () => {
            this.playNextAudio();
        };
        
        source.start();
        console.log('Playing TTS audio');
    }

    playTTSAudioFallback(audioData) {
        try {
            // Create a blob from the audio data
            const blob = new Blob([audioData], { type: 'audio/mpeg' });
            const audioUrl = URL.createObjectURL(blob);
            
            // Create and play audio element
            const audio = new Audio(audioUrl);
            audio.play().then(() => {
                console.log('TTS audio playing via fallback method');
            }).catch(error => {
                console.error('Fallback audio playback failed:', error);
            });
            
            // Clean up URL after playback
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
            };
            
        } catch (error) {
            console.error('TTS audio fallback playback failed:', error);
        }
    }

    // --- UI Initialization and Settings ---
    initializeUI() {
        // Apply default settings or load from local storage
        this.loadSettings();
        // Set initial button states based on enabled flags
        this.toggleVideoBtn.classList.toggle('active', this.isVideoEnabled);
        this.toggleAudioBtn.classList.toggle('active', this.isAudioEnabled);
        this.toggleVideoBtn.title = this.isVideoEnabled ? 'Turn off video' : 'Turn on video';
        this.toggleAudioBtn.title = this.isAudioEnabled ? 'Mute audio' : 'Unmute audio';

        // Set default room ID
        const defaultRoomId = 'interlude-room'; // You can make this dynamic or user-configurable
        this.roomIdInput.value = defaultRoomId;
        
        // Set default role selection
        this.roleHearingRadio.checked = true;
        this.userRole = 'hearing';
    }

    addToggleSwitchListeners() {
        // Make the entire toggle switch area clickable for better UX
        const themeToggleContainer = document.querySelector('.toggle-switch-container label[for="themeToggle"]');
        if (themeToggleContainer) {
            themeToggleContainer.addEventListener('click', (e) => {
                // Prevent default label click if the input itself is clicked, to avoid double toggling
                if (e.target !== this.themeToggle) {
                    this.themeToggle.checked = !this.themeToggle.checked;
                    this.toggleTheme();
                }
            });
        }
    }

    openSettings() {
        this.loadSettings(); // Reload settings to ensure latest are displayed
        this.settingsModal.style.display = 'flex'; // Use flex to center
    }

    closeSettings() {
        this.settingsModal.style.display = 'none';
    }

    saveSettings() {
        const settings = {
            theme: this.themeToggle.checked ? 'dark' : 'light',
            username: this.usernameInput.value.trim(),
            videoQuality: this.videoQualitySelect.value,
            defaultMute: this.defaultMuteToggle.checked,
            defaultVideoOff: this.defaultVideoOffToggle.checked,
        };
        localStorage.setItem('interludeSettings', JSON.stringify(settings));
        this.applySettings(settings);
        this.closeSettings();
        this.updateStatus('Settings saved and applied.', 'success');
    }

    loadSettings() {
        const savedSettings = localStorage.getItem('interludeSettings');
        let settings = {};
        if (savedSettings) {
            try {
                settings = JSON.parse(savedSettings);
            } catch (e) {
                console.error('Error parsing saved settings:', e);
                localStorage.removeItem('interludeSettings'); // Clear corrupt settings
            }
        }

        // Apply settings immediately
        this.applySettings(settings);

        // Update form values
        this.themeToggle.checked = settings.theme === 'dark';
        this.usernameInput.value = settings.username || 'You';
        this.videoQualitySelect.value = settings.videoQuality || '720p';
        this.defaultMuteToggle.checked = settings.defaultMute || false;
        this.defaultVideoOffToggle.checked = settings.defaultVideoOff || false;
    }

    applySettings(settings) {
        // Apply theme
        if (settings.theme === 'dark') {
            document.body.classList.add('dark-theme');
        } else {
            document.body.classList.remove('dark-theme');
        }

        // Update video label with username
        const videoLabel = document.querySelector('.local-video-card .video-label');
        if (videoLabel) {
            videoLabel.textContent = settings.username || 'You';
        }

        // Store settings for use in call initialization
        this.userSettings = settings;
    }

    toggleTheme() {
        const isDark = this.themeToggle.checked;
        if (isDark) {
            document.body.classList.add('dark-theme');
        } else {
            document.body.classList.remove('dark-theme');
        }
    }

    // --- ASL Data Collection Methods ---
    startRecording() {
        if (this.isRecording) {
            this.updateStatus('Already recording', 'warning');
            return;
        }

        const selectedGesture = this.gestureSelect.value;
        if (!selectedGesture) {
            this.updateStatus('Please select a gesture to record', 'error');
            return;
        }

        // Reset recording state
        this.currentRecording = [];
        this.recordingFrameCount = 0;
        this.isRecording = true;

        // Update UI
        this.startRecordingBtn.disabled = true;
        this.stopRecordingBtn.disabled = false;
        this.gestureSelect.disabled = true;
        this.recordingStatus.textContent = `Recording: ${selectedGesture}`;
        this.frameCount.textContent = 'Frames: 0';

        this.updateStatus(`Started recording gesture: ${selectedGesture}`, 'success');
        
        // Automatically stop recording after 10 seconds as safety measure
        this.recordingTimeout = setTimeout(() => {
            if (this.isRecording) {
                this.stopRecording();
                this.updateStatus('Recording stopped automatically after 10 seconds', 'info');
            }
        }, 10000);
    }

    stopRecording() {
        if (!this.isRecording) {
            this.updateStatus('Not currently recording', 'warning');
            return;
        }

        const selectedGesture = this.gestureSelect.value;
        
        // Clear timeout
        if (this.recordingTimeout) {
            clearTimeout(this.recordingTimeout);
            this.recordingTimeout = null;
        }

        // Stop recording
        this.isRecording = false;

        // Check if we have enough data
        if (this.currentRecording.length < 15) {
            this.updateStatus(`Recording too short (${this.currentRecording.length} frames). Need at least 15 frames.`, 'error');
        } else {
            // Save the recording with metadata
            const recordingData = {
                gesture: selectedGesture,
                frames: this.currentRecording,
                frameCount: this.recordingFrameCount,
                timestamp: new Date().toISOString(),
                duration: this.currentRecording.length / 15 // Approximate duration in seconds at 15 fps
            };

            this.collectedData.push(recordingData);
            this.updateStatus(`Recording saved: ${selectedGesture} (${this.recordingFrameCount} frames)`, 'success');
        }

        // Update UI
        this.startRecordingBtn.disabled = false;
        this.stopRecordingBtn.disabled = true;
        this.gestureSelect.disabled = false;
        this.recordingStatus.textContent = 'Ready to record';
        this.datasetCount.textContent = `Recordings in session: ${this.collectedData.length}`;

        // Reset recording data
        this.currentRecording = [];
        this.recordingFrameCount = 0;
        this.frameCount.textContent = 'Frames: 0';
    }

    downloadCollectedData() {
        if (this.collectedData.length === 0) {
            this.updateStatus('No data to download. Record some gestures first.', 'warning');
            return;
        }

        // Create download data with metadata
        const downloadData = {
            metadata: {
                recordingSession: new Date().toISOString(),
                totalRecordings: this.collectedData.length,
                userRole: this.userRole,
                roomId: this.currentRoom
            },
            recordings: this.collectedData
        };

        // Convert to JSON and create download
        const dataStr = JSON.stringify(downloadData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        // Create temporary download link
        const downloadLink = document.createElement('a');
        downloadLink.href = url;
        downloadLink.download = `asl_training_data_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        URL.revokeObjectURL(url);

        this.updateStatus(`Downloaded training data: ${this.collectedData.length} recordings`, 'success');
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.interludeApp = new InterludeApp();
});

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InterludeApp;
}
