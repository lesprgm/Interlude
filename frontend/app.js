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
            this.socket.on('transcribed_text', (data) => { // <--- Changed from 'subtitle' to 'transcribed_text'
                console.log('[Socket.IO] Received transcribed_text event:', data);
                // data: { text: string, isFinal: boolean }
                
                if (this.userRole === 'deaf') {
                    const subtitleDiv = document.getElementById('subtitleDisplay');
                    if (subtitleDiv && data && typeof data.text === 'string') {
                        subtitleDiv.textContent = data.text;
                        // Optionally, add a class for final/partial
                        if (data.isFinal) {
                            subtitleDiv.classList.add('final');
                        } else {
                            subtitleDiv.classList.remove('final');
                        }
                    } else {
                        console.warn('[Socket.IO] transcribed_text event missing text or subtitleDisplay element:', data);
                    }
                } else {
                    console.log('Not displaying subtitles: User role is not "deaf".');
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
                this.startSpeechProcessing(); // This will now include logging for data flow
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

            console.log('Audio track state:', audioTrack.readyState, 'enabled:', audioTrack.enabled);
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
                    console.log(`Sending audio chunk: ${event.data.size} bytes`); // Log chunk size
                    // Send audio chunk to backend
                    this.socket.emit('send_audio_chunk', event.data);
                } else {
                    console.warn('Empty audio chunk received from MediaRecorder.');
                }
            };

            this.mediaRecorder.onstop = () => {
                console.log('MediaRecorder stopped.');
                this.updateStatus('MediaRecorder stopped.', 'info');
                this.isAudioStreaming = false;
                this.speechToAslStatus.textContent = 'Speech processing stopped.';
                this.audioChunks = []; // Clear audio chunks on stop
                // Inform backend that audio stream has ended
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
            // Initialize MediaPipe Holistic
            this.holistic = new window.Holistic({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
                }
            });

            // Set MediaPipe options
            this.holistic.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                enableSegmentation: false,
                smoothSegmentation: false,
                refineFaceLandmarks: false,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            // Set up results callback
            this.holistic.onResults = (results) => {
                // Clear the canvas
                this.localCanvasCtx.clearRect(0, 0, this.localCanvas.width, this.localCanvas.height);
                
                // Draw landmarks if available using MediaPipe drawing utilities
                if (results.poseLandmarks) {
                    window.drawConnectors(this.localCanvasCtx, results.poseLandmarks, window.POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
                    window.drawLandmarks(this.localCanvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
                    console.log('Pose landmarks detected:', results.poseLandmarks.length, 'points');
                }
                
                if (results.leftHandLandmarks) {
                    window.drawConnectors(this.localCanvasCtx, results.leftHandLandmarks, window.HAND_CONNECTIONS, {color: '#CC0000', lineWidth: 2});
                    window.drawLandmarks(this.localCanvasCtx, results.leftHandLandmarks, {color: '#00FF00', lineWidth: 1, radius: 2});
                    console.log('Left hand landmarks detected:', results.leftHandLandmarks.length, 'points');
                }
                
                if (results.rightHandLandmarks) {
                    window.drawConnectors(this.localCanvasCtx, results.rightHandLandmarks, window.HAND_CONNECTIONS, {color: '#0000CC', lineWidth: 2});
                    window.drawLandmarks(this.localCanvasCtx, results.rightHandLandmarks, {color: '#0000FF', lineWidth: 1, radius: 2});
                    console.log('Right hand landmarks detected:', results.rightHandLandmarks.length, 'points');
                }

                // Log the complete results for debugging
                console.log('MediaPipe Results:', {
                    pose: results.poseLandmarks ? results.poseLandmarks.length : 0,
                    leftHand: results.leftHandLandmarks ? results.leftHandLandmarks.length : 0,
                    rightHand: results.rightHandLandmarks ? results.rightHandLandmarks.length : 0
                });

                // Prepare keypoint data for streaming to backend
                const keypointData = this.prepareKeypointData(results);
                
                // Stream keypoints to backend via Socket.IO
                if (this.socket && this.socket.connected && keypointData) {
                    this.socket.emit('asl_keypoints', keypointData);
                    console.log('Sending ASL keypoints to backend:', keypointData);
                }
            };

            // Start processing video frames
            this.processVideoFrames();
            
            this.aslToSpeechStatus.textContent = 'Processing ASL...';
            this.updateStatus('ASL recognition started with MediaPipe Holistic', 'success');
            
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
            
            this.aslToSpeechStatus.textContent = 'ASL recognition stopped';
            this.updateStatus('ASL recognition stopped', 'info');
            
        } catch (error) {
            console.error('Error stopping ASL recognition:', error);
            this.updateStatus(`Error stopping ASL recognition: ${error.message}`, 'error');
        }
    }

    // Process video frames for MediaPipe
    processVideoFrames() {
        const sendFrame = async () => {
            if (this.holistic && this.localVideo.readyState >= 2) { // Video has data
                try {
                    await this.holistic.send({image: this.localVideo});
                } catch (error) {
                    console.error('Error sending frame to MediaPipe:', error);
                }
            }
            
            // Continue processing frames
            if (this.holistic) {
                requestAnimationFrame(sendFrame);
            }
        };
        
        sendFrame();
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

    // Prepare keypoint data for streaming to backend
    prepareKeypointData(results) {
        try {
            const keypointData = {
                timestamp: Date.now(),
                pose: null,
                leftHand: null,
                rightHand: null
            };

            // Process pose landmarks (33 points with visibility)
            if (results.poseLandmarks && results.poseLandmarks.length > 0) {
                keypointData.pose = results.poseLandmarks.map((landmark, index) => ({
                    id: index,
                    x: parseFloat(landmark.x.toFixed(4)),
                    y: parseFloat(landmark.y.toFixed(4)),
                    z: parseFloat(landmark.z.toFixed(4)),
                    visibility: parseFloat((landmark.visibility || 0).toFixed(4))
                }));
            }

            // Process left hand landmarks (21 points)
            if (results.leftHandLandmarks && results.leftHandLandmarks.length > 0) {
                keypointData.leftHand = results.leftHandLandmarks.map((landmark, index) => ({
                    id: index,
                    x: parseFloat(landmark.x.toFixed(4)),
                    y: parseFloat(landmark.y.toFixed(4)),
                    z: parseFloat(landmark.z.toFixed(4))
                }));
            }

            // Process right hand landmarks (21 points)
            if (results.rightHandLandmarks && results.rightHandLandmarks.length > 0) {
                keypointData.rightHand = results.rightHandLandmarks.map((landmark, index) => ({
                    id: index,
                    x: parseFloat(landmark.x.toFixed(4)),
                    y: parseFloat(landmark.y.toFixed(4)),
                    z: parseFloat(landmark.z.toFixed(4))
                }));
            }

            // Only return data if we have at least one set of landmarks
            if (keypointData.pose || keypointData.leftHand || keypointData.rightHand) {
                return keypointData;
            }

            return null;
        } catch (error) {
            console.error('Error preparing keypoint data:', error);
            return null;
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
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.interludeApp = new InterludeApp();
});

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InterludeApp;
}
