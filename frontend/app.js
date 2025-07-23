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
        
        // Audio streaming for STT
        this.mediaRecorder = null;
        this.isAudioStreaming = false;
        this.audioChunks = [];
        
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
            this.socket = io('http://localhost:8000');
            
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

        this.currentRoom = roomId;
        this.socket.emit('join_room', { roomId: roomId });
        this.updateStatus(`Joining room: ${roomId}...`, 'info', true);
        
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

            // Start ASL recognition and speech processing
            this.startAslRecognition();
            this.startSpeechProcessing();

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
                    
                    // Resume audio streaming if call is active
                    if (this.isCallActive && !this.isAudioStreaming) {
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
            
        } catch (error) {
            console.error('Error initializing peer connection:', error);
            this.updateStatus(`Failed to initialize WebRTC connection: ${error.message}`, 'error');
            throw error;
        }
    }

    setupPeerConnectionEventHandlers() {
        // Handle ICE candidate events
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate && this.remotePeerId && this.socket) {
                this.socket.emit('webrtc-ice-candidate', {
                    candidate: event.candidate,
                    to: this.remotePeerId,
                    from: this.socket.id
                });
            }
        };

        // Handle ICE connection state changes
        this.peerConnection.oniceconnectionstatechange = () => {
            this.updateConnectionStatus();
        };

        // Handle peer connection state changes
        this.peerConnection.onconnectionstatechange = () => {
            this.updateConnectionStatus();
        };

        // Handle remote stream
        this.peerConnection.ontrack = (event) => {
            if (event.streams && event.streams[0]) {
                this.remoteStream = event.streams[0];
                this.remoteVideo.srcObject = this.remoteStream;
                // Hide placeholder and show video
                this.remoteVideoPlaceholder.style.display = 'none';
                this.updateStatus('Remote video stream connected!', 'success');
            }
        };

        // Handle negotiation needed
        this.peerConnection.onnegotiationneeded = async () => {
            if (this.isInitiator && this.remotePeerId && this.socket && this.socket.connected) {
                await this.createOffer();
            }
        };
    }

    updateConnectionStatus() {
        const iceState = this.peerConnection?.iceConnectionState || 'not-started';
        
        // Update UI based on connection state
        if (iceState === 'connected' || iceState === 'completed') {
            this.updateStatus('WebRTC connection established successfully!', 'success');
        } else if (iceState === 'disconnected') {
            this.updateStatus('WebRTC connection lost, attempting to reconnect...', 'info');
        } else if (iceState === 'failed') {
            this.updateStatus('WebRTC connection failed', 'error');
        } else if (iceState === 'checking') {
            this.updateStatus('Establishing WebRTC connection...', 'info');
        }
    }

    // Helper function to add local tracks to peer connection
    addLocalTracksToConnection() {
        if (!this.peerConnection || !this.localStream) {
            return;
        }

        // Check if tracks are already added
        const senders = this.peerConnection.getSenders();
        const hasVideoSender = senders.some(sender => sender.track?.kind === 'video');
        const hasAudioSender = senders.some(sender => sender.track?.kind === 'audio');

        if (hasVideoSender && hasAudioSender) {
            return;
        }

        this.localStream.getTracks().forEach((track) => {
            const existingSender = senders.find(sender => sender.track?.kind === track.kind);
            if (!existingSender) {
                this.peerConnection.addTrack(track, this.localStream);
            }
        });
    }

    // WebRTC Signaling Methods
    async createOffer() {
        try {
            if (!this.peerConnection || !this.remotePeerId) {
                return;
            }
            
            const offer = await this.peerConnection.createOffer({
                offerToReceiveAudio: true,
                offerToReceiveVideo: true
            });
            
            await this.peerConnection.setLocalDescription(offer);
            
            this.socket.emit('webrtc-offer', {
                offer: offer,
                to: this.remotePeerId,
                from: this.socket.id
            });
            
            this.updateStatus('WebRTC offer sent to remote peer', 'info');
        } catch (error) {
            console.error('Error creating offer:', error);
            this.updateStatus(`Failed to create offer: ${error.message}`, 'error');
        }
    }

    async handleOffer(offer, senderId) {
        try {
            // Ensure we have local media first
            if (!this.localStream) {
                this.updateStatus('Getting camera for incoming call...', 'info');
                
                try {
                    // Get user media with same constraints as startCall
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

                    try {
                        this.localStream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
                    } catch (constraintError) {
                        mediaConstraints = { video: true, audio: true };
                        this.localStream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
                    }
                    
                    // Display local video
                    this.localVideo.srcObject = this.localStream;
                    
                } catch (error) {
                    console.error('Failed to get media for incoming call:', error);
                    this.updateStatus('Failed to access camera for incoming call', 'error');
                    return;
                }
            }
            
            // Ensure we have a peer connection
            if (!this.peerConnection) {
                await this.initializePeerConnection();
            } else {
                // If peer connection exists, make sure our local tracks are added
                this.addLocalTracksToConnection();
            }
            
            await this.peerConnection.setRemoteDescription(offer);
            
            const answer = await this.peerConnection.createAnswer();
            await this.peerConnection.setLocalDescription(answer);
            
            this.socket.emit('webrtc-answer', {
                answer: answer,
                to: senderId,
                from: this.socket.id
            });
            
            this.updateStatus('WebRTC answer sent to remote peer', 'success');
        } catch (error) {
            console.error('Error handling offer:', error);
            this.updateStatus(`Failed to handle offer: ${error.message}`, 'error');
        }
    }

    async handleAnswer(answer) {
        try {
            // Ensure we have a peer connection
            if (!this.peerConnection) {
                this.updateStatus('Error: No peer connection for answer', 'error');
                return;
            }
            
            await this.peerConnection.setRemoteDescription(answer);
            this.updateStatus('WebRTC connection established!', 'success');
        } catch (error) {
            console.error('Error handling answer:', error);
            this.updateStatus(`Failed to handle answer: ${error.message}`, 'error');
        }
    }

    async handleIceCandidate(candidate) {
        try {
            // Ensure we have a peer connection
            if (!this.peerConnection) {
                await this.initializePeerConnection();
            }
            
            await this.peerConnection.addIceCandidate(candidate);
        } catch (error) {
            console.error('Error adding ICE candidate:', error);
        }
    }

    handleUserLeft(userId) {
        this.updateStatus('Remote user disconnected', 'info');
        
        // Stop audio streaming since conversation is interrupted
        this.stopAudioStreaming();
        
        // Clear remote video
        if (this.remoteVideo) {
            this.remoteVideo.srcObject = null;
        }
        
        // Reset remote peer info
        this.remotePeerId = null;
        this.isInitiator = false;
        
        // Show room selection modal again
        document.getElementById('roomSelection').style.display = 'flex';
        
        // Show remote video placeholder
        if (this.remoteVideoPlaceholder) {
            this.remoteVideoPlaceholder.style.display = 'flex';
        }
        
        // Reset communication status
        this.speechToAslStatus.textContent = 'Ready';
        this.aslToSpeechStatus.textContent = 'Ready';
    }

    startAslRecognition() {
        // ASL recognition initialization
        this.aslToSpeechStatus.textContent = 'Listening for signs...';
    }

    startSpeechProcessing() {
        if (!this.localStream) {
            this.updateStatus('No audio stream available for speech processing', 'error');
            return;
        }

        try {
            // Check if MediaRecorder is supported
            if (!window.MediaRecorder) {
                throw new Error('MediaRecorder not supported by this browser');
            }

            // Get only the audio track for STT processing
            const audioTrack = this.localStream.getAudioTracks()[0];
            if (!audioTrack) {
                throw new Error('No audio track found in stream');
            }

            // Create a new MediaStream with just the audio track
            const audioStream = new MediaStream([audioTrack]);
            
            // Configure MediaRecorder for optimal STT processing
            const options = {
                mimeType: 'audio/webm;codecs=opus', // Opus codec is efficient for speech
                audioBitsPerSecond: 128000 // 128 kbps for good quality speech
            };

            // Fallback MIME types if opus not supported
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                if (MediaRecorder.isTypeSupported('audio/webm')) {
                    options.mimeType = 'audio/webm';
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    options.mimeType = 'audio/mp4';
                } else {
                    // Use default
                    delete options.mimeType;
                }
            }

            // Initialize MediaRecorder
            this.mediaRecorder = new MediaRecorder(audioStream, options);
            this.audioChunks = [];

            // Set up MediaRecorder event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.handleAudioChunk(event.data);
                }
            };

            this.mediaRecorder.onstart = () => {
                console.log('MediaRecorder started for STT');
                this.isAudioStreaming = true;
                this.speechToAslStatus.textContent = 'Initializing speech recognition...';
                
                // Notify backend to start audio streaming
                if (this.socket && this.socket.connected) {
                    this.socket.emit('start-audio-stream', {
                        mimeType: this.mediaRecorder.mimeType,
                        timestamp: Date.now()
                    });
                }
            };

            this.mediaRecorder.onstop = () => {
                console.log('MediaRecorder stopped for STT');
                this.isAudioStreaming = false;
                this.speechToAslStatus.textContent = 'Speech processing stopped';
                
                // Notify backend to stop audio streaming
                if (this.socket && this.socket.connected) {
                    this.socket.emit('stop-audio-stream', {
                        timestamp: Date.now()
                    });
                }
            };

            this.mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
                this.updateStatus(`Audio recording error: ${event.error.name}`, 'error');
                this.speechToAslStatus.textContent = 'Speech processing error';
                this.stopAudioStreaming();
            };

            // Start recording in chunks (send data every 1 second for real-time STT)
            this.mediaRecorder.start(1000);
            this.updateStatus('Started audio capture for speech recognition', 'success');

        } catch (error) {
            console.error('Error starting speech processing:', error);
            this.updateStatus(`Failed to start speech processing: ${error.message}`, 'error');
            this.speechToAslStatus.textContent = 'Speech processing failed';
        }
    }

    async handleAudioChunk(audioBlob) {
        if (!this.isAudioStreaming || !this.socket || !this.socket.connected) {
            return;
        }

        try {
            // Convert blob to base64 for transmission over Socket.IO
            const reader = new FileReader();
            reader.onload = () => {
                const base64Data = reader.result.split(',')[1]; // Remove data:audio/webm;base64, prefix
                
                // Send audio chunk to backend
                this.socket.emit('audio-chunk', {
                    audioData: base64Data,
                    timestamp: Date.now(),
                    size: audioBlob.size,
                    type: audioBlob.type
                });
            };
            
            reader.onerror = (error) => {
                console.error('Error reading audio blob:', error);
                this.updateStatus('Error processing audio chunk', 'error');
            };
            
            reader.readAsDataURL(audioBlob);
            
        } catch (error) {
            console.error('Error handling audio chunk:', error);
            this.updateStatus('Error processing audio for speech recognition', 'error');
        }
    }

    stopAudioStreaming() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                this.mediaRecorder.stop();
            } catch (error) {
                console.error('Error stopping MediaRecorder:', error);
            }
        }
        
        // Clean up
        this.mediaRecorder = null;
        this.isAudioStreaming = false;
        this.audioChunks = [];
        this.speechToAslStatus.textContent = 'Ready';
    }

    updateStatus(message, type = 'info', showSpinner = false, autoHide = true) {
        // Enhanced status message with optional loading spinner
        const statusContent = this.statusMessage.querySelector('.status-content');
        if (statusContent) {
            // Clear existing content
            statusContent.innerHTML = '';
            
            // Add spinner if requested
            if (showSpinner || (type === 'info' && message.includes('...'))) {
                const spinner = document.createElement('div');
                spinner.className = `spinner ${type}`;
                statusContent.appendChild(spinner);
            }
            
            // Add message text
            const textSpan = document.createElement('span');
            textSpan.textContent = message;
            statusContent.appendChild(textSpan);
            
            // Add loading dots for ongoing operations
            if (message.includes('...')) {
                textSpan.className = 'loading-dots';
            }
        }
        
        // Update status message classes with enhanced animations
        this.statusMessage.className = `status-message ${type} show`;
        
        // Enhanced auto-hide behavior
        if (autoHide) {
            const hideDelay = type === 'error' ? 7000 : type === 'success' ? 4000 : 5000;
            setTimeout(() => {
                this.statusMessage.classList.remove('show');
            }, hideDelay);
        }
        
        // Update connection indicator based on message type
        this.updateConnectionIndicator(type, message);
    }
    
    updateConnectionIndicator(type, message) {
        // Enhanced connection status indicators
        if (message.includes('Connected to signaling server') || message.includes('WebRTC connection established')) {
            this.connectionStatus.className = 'status-dot connected';
            this.connectionText.textContent = 'Connected';
        } else if (message.includes('Connecting') || message.includes('Starting') || message.includes('Getting')) {
            this.connectionStatus.className = 'status-dot connecting';
            this.connectionText.textContent = 'Connecting...';
        } else if (type === 'error' && (message.includes('failed') || message.includes('error'))) {
            this.connectionStatus.className = 'status-dot error';
            this.connectionText.textContent = 'Connection Error';
        } else if (message.includes('Disconnected')) {
            this.connectionStatus.className = 'status-dot';
            this.connectionText.textContent = 'Disconnected';
        }
    }

    initializeUI() {
        // Set initial connection state
        this.connectionText.textContent = 'Connecting...';
        
        // Set initial button states
        this.toggleVideoBtn.classList.add('active');
        this.toggleAudioBtn.classList.add('active');
        
        // Initialize modal display
        document.getElementById('roomSelection').style.display = 'flex';
        
        // Load saved settings
        this.loadSettings();
    }

    addToggleSwitchListeners() {
        // Make toggle sliders clickable for better UX
        const toggleSwitches = document.querySelectorAll('.toggle-switch');
        
        toggleSwitches.forEach(toggleSwitch => {
            const checkbox = toggleSwitch.querySelector('input[type="checkbox"]');
            const slider = toggleSwitch.querySelector('.toggle-slider');
            const label = toggleSwitch.querySelector('.toggle-label');
            
            // Make slider and label clickable
            [slider, label].forEach(element => {
                if (element) {
                    element.addEventListener('click', (e) => {
                        e.preventDefault();
                        checkbox.checked = !checkbox.checked;
                        
                        // Trigger change event manually
                        const changeEvent = new Event('change', { bubbles: true });
                        checkbox.dispatchEvent(changeEvent);
                    });
                    
                    // Add cursor pointer
                    element.style.cursor = 'pointer';
                }
            });
        });
    }

    // Settings functionality
    openSettings() {
        this.settingsModal.style.display = 'flex';
    }

    closeSettings() {
        this.settingsModal.style.display = 'none';
    }

    saveSettings() {
        const settings = {
            theme: this.themeToggle.checked ? 'dark' : 'light',
            username: this.usernameInput.value || 'You',
            videoQuality: this.videoQualitySelect.value,
            defaultMute: this.defaultMuteToggle.checked,
            defaultVideoOff: this.defaultVideoOffToggle.checked
        };

        // Save to localStorage
        localStorage.setItem('interludeSettings', JSON.stringify(settings));
        
        // Apply settings immediately
        this.applySettings(settings);
        
        // Show success message
        this.updateStatus('Settings saved successfully!', 'success');
        
        // Close modal
        this.closeSettings();
    }

    loadSettings() {
        const savedSettings = localStorage.getItem('interludeSettings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            this.applySettings(settings);
            
            // Update form values
            this.themeToggle.checked = settings.theme === 'dark';
            this.usernameInput.value = settings.username || 'You';
            this.videoQualitySelect.value = settings.videoQuality || '720p';
            this.defaultMuteToggle.checked = settings.defaultMute || false;
            this.defaultVideoOffToggle.checked = settings.defaultVideoOff || false;
        }
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