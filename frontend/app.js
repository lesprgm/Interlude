// BridgeSpeak - Real-time Communication Aid
// Main application JavaScript

class BridgeSpeakApp {
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

        this.initializeSocket();
        this.bindEventListeners();
        this.checkBrowserCompatibility();
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
    }

    initializeSocket() {
        try {
            // Connect to the backend signaling server
            this.socket = io('http://localhost:8000');
            
            // Socket event handlers
            this.socket.on('connect', () => {
                this.connectionStatus.textContent = 'Connected to server';
                this.connectionStatus.className = 'status connected';
                this.updateStatus('Connected to signaling server', 'success');
            });

            this.socket.on('disconnect', () => {
                this.connectionStatus.textContent = 'Disconnected';
                this.connectionStatus.className = 'status';
                this.updateStatus('Disconnected from signaling server', 'error');
            });

            // WebRTC Signaling Events
            this.socket.on('user-joined', (data) => {
                this.remotePeerId = data.userId;
                this.isInitiator = true;
                this.updateStatus(`User joined. Preparing to initiate call...`, 'info');
                
                if (this.localStream && this.peerConnection) {
                    this.createOffer();
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
        this.updateStatus(`Joining room: ${roomId}...`, 'info');
        
        // Hide room selection UI
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

            this.updateStatus('Starting camera and microphone...', 'info');
            
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
                this.toggleVideoBtn.textContent = this.isVideoEnabled ? 'Turn Off Video' : 'Turn On Video';
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
                this.toggleAudioBtn.textContent = this.isAudioEnabled ? 'Mute' : 'Unmute';
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
                this.localStream.getTracks().forEach(track => {
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

    // WebRTC Signaling Methods
    async createOffer() {
        try {
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
            // Ensure we have a peer connection
            if (!this.peerConnection) {
                await this.initializePeerConnection();
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
        
        // Clear remote video
        if (this.remoteVideo) {
            this.remoteVideo.srcObject = null;
        }
        
        // Reset remote peer info
        this.remotePeerId = null;
        this.isInitiator = false;
        
        // Show room selection again
        document.getElementById('roomSelection').style.display = 'block';
        
        // Close peer connection
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }
    }

    startAslRecognition() {
        // ASL recognition initialization
        this.aslToSpeechStatus.textContent = 'Listening for signs...';
    }

    startSpeechProcessing() {
        // Speech-to-text and ASL avatar generation
        this.speechToAslStatus.textContent = 'Listening for speech...';
    }

    updateStatus(message, type = 'info') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.bridgeSpeakApp = new BridgeSpeakApp();
});

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BridgeSpeakApp;
} 