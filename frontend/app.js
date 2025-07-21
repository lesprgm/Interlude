// BridgeSpeak - Real-time Communication Aid
// Main application JavaScript

class BridgeSpeakApp {
    constructor() {
        this.localVideo = null;
        this.remoteVideo = null;
        this.localStream = null;
        this.remoteStream = null;
        this.peerConnection = null;
        this.isCallActive = false;
        this.isVideoEnabled = true;
        this.isAudioEnabled = true;
        
        // WebRTC Configuration with STUN servers
        this.rtcConfiguration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };
        
        // Initialize the application
        this.init();
    }

    init() {
        console.log('Initializing BridgeSpeak...');
        
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

        // Bind event listeners
        this.bindEventListeners();

        // Check browser compatibility
        this.checkBrowserCompatibility();
    }

    bindEventListeners() {
        this.startCallBtn.addEventListener('click', () => this.startCall());
        this.endCallBtn.addEventListener('click', () => this.endCall());
        this.toggleVideoBtn.addEventListener('click', () => this.toggleVideo());
        this.toggleAudioBtn.addEventListener('click', () => this.toggleAudio());
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
        
        console.log('WebRTC compatibility check passed:', checks);
        return true;
    }

    async startCall() {
        try {
            this.updateStatus('Starting camera and microphone...', 'info');
            
            // Enhanced media constraints for better quality
            const mediaConstraints = {
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

            // Request access to camera and microphone with enhanced constraints
            this.localStream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
            console.log('Local media stream acquired:', this.localStream);

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

            // Start ASL recognition and speech processing (placeholder)
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
            console.log('Initializing RTCPeerConnection with configuration:', this.rtcConfiguration);
            
            // Create RTCPeerConnection with STUN server configuration
            this.peerConnection = new RTCPeerConnection(this.rtcConfiguration);
            
            // Add local stream tracks to peer connection
            if (this.localStream) {
                this.localStream.getTracks().forEach(track => {
                    console.log('Adding local track to peer connection:', track.kind);
                    this.peerConnection.addTrack(track, this.localStream);
                });
            }

            // Set up WebRTC event handlers
            this.setupPeerConnectionEventHandlers();
            
            console.log('RTCPeerConnection initialized successfully');
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
            console.log('ICE candidate event:', event);
            if (event.candidate) {
                console.log('New ICE candidate:', event.candidate);
                // TODO: Send candidate to remote peer via signaling server
            } else {
                console.log('ICE candidate gathering complete');
            }
        };

        // Handle ICE connection state changes
        this.peerConnection.oniceconnectionstatechange = () => {
            console.log('ICE connection state:', this.peerConnection.iceConnectionState);
            this.updateConnectionStatus();
        };

        // Handle peer connection state changes
        this.peerConnection.onconnectionstatechange = () => {
            console.log('Peer connection state:', this.peerConnection.connectionState);
            this.updateConnectionStatus();
        };

        // Handle remote stream
        this.peerConnection.ontrack = (event) => {
            console.log('Remote track received:', event);
            if (event.streams && event.streams[0]) {
                this.remoteStream = event.streams[0];
                this.remoteVideo.srcObject = this.remoteStream;
                this.updateStatus('Remote video stream connected!', 'success');
            }
        };

        // Handle negotiation needed
        this.peerConnection.onnegotiationneeded = () => {
            console.log('Negotiation needed');
            // TODO: Handle offer/answer negotiation when implementing signaling
        };
    }

    updateConnectionStatus() {
        const iceState = this.peerConnection?.iceConnectionState || 'not-started';
        const connectionState = this.peerConnection?.connectionState || 'not-started';
        
        console.log(`Connection status - ICE: ${iceState}, Peer: ${connectionState}`);
        
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

    startAslRecognition() {
        // Placeholder for ASL recognition initialization
        // This will integrate with MediaPipe for hand/pose detection
        console.log('Starting ASL recognition...');
        this.aslToSpeechStatus.textContent = 'Listening for signs...';
    }

    startSpeechProcessing() {
        // Placeholder for speech-to-text and ASL avatar generation
        // This will integrate with Google Cloud Speech-to-Text and GenASL API
        console.log('Starting speech processing...');
        this.speechToAslStatus.textContent = 'Listening for speech...';
    }

    updateStatus(message, type = 'info') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;
        console.log(`Status (${type}): ${message}`);
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