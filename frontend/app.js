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
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.updateStatus('Browser does not support WebRTC. Please use a modern browser.', 'error');
            return false;
        }
        return true;
    }

    async startCall() {
        try {
            this.updateStatus('Starting camera and microphone...', 'info');
            
            // Request access to camera and microphone
            this.localStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            });

            // Display local video stream
            this.localVideo.srcObject = this.localStream;

            // Update UI state
            this.isCallActive = true;
            this.startCallBtn.disabled = true;
            this.endCallBtn.disabled = false;
            this.updateStatus('Call started! Waiting for remote participant...', 'success');

            // Initialize WebRTC peer connection (placeholder for now)
            this.initializePeerConnection();

            // Start ASL recognition and speech processing (placeholder)
            this.startAslRecognition();
            this.startSpeechProcessing();

        } catch (error) {
            console.error('Error starting call:', error);
            this.updateStatus('Failed to start call. Please check camera/microphone permissions.', 'error');
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

    initializePeerConnection() {
        // Placeholder for WebRTC peer connection setup
        // This will be implemented when we add WebRTC functionality
        console.log('Initializing peer connection...');
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