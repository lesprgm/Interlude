# Interlude - Real-time Communication Aid

Interlude is a real-time video calling application designed to facilitate communication between deaf/hard of hearing and hearing individuals. It leverages cutting-edge technologies to provide real-time translation of American Sign Language (ASL) to speech and speech to text.

## Features

*   **Real-time Video Calling:** High-quality video and audio communication using WebRTC.
*   **ASL-to-Speech Translation:**
    *   For deaf users, the application captures video and uses an OpenCV-based solution to analyze ASL signs.
    *   A trained Keras model on the backend predicts the ASL sign from the video stream.
    *   The predicted sign is vocalized as speech for the hearing user using the Eleven Labs text-to-speech API.
*   **Speech-to-Text Transcription:**
    *   For hearing users, the application captures audio and uses Google Cloud Speech-to-Text to generate real-time transcriptions.
    *   The transcriptions are displayed as subtitles for the deaf user.
*   **User Roles:** The application has two distinct user roles: "deaf" and "hearing", each with a tailored user experience.
*   **ASL Data Collection:** A built-in tool for collecting ASL gesture data to train and improve the recognition model.

## How to Run

### 1. Prerequisites

*   Python 3.8+
*   Node.js and npm
*   Google Cloud SDK
*   An Eleven Labs account and API key

### 2. Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the `backend` directory and add the following:
    ```
    ELEVENLABS_API_KEY="your_eleven_labs_api_key"
    GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-cloud-credentials.json"
    ```

5.  **Start the backend server:**
    ```bash
    python main.py
    ```
    The backend will be running at `http://localhost:8000`.

### 3. Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install dependencies (if any):**
    ```bash
    npm install
    ```

3.  **Start the frontend development server:**
    ```bash
    npm start
    ```
    The frontend will be running at `http://localhost:3000`.

### 4. Access Application

*   Open your browser to `http://localhost:3000`.
*   For testing, you can open two browser tabs, join the same room, and start a call. Make sure to select the appropriate "deaf" or "hearing" role for each user.

## Google Cloud Backend VM

- **VM Name:** `##########`
- **External IP:** `###########` 
- **Zone:** (e.g., us-central1-c)
- **Machine Type:** `c2-standard-8` (8 vCPUs, 32 GB memory)
- **Boot Disk Image:** Ubuntu 22.04 LTS (or Debian 11/DLVM Image if you chose that)
- **Confirmed:** SSH access established, Python/pip installed, Firewall rule for port 8000 created.

## Configuration

### Eleven Labs API Key

To use the text-to-speech functionality, you need to sign up for an account at [Eleven Labs](https://beta.elevenlabs.io/) and get an API key.

### Google Cloud Credentials

To use the speech-to-text functionality, you need a Google Cloud Platform account with the Speech-to-Text API enabled. You will need to create a service account and download the JSON credentials file.

## Usage

1.  **Join a Room:** Enter a room ID and choose your role ("deaf" or "hearing").
2.  **Start a Call:** Click the "Start Call" button to begin the video call.
3.  **Communicate:**
    *   If you are a "deaf" user, your ASL signs will be translated to speech for the "hearing" user.
    *   If you are a "hearing" user, your speech will be transcribed to text for the "deaf" user.
4.  **End a Call:** Click the "End Call" button to terminate the session.
---

