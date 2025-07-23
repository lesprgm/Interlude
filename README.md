# Interlude - Real-time Communication Aid

WebRTC video calling application with Socket.IO signaling for real-time peer-to-peer communication.

## How to Run

### 1. Start Backend Server
```bash
cd backend
source ../venv/bin/activate
python main.py
```

### 2. Start Frontend Server
```bash
cd frontend
python3 -m http.server 3000
```

### 3. Access Application
- Open browser to `127.0.0.1:3000`
- For testing: Open 2 browser tabs, join same room, start calls in both

---

## Google Cloud Backend VM

- **VM Name:** `##########`
- **External IP:** `###########` 
- **Zone:** (e.g., us-central1-c)
- **Machine Type:** `c2-standard-8` (8 vCPUs, 32 GB memory)
- **Boot Disk Image:** Ubuntu 22.04 LTS (or Debian 11/DLVM Image if you chose that)
- **Confirmed:** SSH access established, Python/pip installed, Firewall rule for port 8000 created.
