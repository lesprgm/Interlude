# Frontend Engineer Migration Guide: MediaPipe to OpenCV.js

## 🎯 **Mission Critical: ASL Keypoint Detection Pivot**

**Subject:** Urgent Action Required - ASL Keypoint Detection Pivot (MediaPipe to OpenCV.js)

### **Current Situation**
- MediaPipe Holistic library is not reliably providing real-time keypoint data
- Backend is ready to receive `asl_keypoints` events but frontend data flow is broken
- Need robust solution for real-time hand/pose tracking for 2-user demo

### **✅ COMPLETED WORK**

I've already implemented the OpenCV.js transition for you! Here's what's been done:

#### **1. HTML Changes (index.html)**
```html
<!-- REMOVED: MediaPipe scripts -->
<!-- <script crossorigin="anonymous" src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"></script> -->

<!-- ADDED: OpenCV.js -->
<script async src="https://docs.opencv.org/4.8.0/opencv.js" onload="onOpenCvReady();"></script>
```

#### **2. New OpenCV ASL Handler (js/asl/opencv-handler.js)**
- ✅ Complete OpenCV.js integration
- ✅ Real-time frame processing
- ✅ Keypoint data generation (synthetic for demo)
- ✅ Canvas visualization
- ✅ Backend communication via Socket.IO
- ✅ Memory management and cleanup

#### **3. Updated Main App (app.js)**
- ✅ Replaced MediaPipe Holistic with OpenCV handler
- ✅ Updated initialization methods
- ✅ Modified startAslRecognition() and stopAslRecognition()
- ✅ Added OpenCV ready callback

### **🔧 WHAT YOU NEED TO DO**

#### **Step 1: Test the Current Implementation**
```bash
# Start your backend server
cd backend
python main.py

# Open the test page in your browser
open frontend/test-opencv.html
```

#### **Step 2: Verify Backend Integration**
1. **Start your backend server**
2. **Open your main app** (`frontend/index.html`)
3. **Join a room as a deaf user**
4. **Start a call** - you should see:
   - OpenCV keypoints drawn on canvas
   - Console logs showing keypoint data being sent
   - Backend logs: `🔍 DEBUG: Received ASL keypoints from [sid]`

#### **Step 3: Replace Synthetic Keypoints with Real Detection**

**Current Status:** The implementation uses synthetic keypoints for demo purposes.

**Your Task:** Replace `generateSyntheticKeypoints()` with real pose/hand detection.

### **🎯 CRITICAL IMPLEMENTATION TASKS**

#### **Task 1: Load Real Pre-trained Models**

Replace this section in `opencv-handler.js`:

```javascript
// CURRENT: Synthetic keypoints for demo
async loadModels() {
    await this.simulateModelLoading();
}
```

**With real model loading:**

```javascript
async loadModels() {
    try {
        // Load pose detection model
        this.poseNet = this.cv.readNetFromTensorflow('models/pose_model.pb', 'models/pose_model.pbtxt');
        
        // Load hand detection model  
        this.handNet = this.cv.readNetFromTensorflow('models/hand_model.pb', 'models/hand_model.pbtxt');
        
        console.log('✅ Real models loaded successfully');
    } catch (error) {
        console.error('❌ Failed to load models:', error);
        throw error;
    }
}
```

#### **Task 2: Implement Real Keypoint Detection**

Replace this section:

```javascript
// CURRENT: Synthetic keypoints
detectKeypoints(mat) {
    const keypointData = this.generateSyntheticKeypoints(mat.cols, mat.rows);
    return keypointData;
}
```

**With real detection:**

```javascript
detectKeypoints(mat) {
    try {
        // Preprocess image
        const blob = this.cv.blobFromImage(mat, 1.0/255, [256, 256], [0, 0, 0], true, false);
        
        // Run pose detection
        this.poseNet.setInput(blob);
        const poseOutput = this.poseNet.forward();
        
        // Run hand detection
        this.handNet.setInput(blob);
        const handOutput = this.handNet.forward();
        
        // Parse outputs into keypoint format
        const keypointData = this.parseModelOutputs(poseOutput, handOutput, mat.cols, mat.rows);
        
        // Clean up
        blob.delete();
        poseOutput.delete();
        handOutput.delete();
        
        return keypointData;
        
    } catch (error) {
        console.error('❌ Error detecting keypoints:', error);
        return null;
    }
}
```

#### **Task 3: Parse Model Outputs**

Add this method to parse DNN model outputs:

```javascript
parseModelOutputs(poseOutput, handOutput, width, height) {
    // Parse pose landmarks (33 points)
    const poseLandmarks = [];
    const poseData = poseOutput.data32F;
    
    for (let i = 0; i < 33; i++) {
        const x = poseData[i * 3] / width;
        const y = poseData[i * 3 + 1] / height;
        const confidence = poseData[i * 3 + 2];
        
        poseLandmarks.push({
            x: x,
            y: y,
            z: 0.0,
            visibility: confidence
        });
    }
    
    // Parse hand landmarks (21 points each)
    const leftHandLandmarks = [];
    const rightHandLandmarks = [];
    const handData = handOutput.data32F;
    
    // Parse left hand (first 21 points)
    for (let i = 0; i < 21; i++) {
        const x = handData[i * 3] / width;
        const y = handData[i * 3 + 1] / height;
        const z = handData[i * 3 + 2];
        
        leftHandLandmarks.push({ x, y, z });
    }
    
    // Parse right hand (next 21 points)
    for (let i = 21; i < 42; i++) {
        const x = handData[i * 3] / width;
        const y = handData[i * 3 + 1] / height;
        const z = handData[i * 3 + 2];
        
        rightHandLandmarks.push({ x, y, z });
    }
    
    return {
        timestamp: Date.now(),
        pose: poseLandmarks,
        leftHand: leftHandLandmarks,
        rightHand: rightHandLandmarks
    };
}
```

### **📋 MODEL SOURCES & RECOMMENDATIONS**

#### **Option 1: Use Pre-trained OpenCV Models**
```bash
# Download OpenCV pose estimation models
wget https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/openpose_pose_coco.caffemodel
wget https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/openpose_pose_coco.prototxt
```

#### **Option 2: Convert TensorFlow.js Models**
```javascript
// Convert MediaPipe models to OpenCV format
// Use tools like ONNX Runtime or TensorFlow.js conversion
```

#### **Option 3: Use Lightweight Models**
- **PoseNet** (TensorFlow.js) → Convert to OpenCV
- **HandPose** (MediaPipe) → Extract and convert
- **BlazePose** → Convert to OpenCV DNN format

### **🧪 TESTING CHECKLIST**

#### **Phase 1: Basic OpenCV Integration**
- [ ] OpenCV.js loads successfully
- [ ] Camera access works
- [ ] Frame processing runs without errors
- [ ] Synthetic keypoints are generated and sent to backend

#### **Phase 2: Real Model Integration**
- [ ] Pre-trained models load successfully
- [ ] Real keypoint detection works
- [ ] Keypoint data format matches backend expectations
- [ ] Performance is acceptable (30fps target)

#### **Phase 3: Backend Integration**
- [ ] Backend receives `asl_keypoints` events
- [ ] ASL recognition model processes the data
- [ ] TTS is triggered for hearing users
- [ ] End-to-end flow works correctly

### **🚨 CRITICAL FORMAT REQUIREMENTS**

The keypoint data sent to backend MUST match this exact format:

```javascript
{
    timestamp: Date.now(),
    pose: [
        { x: 0.5, y: 0.3, z: 0.0, visibility: 0.8 },
        // ... 33 pose landmarks total
    ],
    leftHand: [
        { x: 0.4, y: 0.6, z: 0.0 },
        // ... 21 left hand landmarks total
    ],
    rightHand: [
        { x: 0.6, y: 0.6, z: 0.0 },
        // ... 21 right hand landmarks total
    ]
}
```

### **🔍 DEBUGGING TOOLS**

#### **1. Browser Console Monitoring**
```javascript
// Add to your keypoint detection
console.log('📤 Sending keypoints:', keypointData);
```

#### **2. Backend Log Monitoring**
Look for these logs in your backend:
```
🔍 DEBUG: Received ASL keypoints from [sid]
🎯 ASL Prediction: [sign] with confidence [X]
🔊 TTS: Synthesizing speech for [text]
```

#### **3. Test Page**
Use `frontend/test-opencv.html` to test each component independently.

### **⚡ PERFORMANCE OPTIMIZATIONS**

#### **1. Frame Rate Control**
```javascript
// Process every 3rd frame for performance
this.processingInterval = 3;
```

#### **2. Canvas Optimization**
```javascript
// Use requestAnimationFrame for smooth rendering
requestAnimationFrame(() => this.processFrames());
```

#### **3. Memory Management**
```javascript
// Always clean up OpenCV Mats
mat.delete();
blob.delete();
```

### **🎯 SUCCESS CRITERIA**

**You'll know it's working when:**
1. ✅ OpenCV.js loads without errors
2. ✅ Camera feed shows keypoints drawn on canvas
3. ✅ Backend logs show: `🔍 DEBUG: Received ASL keypoints from [sid]`
4. ✅ Hearing users receive TTS audio for ASL signs
5. ✅ Performance is smooth (no lag or freezing)

### **🚨 EMERGENCY FALLBACK**

If OpenCV.js integration fails, you can temporarily revert to MediaPipe:

```javascript
// In app.js, comment out OpenCV and uncomment MediaPipe
// this.opencvHandler = new OpenCVASLHandler(this);
this.holistic = new window.Holistic({...});
```

### **📞 SUPPORT**

**For immediate issues:**
1. Check browser console for errors
2. Verify OpenCV.js is loading (`cv.version`)
3. Test with `frontend/test-opencv.html`
4. Monitor backend logs for keypoint reception

**The backend is ready and waiting for your keypoint data!** 🎯

---

**Priority:** HIGH  
**Deadline:** ASAP  
**Status:** OpenCV.js framework implemented, needs real model integration 