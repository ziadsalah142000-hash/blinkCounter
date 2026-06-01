import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from faceMeshModule import FaceMeshDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'grad-proj-secret-key')

# Enable CORS for cross-origin connections (e.g. Flutter app, external clients)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize the detector
detector = FaceMeshDetector(maxFaces=1)

# Landmark IDs for left and right eyes (matching blinkCounter.py)
leftEyeIDs = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
rightEyeIDs = [263, 362, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

# Active connection state manager to handle multi-client sessions
client_states = {}

def decode_image(data_uri):
    """
    Decodes an image from standard binary bytes or base64 data URI string.
    """
    try:
        if isinstance(data_uri, str):
            if "," in data_uri:
                data_uri = data_uri.split(",")[1]
            image_data = base64.b64decode(data_uri)
        else:
            image_data = data_uri

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Failed to decode image: {e}")
        return None

@app.route("/")
def home():
    """
    Returns API status message.
    """
    return jsonify({
        "status": "online",
        "project": "Graduation Project AI Backend",
        "websockets_enabled": True
    })

@app.route("/health", methods=["GET"])
def health():
    """
    Server health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "websockets_enabled": True,
        "active_connections": len(client_states)
    })

@socketio.on('connect')
def handle_connect():
    """
    Triggered when a client connects. Initializes a clean blink counter session.
    """
    sid = request.sid
    client_states[sid] = {
        'ratioListLeft': [],
        'ratioListRight': [],
        'blinkCount': 0,
        'counterTime': 0
    }
    print(f"[SOCKET] Client connected: {sid} (Total active: {len(client_states)})")
    emit('connection_response', {'status': 'connected', 'sid': sid})

@socketio.on('disconnect')
def handle_disconnect():
    """
    Triggered when a client disconnects. Cleans up session resources to prevent memory leaks.
    """
    sid = request.sid
    if sid in client_states:
        del client_states[sid]
    print(f"[SOCKET] Client disconnected: {sid} (Total active: {len(client_states)})")

@socketio.on('reset_counter')
def handle_reset_counter():
    """
    Resets the blink count for the current client's session.
    """
    sid = request.sid
    if sid in client_states:
        client_states[sid]['blinkCount'] = 0
        client_states[sid]['ratioListLeft'] = []
        client_states[sid]['ratioListRight'] = []
        client_states[sid]['counterTime'] = 0
        emit('reset_response', {'status': 'success', 'blinkCount': 0})

@socketio.on('process_frame')
def handle_process_frame(data):
    """
    Core WebSocket event. Receives camera frame, runs detector & blink counters,
    and returns real-time calculations.
    """
    sid = request.sid
    if sid not in client_states:
        # Fallback if connection lifecycle wasn't captured correctly
        client_states[sid] = {
            'ratioListLeft': [],
            'ratioListRight': [],
            'blinkCount': 0,
            'counterTime': 0
        }

    state = client_states[sid]
    
    # Extract request options
    frame_data = data.get('image')
    draw_mesh = data.get('draw_mesh', True)
    return_image = data.get('return_image', False)

    if not frame_data:
        emit('frame_error', {'error': 'No image frame provided'})
        return

    # Decode frame
    img = decode_image(frame_data)
    if img is None:
        emit('frame_error', {'error': 'Invalid image format or decoding failed'})
        return

    # Process using FaceMeshDetector
    # If client requested returning drawn-on image, we draw on it, otherwise False to save CPU cycles
    img, faces = detector.findFaceMesh(img, draw=(draw_mesh or return_image))

    response = {
        'faces_detected': len(faces),
        'blink_detected': False,
        'blink_count': state['blinkCount'],
        'ratio_left': 0.0,
        'ratio_right': 0.0,
        'ratio_avg': 0.0,
        'landmarks': []
    }

    if faces:
        face = faces[0]  # Focus on the first detected face
        response['landmarks'] = face.tolist()

        # Extract Left Eye Landmarks
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        
        # Calculate left eye distances
        vertLeft, _ = FaceMeshDetector.findDistance(leftUp, leftDown)
        horLeft, _ = FaceMeshDetector.findDistance(leftLeft, leftRight)

        # Extract Right Eye Landmarks
        rightUp = face[386]
        rightDown = face[374]
        rightLeft = face[263]
        rightRight = face[362]

        # Calculate right eye distances
        vertRight, _ = FaceMeshDetector.findDistance(rightUp, rightDown)
        horRight, _ = FaceMeshDetector.findDistance(rightLeft, rightRight)

        # Compute Aspect Ratios
        ratioLeft = (vertLeft / horLeft) * 100 if horLeft > 0 else 0.0
        ratioRight = (vertRight / horRight) * 100 if horRight > 0 else 0.0
        ratioAvg = (ratioLeft + ratioRight) / 2.0

        # Append to smooth averages
        state['ratioListLeft'].append(ratioLeft)
        state['ratioListRight'].append(ratioRight)
        if len(state['ratioListLeft']) > 3:
            state['ratioListLeft'].pop(0)
            state['ratioListRight'].pop(0)

        ratioAvgLeft = sum(state['ratioListLeft']) / len(state['ratioListLeft'])
        ratioAvgRight = sum(state['ratioListRight']) / len(state['ratioListRight'])
        smoothedRatioAvg = (ratioAvgLeft + ratioAvgRight) / 2.0

        # Blink Detection Logic
        blink_event = False
        if smoothedRatioAvg < 33 and state['counterTime'] == 0:
            state['blinkCount'] += 1
            state['counterTime'] = 1
            blink_event = True

        if state['counterTime'] != 0:
            state['counterTime'] += 1
            if state['counterTime'] > 10:
                state['counterTime'] = 0

        # Update response fields
        response['blink_detected'] = blink_event
        response['blink_count'] = state['blinkCount']
        response['ratio_left'] = float(round(ratioLeft, 2))
        response['ratio_right'] = float(round(ratioRight, 2))
        response['ratio_avg'] = float(round(ratioAvg, 2))
        response['smoothed_ratio_avg'] = float(round(smoothedRatioAvg, 2))

    # Optional: return the fully processed and drawn-on image frame
    if return_image:
        try:
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response['processed_image'] = f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            print(f"[ERROR] Failed to encode processed frame: {e}")

    emit('frame_result', response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[SERVER] Starting real-time WebSocket server on port {port}...")
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
