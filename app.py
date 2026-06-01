from flask import Flask, request, jsonify
from flask_sock import Sock
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import os

app = Flask(__name__)
sock = Sock(app)

# ==============================
# FaceMeshDetector (inline)
# ==============================
class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=staticMode,
            max_num_faces=maxFaces,
            min_detection_confidence=minDetectionCon,
            min_tracking_confidence=minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms,
                        self.mpFaceMesh.FACEMESH_CONTOURS,
                        self.drawSpec, self.drawSpec
                    )
                face = np.array([
                    [int(lm.x * img.shape[1]), int(lm.y * img.shape[0])]
                    for lm in faceLms.landmark
                ])
                faces.append(face)
        return img, faces

    @staticmethod
    def findDistance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        length = np.hypot(x2 - x1, y2 - y1)
        return length, (x1, y1, x2, y2, (x1+x2)//2, (y1+y2)//2)


# ==============================
# Globals
# ==============================
detector = FaceMeshDetector()

BLINK_RATIO_THRESHOLD = 33
BLINK_COOLDOWN_FRAMES = 10


# ==============================
# Core logic
# ==============================
def process_frame(image_bytes, state):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}

    _, faces = detector.findFaceMesh(img, draw=False)

    if not faces:
        state["ratioListLeft"].clear()
        state["ratioListRight"].clear()
        return {
            "face_detected":  False,
            "blink_detected": False,
            "blink_count":    state["blinkCount"],
            "ratio_avg":      0.0,
            "landmarks":      []
        }

    face = faces[0]

    vertLeft,  _ = detector.findDistance(tuple(face[159]), tuple(face[23]))
    horLeft,   _ = detector.findDistance(tuple(face[130]), tuple(face[243]))
    vertRight, _ = detector.findDistance(tuple(face[386]), tuple(face[374]))
    horRight,  _ = detector.findDistance(tuple(face[263]), tuple(face[362]))

    ratioLeft  = (vertLeft  / horLeft)  * 100 if horLeft  > 0 else 0
    ratioRight = (vertRight / horRight) * 100 if horRight > 0 else 0

    state["ratioListLeft"].append(ratioLeft)
    state["ratioListRight"].append(ratioRight)
    if len(state["ratioListLeft"])  > 3: state["ratioListLeft"].pop(0)
    if len(state["ratioListRight"]) > 3: state["ratioListRight"].pop(0)

    avgLeft  = sum(state["ratioListLeft"])  / len(state["ratioListLeft"])
    avgRight = sum(state["ratioListRight"]) / len(state["ratioListRight"])
    ratioAvg = (avgLeft + avgRight) / 2

    blink_detected = False
    if ratioAvg < BLINK_RATIO_THRESHOLD and state["counterTime"] == 0:
        state["blinkCount"] += 1
        state["counterTime"]  = 1
        blink_detected        = True

    if state["counterTime"] != 0:
        state["counterTime"] += 1
        if state["counterTime"] > BLINK_COOLDOWN_FRAMES:
            state["counterTime"] = 0

    h, w = img.shape[:2]
    landmarks = [
        {"id": i, "x": round(int(pt[0]) / w, 4), "y": round(int(pt[1]) / h, 4)}
        for i, pt in enumerate(face)
    ]

    return {
        "face_detected":  True,
        "blink_detected": blink_detected,
        "blink_count":    state["blinkCount"],
        "ratio_avg":      round(ratioAvg, 2),
        "landmarks":      landmarks
    }


def new_state():
    return {
        "blinkCount":     0,
        "counterTime":    0,
        "ratioListLeft":  [],
        "ratioListRight": [],
    }


# ==============================
# HTTP endpoints
# ==============================
@app.route("/")
def home():
    return (
        "Blink Counter API | "
        "HTTP POST /predict/blink | "
        "WebSocket  /ws/blink"
    )

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/predict/blink", methods=["POST"])
def predict_blink():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})
    try:
        return jsonify(process_frame(request.files["image"].read(), new_state()))
    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# WebSocket endpoint
# ==============================
@sock.route("/ws/blink")
def ws_blink(ws):
    state = new_state()          # one state per connection
    while True:
        try:
            data = ws.receive()
            if data is None:
                break

            # Accept raw bytes OR JSON { "frame": "<base64>" }
            if isinstance(data, (bytes, bytearray)):
                image_bytes = bytes(data)
            else:
                image_bytes = base64.b64decode(json.loads(data)["frame"])

            ws.send(json.dumps(process_frame(image_bytes, state)))

        except Exception as e:
            ws.send(json.dumps({"error": str(e)}))
            break


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
