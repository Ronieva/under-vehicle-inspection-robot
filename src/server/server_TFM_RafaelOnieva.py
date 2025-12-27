import sys
import time
import math
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
import serial
from flask import Flask, Response, render_template_string, jsonify, request
from openvino.runtime import Core


# ----------------------------
# Configuration
# ----------------------------
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUDRATE = 115200

COLOR_W, COLOR_H, COLOR_FPS = 640, 480, 15
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 480, 15

HOST = "0.0.0.0"
PORT = 8000

# Model input sizes used by each network (as in your original script)
ARMAS_SIZE = 640
MOCHILAS_SIZE = 960
RUEDAS_SIZE = 960

CONF_ARMAS = 0.5
CONF_RUEDAS = 0.5
CONF_MOCHILAS = 0.7

NMS_IOU = 0.45


app = Flask(__name__)

# ----------------------------
# Arduino serial (teleoperation commands)
# ----------------------------
try:
    arduino = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=0)
    time.sleep(2)  # Give the board time to reset after opening the port
    print(f"Arduino connected on {SERIAL_PORT}")
except Exception as e:
    arduino = None
    print(f"Arduino connection error: {e}")


# ----------------------------
# RealSense pipeline
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
pipeline.start(config)

# Depth filters (used for more robust wheel depth estimation)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()


# ----------------------------
# OpenVINO models (YOLO)
# ----------------------------
ie = Core()

models = {
    "armas": ie.compile_model("pesos_armas.onnx", "CPU"),
    "ruedas": ie.compile_model("pesos_ruedas.onnx", "CPU"),
    "mochilas": ie.compile_model("pesos_mochilitas.onnx", "CPU"),
}
outputs = {k: v.output(0) for k, v in models.items()}


# ----------------------------
# Shared state (protected with a lock)
# ----------------------------
lock = threading.Lock()

frame_rgb = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
depth_frame_global = None

armas_boxes = []
mochilas_boxes = []
ruedas_boxes = []

imu_data = {"acc": [0.0, 0.0, 0.0], "gyro": [0.0, 0.0, 0.0]}
posicion_rel = {"X": "---", "Z": "---", "angulo": "---"}

# Last rendered frame that already contains bounding boxes / overlays
last_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)

# Pixel centroids of the wheel pair selected for relative position estimation
ruedas_centroides_seleccionadas = None  # ((cx1, cy1), (cx2, cy2))


# ----------------------------
# Helper functions
# ----------------------------
def preprocess(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize and normalize an image to match YOLO input requirements.
    Output shape: (1, 3, size, size) float32 in [0, 1].
    """
    img_r = cv2.resize(img, (size, size))
    img_r = img_r.transpose(2, 0, 1)[None] / 255.0
    return img_r.astype(np.float32)


def process_yolo_out(output: np.ndarray, conf: float = 0.7) -> np.ndarray:
    """
    Convert raw YOLO output into bounding boxes in (x1, y1, x2, y2) format.
    Confidence threshold is applied on objectness score p[4].
    """
    preds = np.squeeze(output).T
    dets = []
    for p in preds:
        if p[4] < conf:
            continue
        x, y, w, h = p[:4]
        dets.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    return np.array(dets)


def nms(dets: np.ndarray, iou: float = 0.45) -> np.ndarray:
    """
    Apply non-maximum suppression to reduce duplicate detections.
    """
    if len(dets) == 0:
        return dets

    boxes = []
    scores = []
    for d in dets:
        x1, y1, x2, y2 = d
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        scores.append(0.9)  # fixed score, because objectness was already thresholded

    idx = cv2.dnn.NMSBoxes(boxes, scores, 0.4, iou)
    if len(idx) == 0:
        return np.empty((0, 4))
    return dets[idx.flatten()]


def pixel_to_3d(cx: int, cy: int, dist: float, intr) -> tuple[float, float, float]:
    """
    Project a pixel (cx, cy) with depth 'dist' into 3D camera coordinates.
    Units follow RealSense depth (typically meters).
    """
    X = (cx - intr.ppx) * dist / intr.fx
    Y = (cy - intr.ppy) * dist / intr.fy
    Z = dist
    return X, Y, Z


def get_depth_safe(depth_frame: rs.depth_frame, cx: int, cy: int) -> float:
    """
    Robust depth retrieval:
    - Use center pixel if valid
    - Otherwise take the median of valid values in a 7x7 neighborhood
    """
    d = depth_frame.get_distance(cx, cy)
    if d > 0:
        return d

    neighbors = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            xx, yy = cx + dx, cy + dy
            if 0 <= xx < COLOR_W and 0 <= yy < COLOR_H:
                dd = depth_frame.get_distance(xx, yy)
                if dd > 0:
                    neighbors.append(dd)

    if not neighbors:
        return 0.0
    return float(np.median(neighbors))


def send_to_arduino(cmd: str) -> bool:
    """
    Send a motion command to the Arduino controller.
    """
    global arduino
    if arduino is None:
        print("Arduino not available. Command ignored.")
        return False

    try:
        message = cmd.strip() + "\n"
        print(f"Sending command to Arduino: {message.strip()}")
        sys.stdout.flush()

        # Clear old buffered content to reduce delayed commands
        arduino.reset_output_buffer()
        arduino.reset_input_buffer()

        arduino.write(message.encode())
        arduino.flush()
        return True

    except Exception as e:
        print(f"Arduino write error: {e}")
        return False


# ----------------------------
# Worker threads
# ----------------------------
def thread_rgb_imu():
    """
    Continuously capture RGB, depth, and IMU data from the RealSense device.
    Updates shared frames and IMU measurements.
    """
    global frame_rgb, depth_frame_global, imu_data

    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        # Read IMU frames from the same frameset
        for f in frames:
            if f.is_motion_frame():
                d = f.as_motion_frame().get_motion_data()
                if f.get_profile().stream_type() == rs.stream.accel:
                    imu_data["acc"] = [d.x, d.y, d.z]
                elif f.get_profile().stream_type() == rs.stream.gyro:
                    imu_data["gyro"] = [d.x, d.y, d.z]

        frame = np.asanyarray(color.get_data())

        with lock:
            frame_rgb = frame.copy()
            depth_frame_global = depth


def thread_armas():
    """
    Run the weapons detection model in a tight loop and update shared bounding boxes.
    """
    global armas_boxes

    while True:
        with lock:
            frame = frame_rgb.copy()

        if frame.sum() == 0:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]

        inp = preprocess(frame, ARMAS_SIZE)
        out = models["armas"]([inp])[outputs["armas"]]
        dets = nms(process_yolo_out(out, conf=CONF_ARMAS), iou=NMS_IOU)

        sx = w / float(ARMAS_SIZE)
        sy = h / float(ARMAS_SIZE)

        scaled = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in dets]

        with lock:
            armas_boxes = scaled

        time.sleep(0.01)


def thread_mochilas():
    """
    Run the backpack detection model in a tight loop and update shared bounding boxes.
    """
    global mochilas_boxes

    while True:
        with lock:
            frame = frame_rgb.copy()

        if frame.sum() == 0:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]

        inp = preprocess(frame, MOCHILAS_SIZE)
        out = models["mochilas"]([inp])[outputs["mochilas"]]
        dets = nms(process_yolo_out(out, conf=CONF_MOCHILAS), iou=NMS_IOU)

        sx = w / float(MOCHILAS_SIZE)
        sy = h / float(MOCHILAS_SIZE)

        scaled = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in dets]

        with lock:
            mochilas_boxes = scaled

        time.sleep(0.01)


def thread_ruedas_position():
    """
    Detect wheels, estimate the 3D position of the closest wheel pair,
    and compute a relative yaw angle with respect to the camera axis.
    """
    global ruedas_boxes, posicion_rel, ruedas_centroides_seleccionadas

    while True:
        with lock:
            frame = frame_rgb.copy()
            depth = depth_frame_global

        if depth is None or frame.sum() == 0:
            time.sleep(0.01)
            continue

        # Apply depth filtering to improve measurement stability
        depth_f = spatial.process(depth)
        depth_f = temporal.process(depth_f)
        depth_f = hole_filling.process(depth_f)
        depth_f = depth_f.as_depth_frame()

        intr = depth.profile.as_video_stream_profile().intrinsics

        h, w = frame.shape[:2]

        inp = preprocess(frame, RUEDAS_SIZE)
        out = models["ruedas"]([inp])[outputs["ruedas"]]
        dets = nms(process_yolo_out(out, conf=CONF_RUEDAS), iou=NMS_IOU)

        sx = w / float(RUEDAS_SIZE)
        sy = h / float(RUEDAS_SIZE)

        scaled = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in dets]

        with lock:
            ruedas_boxes = scaled

        # Build a list of wheel candidates with 3D coordinates
        wheels_world = []  # each entry: {"X":..., "Z":..., "cx":..., "cy":...}

        for x1, y1, x2, y2 in scaled:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if not (0 <= cx < COLOR_W and 0 <= cy < COLOR_H):
                continue

            dist = get_depth_safe(depth_f, cx, cy)
            if dist <= 0:
                continue

            X, _, Z = pixel_to_3d(cx, cy, dist, intr)
            wheels_world.append({"X": X, "Z": Z, "cx": cx, "cy": cy})

        pos_new = {"X": "---", "Z": "---", "angulo": "---"}
        cent_pair = None

        # Select the two wheels closest to the camera (smallest Z)
        if len(wheels_world) >= 2:
            wheels_sorted = sorted(wheels_world, key=lambda r: r["Z"])
            r1, r2 = wheels_sorted[0], wheels_sorted[1]

            Xc = (r1["X"] + r2["X"]) / 2.0
            Zc = (r1["Z"] + r2["Z"]) / 2.0

            # Relative yaw angle (camera -> platform direction)
            angle_deg = math.degrees(math.atan2(Xc, Zc))

            pos_new = {"X": round(Xc, 3), "Z": round(Zc, 3), "angulo": round(angle_deg, 2)}
            cent_pair = ((r1["cx"], r1["cy"]), (r2["cx"], r2["cy"]))

        with lock:
            posicion_rel = pos_new
            ruedas_centroides_seleccionadas = cent_pair

        time.sleep(0.02)


def thread_render():
    """
    Render bounding boxes and overlays on top of the RGB frame.
    The result is stored in 'last_frame' for streaming and snapshots.
    """
    global last_frame

    while True:
        with lock:
            base = frame_rgb.copy()
            armas = list(armas_boxes)
            moch = list(mochilas_boxes)
            rued = list(ruedas_boxes)
            cent_pair = ruedas_centroides_seleccionadas

        if base.sum() == 0:
            time.sleep(0.01)
            continue

        # Wheels (green)
        for x1, y1, x2, y2 in rued:
            cv2.rectangle(base, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(base, "Wheel", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Line between the selected wheel pair
        if cent_pair is not None:
            (cx1, cy1), (cx2, cy2) = cent_pair
            cv2.line(base, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

        # Weapons (red)
        for x1, y1, x2, y2 in armas:
            cv2.rectangle(base, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(base, "Weapon", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Backpacks (blue)
        for x1, y1, x2, y2 in moch:
            cv2.rectangle(base, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(base, "Backpack", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        with lock:
            last_frame = cv2.resize(base, (COLOR_W, COLOR_H))

        time.sleep(0.02)


# ----------------------------
# Web endpoints
# ----------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>RealSense Stream</title>
<style>
img {
    width: 100%;
    height: auto;
    object-fit: contain;
    background: black;
}
body {
    margin: 0;
    padding: 0;
    background: black;
}
</style>
</head>
<body>
<img src="{{ stream_url }}">
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE, stream_url="/video_rgb")


@app.route("/view/<mode>")
def view(mode: str):
    return render_template_string(HTML_TEMPLATE, stream_url="/video_" + mode)


@app.route("/video_rgb")
def video_rgb():
    def gen():
        while True:
            with lock:
                f = last_frame.copy()
            ok, buf = cv2.imencode(".jpg", f)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_depth")
def video_depth():
    def gen():
        while True:
            frames = pipeline.wait_for_frames()
            d = frames.get_depth_frame()
            if not d:
                continue
            img = np.asanyarray(d.get_data())
            img = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.08), cv2.COLORMAP_JET)
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot_rgb")
def snapshot_rgb():
    with lock:
        f = last_frame.copy()
    ok, buf = cv2.imencode(".jpg", f)
    if not ok:
        return Response(status=500)
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/snapshot_depth")
def snapshot_depth():
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return Response(status=500)

    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    ok, buf = cv2.imencode(".jpg", depth_colored)
    if not ok:
        return Response(status=500)

    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/cmd", methods=["POST"])
def cmd():
    data = request.get_json() or {}
    print(f"Received /cmd payload: {data}")

    action = (data.get("accion") or data.get("cmd") or "").lower().strip()
    if action == "":
        return jsonify({"status": "ok"})

    valid_commands = {
        "adelante", "atras",
        "der_adelante", "izq_adelante",
        "der_atras", "izq_atras",
        "parar", "giro_der", "giro_izq"
    }

    if action not in valid_commands:
        print(f"Invalid command: {action}")
        return jsonify({"status": "error", "msg": "invalid command"}), 400

    send_to_arduino(action)
    return jsonify({"status": "ok"})


@app.route("/estado_plataforma")
def estado_plataforma():
    with lock:
        pos = posicion_rel.copy()
        imu = imu_data.copy()
    return jsonify({"posicion": pos, "imu": imu})


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    threading.Thread(target=thread_rgb_imu, daemon=True).start()
    threading.Thread(target=thread_armas, daemon=True).start()
    threading.Thread(target=thread_mochilas, daemon=True).start()
    threading.Thread(target=thread_ruedas_position, daemon=True).start()
    threading.Thread(target=thread_render, daemon=True).start()

    print(f"Starting server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True)

