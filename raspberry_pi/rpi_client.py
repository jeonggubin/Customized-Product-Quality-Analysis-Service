"""
rpi_client.py — 라즈베리파이5 클라이언트 (카메라 중복 선언 버그 완벽 수정본)
═══════════════════════════════════════════════════════════════
"""

import argparse
import csv
import os
import random
import re
import socket
import threading
import time
from collections import Counter, deque
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("pip install ultralytics 를 먼저 실행하세요.")


# ═══════════════════════════════════════════════════════════════
#  ★ 설정값
# ═══════════════════════════════════════════════════════════════

CLASS_NAMES = ['Bloom', 'Crack', 'NoDefects', 'Crack_Bloom']

CAM_WIDTH,  CAM_HEIGHT  = 640, 480
YOLO_IMGSZ              = 640
DISP_WIDTH, DISP_HEIGHT = 1280, 960

CONF_THRESHOLD = 0.50

MOTOR_PORT = 8888

MOTOR_COOLDOWN = 9.0   # 서보모터 동작 시간(0.5+7+0.5=8초) + 여유 1초
STABLE_SEC     = 3.0   # 정상 제품(NoDefects) 캡처용 안정화 시간

VOTE_FRAMES = 10
VOTE_RATIO  = 0.70

DEFECT_KEYWORDS = ['crack', 'bloom']

# ═══════════════════════════════════════════════════════════════
#  ★ 파이프라인 설정 (product_id / 센서 CSV 경로)
# ═══════════════════════════════════════════════════════════════

PRODUCT_CSV  = "product_id.csv"               # product_id ↔ 파일명 매핑
SENSOR_CSV   = "sensor_with_product_id.csv"   # 센서 더미 데이터

SCREENSHOT_DIR = "images"   # jpg 저장 경로
LABEL_DIR      = "labels"   # txt 저장 경로

# YOLO 클래스 인덱스 → 불량 레이블
CLASS_MAP = {0: "Bloom", 1: "Crack", 2: "Crack_Bloom", 3: "NoDefects"}

# 공정별 사용 센서
_PROCESS_SENSORS = {
    "로스팅": ["temperature_c", "pressure_bar", "smoke_adc"],
    "분쇄":   ["temperature_c", "vibration_hz", "particle_size_um"],
    "콘칭":   ["temperature_c", "humidity_pct", "viscosity_cp"],
    "템퍼링": ["temperature_c", "viscosity_cp"],
    "몰딩":   ["temperature_c", "humidity_pct", "pressure_bar"],
    "냉각":   ["temperature_c", "humidity_pct", "cooling_temp_c"],
}
_PROCESSES = ["로스팅", "분쇄", "콘칭", "템퍼링", "몰딩", "냉각"]

# 불량 유형 × 공정별 센서 범위
_SENSOR_RANGES = {
    "로스팅": {
        "Bloom":       {"temperature_c": (206.6, 225.6), "pressure_bar": (1.12, 1.29), "smoke_adc": (983.0,  2274.0)},
        "Crack":       {"temperature_c": (197.2, 227.6), "pressure_bar": (1.06, 1.30), "smoke_adc": (815.0,  2464.0)},
        "Crack_Bloom": {"temperature_c": (199.0, 237.3), "pressure_bar": (1.09, 1.43), "smoke_adc": (797.0,  1299.0)},
        "NoDefects":   {"temperature_c": (204.4, 252.0), "pressure_bar": (1.11, 1.28), "smoke_adc": (938.0,  1417.0)},
    },
    "분쇄": {
        "Bloom":       {"temperature_c": (37.5, 70.5), "vibration_hz": (45.0, 77.9), "particle_size_um": (20.5, 34.1)},
        "Crack":       {"temperature_c": (48.5, 60.8), "vibration_hz": (49.4, 79.3), "particle_size_um": (19.2, 36.2)},
        "Crack_Bloom": {"temperature_c": (51.1, 61.7), "vibration_hz": (65.7, 76.6), "particle_size_um": (20.5, 37.0)},
        "NoDefects":   {"temperature_c": (52.7, 75.8), "vibration_hz": (55.6, 78.1), "particle_size_um": (20.4, 25.1)},
    },
    "콘칭": {
        "Bloom":       {"temperature_c": (69.2, 78.9), "humidity_pct": (13.9, 34.6), "viscosity_cp": (3351.0, 6038.0)},
        "Crack":       {"temperature_c": (66.3, 81.3), "humidity_pct": (24.3, 36.3), "viscosity_cp": (3315.0, 7369.0)},
        "Crack_Bloom": {"temperature_c": (68.5, 79.4), "humidity_pct": (25.2, 35.1), "viscosity_cp": (3351.0, 6923.0)},
        "NoDefects":   {"temperature_c": (60.0, 88.4), "humidity_pct": (13.0, 34.9), "viscosity_cp": (2387.0, 7197.0)},
    },
    "템퍼링": {
        "Bloom":       {"temperature_c": (26.6, 41.0), "viscosity_cp": (5643.0, 7821.0)},
        "Crack":       {"temperature_c": (20.0, 50.8), "viscosity_cp": (5469.0, 7510.0)},
        "Crack_Bloom": {"temperature_c": (26.4, 48.5), "viscosity_cp": (5561.0, 6331.0)},
        "NoDefects":   {"temperature_c": (25.7, 44.6), "viscosity_cp": (5448.0, 8984.0)},
    },
    "몰딩": {
        "Bloom":       {"temperature_c": (29.5, 31.1), "humidity_pct": (31.9, 48.0), "pressure_bar": (0.42, 0.47)},
        "Crack":       {"temperature_c": (29.3, 34.8), "humidity_pct": (27.5, 51.0), "pressure_bar": (0.25, 0.64)},
        "Crack_Bloom": {"temperature_c": (29.7, 31.7), "humidity_pct": (27.2, 49.9), "pressure_bar": (0.25, 0.48)},
        "NoDefects":   {"temperature_c": (26.5, 31.7), "humidity_pct": (38.8, 53.0), "pressure_bar": (0.42, 0.63)},
    },
    "냉각": {
        "Bloom":       {"temperature_c": (23.6, 29.6), "humidity_pct": (44.3, 51.6), "cooling_temp_c": (10.0, 13.9)},
        "Crack":       {"temperature_c": (24.1, 30.7), "humidity_pct": (32.9, 53.0), "cooling_temp_c": (9.9,  21.6)},
        "Crack_Bloom": {"temperature_c": (24.0, 30.0), "humidity_pct": (44.2, 53.5), "cooling_temp_c": (10.2, 14.1)},
        "NoDefects":   {"temperature_c": (24.0, 30.0), "humidity_pct": (43.4, 53.2), "cooling_temp_c": (9.9,  13.5)},
    },
}

_CLASS_NUM = {"Bloom": 0.0, "Crack": 1.0, "Crack_Bloom": 2.0, "NoDefects": 3.0}

_SENSOR_ALL_COLS = [
    "product_id", "defect_class", "defect_label", "timestamp", "process",
    "temperature_c", "humidity_pct", "pressure_bar", "viscosity_cp",
    "vibration_hz", "smoke_adc", "cooling_temp_c", "particle_size_um",
]

# ───────────────────────────────────────────────────────────────
#  파이프라인 헬퍼 함수
# ───────────────────────────────────────────────────────────────

def _next_product_id() -> str:
    if not os.path.exists(PRODUCT_CSV):
        return "P000001"
    last_pid = "P000000"
    with open(PRODUCT_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            last_pid = row["product_id"]
    num = int(re.sub(r"[^0-9]", "", last_pid))
    return f"P{num + 1:06d}"

def save_screenshot(annotated_frame: "np.ndarray", base_name: str) -> str:
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    img_filename = f"{base_name}.jpg"
    cv2.imwrite(os.path.join(SCREENSHOT_DIR, img_filename), annotated_frame)
    return img_filename

def save_label_txt(base_name: str, defect_class_idx: int, result, img_w: int, img_h: int) -> str:
    os.makedirs(LABEL_DIR, exist_ok=True)
    txt_filename = f"{base_name}.txt"

    if result.boxes is not None and len(result.boxes) > 0:
        best_idx = int(result.boxes.conf.cpu().numpy().argmax())
        box_xyxy = result.boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = box_xyxy
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
    else:
        cx, cy, bw, bh = 0.5, 0.5, 1.0, 1.0

    with open(os.path.join(LABEL_DIR, txt_filename), "w") as f:
        f.write(f"{defect_class_idx} {cx:.10g} {cy:.10g} {bw:.10g} {bh:.10g}\n")

    return txt_filename

def update_product_csv(product_id: str, label_filename: str, image_filename: str):
    write_header = not os.path.exists(PRODUCT_CSV)
    with open(PRODUCT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["product_id", "label_filename", "image_filename"])
        writer.writerow([product_id, label_filename, image_filename])

def append_sensor_data(product_id: str, defect_label: str):
    write_header = not os.path.exists(SENSOR_CSV)
    base_ts      = datetime.now()

    with open(SENSOR_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_SENSOR_ALL_COLS)

        for i, process in enumerate(_PROCESSES):
            ts      = base_ts + timedelta(seconds=i * 1000)
            ranges  = _SENSOR_RANGES[process][defect_label]
            row_d   = {col: "" for col in _SENSOR_ALL_COLS}
            row_d["product_id"]   = product_id
            row_d["defect_class"] = _CLASS_NUM[defect_label]
            row_d["defect_label"] = defect_label
            row_d["timestamp"]    = ts.strftime("%Y-%m-%d %H:%M:%S")
            row_d["process"]      = process

            for sensor, (lo, hi) in ranges.items():
                val = random.uniform(lo, hi)
                if sensor in ("smoke_adc", "viscosity_cp"):
                    row_d[sensor] = f"{val:.0f}"
                elif sensor == "particle_size_um":
                    row_d[sensor] = f"{val:.1f}"
                else:
                    row_d[sensor] = f"{val:.3f}"

            writer.writerow([row_d[c] for c in _SENSOR_ALL_COLS])

def run_capture_pipeline(annotated_frame: "np.ndarray", result, label: str, conf: float, capture_type: str = "auto") -> str:
    class_idx = next((k for k, v in CLASS_MAP.items() if v == label), 3)
    defect_label = CLASS_MAP[class_idx]

    pid       = _next_product_id()
    ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{capture_type}_{ts_str}_{pid}"

    img_h, img_w = annotated_frame.shape[:2]

    img_filename = save_screenshot(annotated_frame, base_name)
    txt_filename = save_label_txt(base_name, class_idx, result, img_w, img_h)
    update_product_csv(pid, txt_filename, img_filename)
    append_sensor_data(pid, defect_label)

    print(f"  📋 product_id  : {pid}")
    print(f"  🏷️  불량 유형  : {defect_label} (class={class_idx})")
    print(f"  📸 캡처 완료   : {SCREENSHOT_DIR}/{img_filename}")
    return pid

# ──────────────────────────────────────────────
#  한국어 폰트
# ──────────────────────────────────────────────

def load_korean_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def put_text_kr(frame, text, pos, font, color_bgr):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img_pil).text(
        pos, text, font=font,
        fill=(color_bgr[2], color_bgr[1], color_bgr[0])
    )
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ──────────────────────────────────────────────
#  추론 결과 파싱
# ──────────────────────────────────────────────

def parse_result(result, conf_thres: float):
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0

    confs   = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    valid_mask = confs >= conf_thres
    if not valid_mask.any():
        return None, 0.0

    v_confs  = confs[valid_mask]
    v_ids    = cls_ids[valid_mask]

    best_i     = int(v_confs.argmax())
    raw_conf   = float(v_confs[best_i])
    best_cls   = int(v_ids[best_i])
    raw_label  = result.names.get(best_cls, f"Class{best_cls}")

    return raw_label, raw_conf

# ──────────────────────────────────────────────
#  화면 오버레이 (상태 배너)
# ──────────────────────────────────────────────

def draw_status_banner(frame, display_label, conf, fps,
                       motor_status, stable_progress, target_sec,
                       font_main, font_sub):
    is_none   = display_label == "감지 못함"
    is_defect = (not is_none) and display_label != 'NoDefects'
    color = (100,100,100) if is_none else (0,0,220) if is_defect else (0,200,60)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1], 88), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if is_none:
        status_text = f"[대기 중]  {conf*100:.1f}%"
    elif is_defect:
        status_text = f"[불량 검출]  {display_label}  {conf*100:.1f}%"
    else:
        status_text = f"[정상 제품]  {display_label}  {conf*100:.1f}%"
    frame = put_text_kr(frame, status_text, (12, 4), font_main, color)

    motor_col = (0,220,80) if motor_status == "연결됨" else (0,100,200)
    frame = put_text_kr(frame,
        f"모터파이: {motor_status}  FPS:{fps:.1f}",
        (frame.shape[1]-300, 4), font_sub, motor_col)

    bar_y = 62
    bx1, bx2 = 12, frame.shape[1] - 12
    filled = int((bx2 - bx1) * stable_progress)
    cv2.rectangle(frame, (bx1, bar_y), (bx2, bar_y+10), (60,60,60), -1)
    if filled > 0:
        bar_col = (0,220,80) if stable_progress >= 1.0 else (0,180,220)
        cv2.rectangle(frame, (bx1, bar_y), (bx1+filled, bar_y+10), bar_col, -1)

    if stable_progress >= 1.0:
        stab_text, stab_col = "✔ 처리 및 캡처 완료", (0,220,80)
    elif is_none:
        stab_text, stab_col = "초콜릿 없음 — 대기 중", (120,120,120)
    else:
        remain = target_sec * (1.0 - stable_progress)
        mode_text = "디바운스" if is_defect else "안정화"
        stab_text, stab_col = f"{mode_text} 대기 {remain:.1f}s", (180,180,180)
    
    frame = put_text_kr(frame, stab_text, (bx1, bar_y+12), font_sub, stab_col)

    h = frame.shape[0]
    frame = put_text_kr(frame, "ESC / Q: 종료",
                        (12, h-28), font_sub, (120,120,120))
    return frame

# ──────────────────────────────────────────────
#  소켓 클라이언트 (모터 파이)
# ──────────────────────────────────────────────

class MotorClient:
    def __init__(self, host: str, port: int):
        self._sock      = None
        self._connected = False
        self._lock      = threading.Lock()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((host, port))
            sock.settimeout(None)
            self._sock      = sock
            self._connected = True
            print(f"✅  모터파이 소켓 연결 성공: {host}:{port}")
        except Exception as e:
            print(f"⚠️   모터파이 연결 실패: {e}")

    def send(self, message: str) -> bool:
        if not self._connected or self._sock is None:
            return False
        try:
            with self._lock:
                self._sock.sendall(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"📡 모터 통신 에러: {e}")
            self._connected = False
            return False

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    @property
    def status(self):
        return "연결됨" if self._connected else "연결 안됨"

# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="라즈베리파이5 클라이언트")
    parser.add_argument("--motor-host", default="10.10.16.199")
    parser.add_argument("--motor-port", type=int, default=MOTOR_PORT)
    parser.add_argument("--model",      default="best.onnx")
    args = parser.parse_args()

    # ── 모델 로드 ─────────────────────────────────────────────
    print(f"📦 YOLO 모델 로드 중: {args.model}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"모델 없음: {args.model}")
    model = YOLO(args.model, task="detect")
    print("✅ 모델 로드 완료\n")

    font_main = load_korean_font(32)
    font_sub  = load_korean_font(18)

    motor_client = MotorClient(args.motor_host, args.motor_port)

    # ── 📍 웹캠 포트를 1번으로 단 한 번만 올바르게 설정합니다 ──
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("❌ 카메라를 열 수 없습니다.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    WINDOW_NAME = "Smart Factory - Chocolate Defect Detect"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISP_WIDTH, DISP_HEIGHT)

    # ── 타이머 및 상태 변수 ────────────────────────────────────
    prev_time  = time.time()
    capture_count = 0
    label_buffer = deque(maxlen=VOTE_FRAMES)

    stable_label = None
    stable_start = None
    already_sent = False
    last_motor_time = 0.0

    # ── 디바운싱 설정 변수 ─────────────────────────────────────
    DEBOUNCE_SEC = 1.5         # 상태 확정을 위한 디바운싱 시간 (1.5초)
    defect_start_time = None   # 불량이 처음 시작된 시각 기록용
    is_state_confirmed = False # 1.5초 디바운싱 통과 여부

    print("=" * 55)
    print("▶ 스마트팩토리 비전 마스터 가동 (완벽 차단 버전)")
    print(f"  모터파이   : {args.motor_host}:{args.motor_port}")
    print(f"  정상안정화 : {STABLE_SEC}초")
    print(f"  불량디바운스: {DEBOUNCE_SEC}초")
    print(f"  모터쿨다운 : {MOTOR_COOLDOWN}초")
    print("  ESC/Q: 종료")
    print("=" * 55 + "\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # ── YOLO 추론 ────────────────────────────────────────
        results = model(frame, conf=CONF_THRESHOLD, imgsz=YOLO_IMGSZ, verbose=False)
        result  = results[0]

        annotated = result.plot()
        annotated = cv2.resize(annotated, (frame.shape[1], frame.shape[0]))

        raw_label, raw_conf = parse_result(result, CONF_THRESHOLD)

        # ── 다수결 투표 ───────────────────────────────────────
        label_buffer.append(raw_label)
        vote  = Counter(label_buffer)
        total = len(label_buffer)
        best_vote_label, best_vote_cnt = vote.most_common(1)[0]

        if best_vote_cnt / total >= VOTE_RATIO:
            label = best_vote_label
            conf  = raw_conf if raw_label == label else 0.0
        else:
            label = None
            conf  = 0.0

        display_label = label if label else "감지 못함"
        now = time.time()

        # ════════════════════════════════════════════════════════
        #  💡 핵심 상태머신
        # ════════════════════════════════════════════════════════

        if label is None:
            defect_start_time = None
            is_state_confirmed = False
            stable_label = None
            stable_start = None
            already_sent = False

        elif label != "NoDefects":
            stable_label = None
            stable_start = None
            
            if defect_start_time is None:
                defect_start_time = now
                is_state_confirmed = False
                already_sent = False
                print(f"⏳ [디바운스 시작] 불량 감지 [{label}] -> 1.5초 유지 검사 중...")
            else:
                elapsed_defect = now - defect_start_time
                if elapsed_defect >= DEBOUNCE_SEC and not is_state_confirmed:
                    is_state_confirmed = True
                    print(f"🚨 [디바운스 통과] 1.5초 연속 유지 확정! 진짜 불량으로 판정합니다.")
                    
                    if now - last_motor_time > MOTOR_COOLDOWN:
                        last_motor_time = now
                        packet = label.upper()
                        if motor_client.send(packet):
                            print(f"   ➔ 📡 모터파이 소켓 전송 완료: [{packet}]")

                    if not already_sent:
                        print(f"   ➔ 📸 불량 캡처 진행 중...")
                        new_pid = run_capture_pipeline(annotated, result, label, conf)
                        capture_count += 1
                        already_sent = True

        else: # label == "NoDefects"
            defect_start_time = None
            is_state_confirmed = False

            if stable_label != "NoDefects":
                stable_label = "NoDefects"
                stable_start = now
                already_sent = False
            else:
                if stable_start is not None and (now - stable_start) >= STABLE_SEC and not already_sent:
                    print(f"✅ [안정화 완료] 정상 제품 3초 유지 확정 -> 캡처 진행 중...")
                    new_pid = run_capture_pipeline(annotated, result, label, conf)
                    capture_count += 1
                    already_sent = True

        # ── 화면 렌더링을 위한 프로그레스 값 계산 ─────────────────
        if defect_start_time is not None:
            current_target_sec = DEBOUNCE_SEC
            stable_progress = min((now - defect_start_time) / DEBOUNCE_SEC, 1.0)
        elif stable_start is not None:
            current_target_sec = STABLE_SEC
            stable_progress = min((now - stable_start) / STABLE_SEC, 1.0)
        else:
            current_target_sec = STABLE_SEC
            stable_progress = 0.0

        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        annotated = draw_status_banner(
            annotated, display_label, conf, fps,
            motor_client.status, stable_progress, current_target_sec,
            font_main, font_sub,
        )
        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            print("👋 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()
    motor_client.close()
    print("종료 완료.")


if __name__ == "__main__":
    main()
