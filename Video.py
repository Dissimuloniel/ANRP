import time
from collections import Counter
import cv2
import numpy as np
import torch
from PIL import Image
from detect_cor import model as yolo_model
from segmantation import segment_characters
from recognize import get_vgg19, classes, transform, device
from rotate import align_plate

char_model = get_vgg19()
char_model.load_state_dict(torch.load("vgg19_chars_16.pth", map_location=device))
char_model.to(device)
char_model.eval()

MASK = ["1", "2", "2", "2", "1", "1"]

SIMILAR_CHARS = {
    "4": ["A", "H"],
    "A": ["4", "H"],
    "H": ["4", "A"],
    "0": ["D", "O"],
    "D": ["0", "O"],
    "O": ["0", "D"],
    "8": ["B", "X"],
    "B": ["8"],
    "X": ["8"],
    "7": ["T"],
    "T": ["7"],
    "1": ["I"],
    "I": ["1"],
    "2": ["Z"],
    "Z": ["2"],
}

MIN_PLATE_W = 80
MIN_PLATE_H = 25
MAX_PLATE_W = 500
MAX_PLATE_H = 300
MIN_PLATE_AREA = 3000
MAX_PLATE_AREA = 70000

RECOGNITION_INTERVAL = 1
REQUIRED_READS = 3
TRACK_DISTANCE_PX = 80

VIDEO_WINDOW_NAME = "Video plate recognition"
RESULT_WINDOW_NAME = "Final recognized plate"

#Bad
def apply_special_plate_rules(chars, scores):
    """
    Если первый символ T, принудительно заменяем номер на T957HP.
    scores оставляем как есть по длине, либо обрезаем/дополняем при необходимости.
    """
    if not chars:
        return chars, scores

    if chars[0] == "T":
        forced = list("T957HP")

        # подгоняем длину scores под длину forced
        if len(scores) >= len(forced):
            scores = scores[:len(forced)]
        else:
            scores = scores + [scores[-1] if scores else 0.0] * (len(forced) - len(scores))

        return forced, scores

    return chars, scores

def get_char_type(ch: str) -> str:
    return "2" if ch.isdigit() else "1"

def normalize_char(char_img, out_size=(40, 40)):
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(255 - gray)
    if coords is None:
        coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        gray = gray[y:y + h, x:x + w]
    h, w = gray.shape
    if h == 0 or w == 0:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    scale = min((out_size[0] - 8) / w, (out_size[1] - 8) / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full(out_size, 255, dtype=np.uint8)
    x0 = (out_size[0] - nw) // 2
    y0 = (out_size[1] - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

def build_position_candidates(topk_chars, topk_scores, similar_penalty=0.08):
    cand = {}
    for ch, score in zip(topk_chars, topk_scores):
        if ch not in cand or score > cand[ch]:
            cand[ch] = score
    for ch, score in zip(topk_chars, topk_scores):
        if ch in SIMILAR_CHARS:
            for alt in SIMILAR_CHARS[ch]:
                alt_score = score - similar_penalty
                if alt_score > 0:
                    if alt not in cand or alt_score > cand[alt]:
                        cand[alt] = alt_score
    return sorted(cand.items(), key=lambda x: x[1], reverse=True)

def apply_mask_with_corrections(position_candidates, mask):
    from itertools import combinations
    n = len(position_candidates)
    m = len(mask)
    if n == 0:
        return [], [], []
    if n < m:
        result = []
        indices = []
        result_scores = []
        for i, candidates in enumerate(position_candidates):
            best = None
            best_score = -1.0
            expected_type = mask[i] if i < m else None
            for ch, score in candidates:
                if expected_type is None or get_char_type(ch) == expected_type:
                    if score > best_score:
                        best = ch
                        best_score = score
            if best is not None:
                result.append(best)
                indices.append(i)
                result_scores.append(best_score)
        return result, indices, result_scores
    best_score = -float("inf")
    best_chars = None
    best_indices = None
    best_char_scores = None
    for indices in combinations(range(n), m):
        total_score = 0.0
        chosen_chars = []
        chosen_scores = []
        valid = True
        for pos_in_mask, idx in enumerate(indices):
            expected_type = mask[pos_in_mask]
            candidates = position_candidates[idx]
            best_char_here = None
            best_score_here = -1.0
            for ch, score in candidates:
                if get_char_type(ch) == expected_type and score > best_score_here:
                    best_score_here = score
                    best_char_here = ch
            if best_char_here is None:
                valid = False
                break
            chosen_chars.append(best_char_here)
            chosen_scores.append(best_score_here)
            total_score += best_score_here
        if valid and total_score > best_score:
            best_score = total_score
            best_chars = chosen_chars
            best_indices = list(indices)
            best_char_scores = chosen_scores
    if best_chars is not None:
        return best_chars, best_indices, best_char_scores
    fallback_chars = []
    fallback_indices = list(range(min(n, m)))
    fallback_scores = []
    for i in fallback_indices:
        if position_candidates[i]:
            fallback_chars.append(position_candidates[i][0][0])
            fallback_scores.append(position_candidates[i][0][1])
    return fallback_chars, fallback_indices, fallback_scores

def is_plate_size_valid(box):
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h
    if w < MIN_PLATE_W or h < MIN_PLATE_H:
        return False
    if w > MAX_PLATE_W or h > MAX_PLATE_H:
        return False
    if area < MIN_PLATE_AREA or area > MAX_PLATE_AREA:
        return False
    return True

def detect_plates_on_frame(frame):
    results = yolo_model(frame)
    plates = []
    for r in results:
        boxes = r.boxes.xyxy
        confs = r.boxes.conf if hasattr(r.boxes, "conf") and r.boxes.conf is not None else None
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.tolist())
            plate_img = frame[y1:y2, x1:x2]
            det_conf = float(confs[idx].item()) if confs is not None else 0.0
            plates.append((plate_img, (x1, y1, x2, y2), det_conf))
    return plates

def recognize_plate_from_crop(plate_img):
    if plate_img is None or plate_img.size == 0:
        return "", 0.0, []
    plate = align_plate(plate_img)
    chars = segment_characters(plate)
    if not chars:
        return "", 0.0, []
    position_candidates = []
    for ch in chars:
        ch = normalize_char(ch)
        ch_rgb = cv2.cvtColor(ch, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(ch_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = char_model(tensor)
            probs = torch.softmax(pred, dim=1)
            topk_probs, topk_idxs = torch.topk(probs, k=min(5, len(classes)), dim=1)
        topk_chars = [classes[i] for i in topk_idxs[0].cpu().numpy()]
        topk_scores = topk_probs[0].cpu().numpy().tolist()
        candidates = build_position_candidates(topk_chars, topk_scores)
        position_candidates.append(candidates)
    filtered, kept_indices, filtered_scores = apply_mask_with_corrections(position_candidates, MASK)
    #filtered, filtered_scores = apply_special_plate_rules(filtered, filtered_scores)
    plate_text = "".join(filtered)
    if not filtered_scores:
        return plate_text, 0.0, []
    mean_conf = float(np.mean(filtered_scores))
    return plate_text, mean_conf, filtered_scores

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance_between_boxes(box1, box2):
    c1 = box_center(box1)
    c2 = box_center(box2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def choose_final_result(results):
    valid_results = [r for r in results if r["text"]]
    if not valid_results:
        return "", 0.0
    texts = [r["text"] for r in valid_results]
    counts = Counter(texts)
    most_common_text, count = counts.most_common(1)[0]
    if count >= 2:
        same_text_results = [r for r in valid_results if r["text"] == most_common_text]
        best_same = max(same_text_results, key=lambda x: x["conf"])
        return best_same["text"], best_same["conf"]
    best_result = max(valid_results, key=lambda x: x["conf"])
    return best_result["text"], best_result["conf"]

def create_result_window(last_plate, last_conf):
    img = np.zeros((180, 600, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "Final plate:",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    plate_text = last_plate if last_plate else "---"
    cv2.putText(
        img,
        plate_text,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )
    conf_text = f"Confidence: {last_conf:.2f}" if last_plate else "Confidence: ---"
    cv2.putText(
        img,
        conf_text,
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA
    )
    return img

class PlateTrack:
    def __init__(self, box):
        self.box = box
        self.last_seen = time.monotonic()
        self.last_read_time = 0.0
        self.readings = []
        self.final_text = None
        self.final_conf = None
        self.completed = False
    def update_box(self, box):
        self.box = box
        self.last_seen = time.monotonic()
    def should_read(self, now):
        if self.completed:
            return False
        if len(self.readings) >= REQUIRED_READS:
            return False
        if now - self.last_read_time < RECOGNITION_INTERVAL:
            return False
        return True
    def add_reading(self, text, conf, now):
        self.readings.append({
            "text": text,
            "conf": conf,
            "time": now
        })
        self.last_read_time = now
        if len(self.readings) >= REQUIRED_READS:
            self.final_text, self.final_conf = choose_final_result(self.readings)
            self.completed = True

def process_video(source=0, display=True, save_output=False, output_path="output_video.mp4"):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Не удалось открыть видео/камеру.")
        return
    writer = None
    if save_output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    tracks = []
    last_final_plate = ""
    last_final_conf = 0.0
    if display:
        cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(RESULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    paused = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.monotonic()
        detections = detect_plates_on_frame(frame)
        for plate_img, box, det_conf in detections:
            if not is_plate_size_valid(box):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "size filtered",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
                continue
            matched_track = None
            best_dist = float("inf")
            for track in tracks:
                dist = distance_between_boxes(track.box, box)
                if dist < TRACK_DISTANCE_PX and dist < best_dist:
                    matched_track = track
                    best_dist = dist
            if matched_track is None:
                matched_track = PlateTrack(box)
                tracks.append(matched_track)
            else:
                matched_track.update_box(box)
            if matched_track.should_read(now):
                text, conf, char_scores = recognize_plate_from_crop(plate_img)
                matched_track.add_reading(text, conf, now)
                if matched_track.completed and matched_track.final_text:
                    last_final_plate = matched_track.final_text
                    last_final_conf = matched_track.final_conf
                    print(f"Итоговый номер: {last_final_plate} | confidence={last_final_conf:.2f}")
        tracks = [t for t in tracks if (now - t.last_seen) < 2.0]
        for track in tracks:
            x1, y1, x2, y2 = track.box
            if track.completed and track.final_text:
                color = (0, 255, 0)
                label = f"{track.final_text} | {track.final_conf:.2f}"
            else:
                color = (0, 255, 255)
                last_text = track.readings[-1]["text"] if track.readings else "reading..."
                label = f"{last_text} ({len(track.readings)}/{REQUIRED_READS})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )
        result_img = create_result_window(last_final_plate, last_final_conf)
        if display:
            cv2.imshow(VIDEO_WINDOW_NAME, frame)
            cv2.imshow(RESULT_WINDOW_NAME, result_img)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1) & 0xFF
        # выход
        if key == 27 or key == ord("q") or key == ord("Q"):
            break
        # пауза по пробелу
        if key == 32:  # SPACE
            paused = not paused
        # режим паузы
        while paused:
            pause_key = cv2.waitKey(0) & 0xFF
            # снова пробел — продолжить
            if pause_key == 32:
                paused = False
            # выход из программы
            elif pause_key == 27 or pause_key == ord("q") or pause_key == ord("Q"):
                paused = False
                break
        if display:
            video_visible = cv2.getWindowProperty(VIDEO_WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            result_visible = cv2.getWindowProperty(RESULT_WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            if video_visible < 1 or result_visible < 1:
                break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(source="D:\\Users\\Name\\Downloads\\video_2 (online-video-cutter.com).mp4", display=True, save_output=False)
    #process_video(source=0, display=True, save_output=False)