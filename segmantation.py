import cv2
import numpy as np

def preprocess_plate(plate_img):
    # Нормализуем размер номера
    target_w, target_h = 240, 80
    plate = cv2.resize(plate_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8
    )
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    return plate, thresh

def segment_characters(plate_img, debug=True):
    pad = 3
    plate, thresh = preprocess_plate(plate_img)
    h_img, w_img = thresh.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    candidates = []
    for i in range(1, num_labels):  # 0 = background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if w <= 0 or h <= 0:
            continue
        aspect = w / float(h)
        rel_h = h / float(h_img)
        rel_w = w / float(w_img)
        rel_area = area / float(h_img * w_img)
        if rel_area < 0.003 or rel_area > 0.15:
            continue
        if rel_h < 0.35 or rel_h > 0.95:
            continue
        if rel_w < 0.02 or rel_w > 0.22:
            continue
        if aspect < 0.15 or aspect > 1.0:
            continue
        if y > h_img * 0.75 or (y + h) < h_img * 0.25:
            continue
        x_start = max(x - pad, 0)
        y_start = max(y - pad, 0)
        x_end = min(x + w + pad, w_img)
        y_end = min(y + h + pad, h_img)
        char = plate[y_start:y_end, x_start:x_end]
        candidates.append((x, y, w, h, char))
    candidates = sorted(candidates, key=lambda item: item[0])
    if candidates:
        heights = np.array([c[3] for c in candidates], dtype=np.float32)
        median_h = np.median(heights)
        filtered = []
        for item in candidates:
            _, _, _, h, _ = item
            if 0.65 * median_h <= h <= 1.35 * median_h:
                filtered.append(item)
        candidates = filtered
    chars = [c[4] for c in candidates]
    if debug:
        dbg = plate.copy()
        for x, y, w, h, _ in candidates:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imwrite("segm_thresh.png", thresh)
        cv2.imwrite("segm_boxes.png", dbg)
    return chars