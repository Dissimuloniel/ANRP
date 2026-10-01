import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from itertools import combinations
import torch
import numpy as np
from detect_cor import detect_plate_cor
from segmantation import segment_characters
from recognize import get_vgg19, classes, transform, device
from rotate import align_plate

#загрузка модели
model = get_vgg19()
model.load_state_dict(torch.load("vgg19_chars_16.pth", map_location=device))
model.to(device)
model.eval()
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

def build_position_candidates(topk_chars, topk_scores, classes, similar_penalty=0.08):
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
    result = sorted(cand.items(), key=lambda x: x[1], reverse=True)
    return result

def normalize_char(char_img, out_size=(40, 40)):
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(255 - gray)
    if coords is None:
        coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        gray = gray[y:y+h, x:x+w]
    h, w = gray.shape
    if h == 0 or w == 0:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    scale = min((out_size[0] - 8) / w, (out_size[1] - 8) / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)

    canvas = np.full(out_size, 255, dtype=np.uint8)
    x0 = (out_size[0] - nw) // 2
    y0 = (out_size[1] - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

def get_char_type(ch):
        if ch.isdigit():
            return "2"
        else:
            return "1"
        
def apply_mask(predicted_chars, confidences):
    n = len(predicted_chars)
    m = len(MASK)
    if n == 0:
        return [], []
    if n <= m:
        return predicted_chars, list(range(n))
    best_score = -float("inf")
    best_seq = None
    best_indices = None
    from itertools import combinations
    for indices in combinations(range(n), m):
        seq = [predicted_chars[i] for i in indices]
        confs = [confidences[i] for i in indices]
        ok = True
        score = 0.0
        for ch, conf, expected_type in zip(seq, confs, MASK):
            if get_char_type(ch) != expected_type:
                ok = False
                break
            score += conf
        if ok and score > best_score:
            best_score = score
            best_seq = seq
            best_indices = list(indices)
    if best_seq is not None:
        return best_seq, best_indices
    return predicted_chars[:m], list(range(min(n, m)))

'''def apply_mask_with_corrections(position_candidates, mask):
    n = len(position_candidates)
    m = len(mask)
    if n == 0:
        return [], []
    if n < m:
        result = []
        indices = []
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
        return result, indices
    best_score = -float("inf")
    best_chars = None
    best_indices = None
    for indices in combinations(range(n), m):
        total_score = 0.0
        chosen_chars = []
        valid = True
        for pos_in_mask, idx in enumerate(indices):
            expected_type = mask[pos_in_mask]
            candidates = position_candidates[idx]
            best_char_here = None
            best_score_here = -1.0
            for ch, score in candidates:
                if get_char_type(ch) == expected_type:
                    if score > best_score_here:
                        best_score_here = score
                        best_char_here = ch
            if best_char_here is None:
                valid = False
                break
            chosen_chars.append(best_char_here)
            total_score += best_score_here
        if valid and total_score > best_score:
            best_score = total_score
            best_chars = chosen_chars
            best_indices = list(indices)
    if best_chars is not None:
        return best_chars, best_indices
    fallback_chars = []
    fallback_indices = list(range(min(n, m)))
    for i in fallback_indices:
        if position_candidates[i]:
            fallback_chars.append(position_candidates[i][0][0])
    return fallback_chars, fallback_indices'''

def apply_mask_with_corrections(position_candidates, mask):
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
                if get_char_type(ch) == expected_type:
                    if score > best_score_here:
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

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание номеров")
        self.canvas = tk.Canvas(root, width=850, height=600)
        self.canvas.pack(pady=10)
        self.btn = tk.Button(root, text="Загрузить изображение", command=self.load_image)
        self.btn.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        img = cv2.imread(file_path)
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plates = detect_plate_cor(file_path)
        if not plates:
            print("Номер не найден")
            return
        plate, (x1, y1, x2, y2) = plates[0]
        display_img = img.copy()
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plate = align_plate(plate)
        chars = segment_characters(plate)
        char_images = []
        predicted_chars = []
        confidences = []
        '''#os.makedirs("debug_chars", exist_ok=True)
        for idx, ch in enumerate(chars):
            ch = normalize_char(ch)

            # сохраняем для отладки
            #cv2.imwrite(f"debug_chars/char_{idx}.png", ch)

            ch_rgb = cv2.cvtColor(ch, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(ch_rgb)
            tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tensor)
                probs = torch.softmax(pred, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

            predicted_chars.append(classes[pred_idx.item()])
            confidences.append(conf.item())
            char_images.append(ch_rgb)'''
        position_candidates = []
        for idx, ch in enumerate(chars):
            ch = normalize_char(ch)
            ch_rgb = cv2.cvtColor(ch, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(ch_rgb)
            tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor)
                probs = torch.softmax(pred, dim=1)
                topk_probs, topk_idxs = torch.topk(probs, k=min(5, len(classes)), dim=1)
            topk_chars = [classes[i] for i in topk_idxs[0].cpu().numpy()]
            topk_scores = topk_probs[0].cpu().numpy().tolist()
            candidates = build_position_candidates(topk_chars, topk_scores, classes)
            position_candidates.append(candidates)
            predicted_chars.append(topk_chars[0])
            confidences.append(topk_scores[0])
            char_images.append(ch_rgb)
        #применяем маску
        filtered, kept_indices, filtered_scores = apply_mask_with_corrections(position_candidates, MASK)
        #filtered, kept_indices = apply_mask_with_corrections(position_candidates, MASK)
        #filtered, kept_indices = apply_mask(predicted_chars, confidences)
        plate_text = "".join(filtered)
        filtered_char_images = [char_images[i] for i in kept_indices] if kept_indices else char_images
        self.show_images(display_img, plate, filtered_char_images, plate_text, filtered_scores)
        #self.show_images(display_img, plate, filtered_char_images, plate_text)
        #self.show_images(display_img, plate, char_images, plate_text, filtered_scores)

    def show_images(self, original_img, plate_img, chars, plate_text, char_scores):
    #def show_images(self, original_img, plate_img, chars, plate_text):
        #настройки
        left_w = 500
        right_w = 300
        gap_between_blocks = 20     # промежутки внутри правой колонки
        border = 40                 # внешний край
        gap_between_columns = 20    # расстояние между левым и правым изображениями
        frame_thickness = 5         # толщина рамки вокруг колонок
        #ЛЕВАЯ ЧАСТЬ
        left_img = cv2.resize(original_img, (left_w, 400))
        left_img = np.pad(left_img, ((border, border), (border, border), (0,0)), mode='constant')
        #добавляем рамку
        left_img[:frame_thickness, :, :] = 0
        left_img[-frame_thickness:, :, :] = 0
        left_img[:, :frame_thickness, :] = 0
        left_img[:, -frame_thickness:, :] = 0
        #ПРАВАЯ ЧАСТЬ
        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        plate_resized = cv2.resize(plate_rgb, (right_w, 100))
        char_imgs = []
        for ch in chars:
            ratio = 60 / ch.shape[0]
            w = int(ch.shape[1] * ratio)
            ch_resized = cv2.resize(ch, (w, 60))
            char_imgs.append(ch_resized)
        if char_imgs:
            char_row = np.hstack(char_imgs)
            #центрируем по ширине правого блока
            if char_row.shape[1] < right_w:
                pad_left = (right_w - char_row.shape[1]) // 2
                pad_right = right_w - char_row.shape[1] - pad_left
                char_row = np.pad(char_row, ((0,0),(pad_left,pad_right),(0,0)), mode='constant')
            elif char_row.shape[1] > right_w:
                char_row = cv2.resize(char_row, (right_w, char_row.shape[0]))
        else:
            char_row = np.zeros((60, right_w, 3), dtype=np.uint8)
        '''#текст номера
        text_img = np.zeros((60, right_w, 3), dtype=np.uint8)
        cv2.putText(text_img, plate_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        #объединяем правую колонку с промежутками
        right_col = np.vstack([
            plate_resized,
            np.zeros((gap_between_blocks, right_w, 3), dtype=np.uint8),
            char_row,
            np.zeros((gap_between_blocks, right_w, 3), dtype=np.uint8),
            text_img
        ])'''
        # текст номера
        text_img = np.zeros((70, right_w, 3), dtype=np.uint8)
        cv2.putText(text_img, plate_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        conf_lines = [f"{ch}: {score:.2f}" for ch, score in zip(plate_text, char_scores)]

        conf_img = np.zeros((25 * len(conf_lines) + 20, right_w, 3), dtype=np.uint8)

        for i, line in enumerate(conf_lines):
            y = 25 + i * 25
            cv2.putText(conf_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # объединяем правую колонку с промежутками
        '''right_col = np.vstack([
            plate_resized,
            np.zeros((gap_between_blocks, right_w, 3), dtype=np.uint8),
            char_row,
            np.zeros((gap_between_blocks, right_w, 3), dtype=np.uint8),
            text_img,
            np.zeros((10, right_w, 3), dtype=np.uint8),
            conf_img
        ])'''
        right_col = np.vstack([
            plate_resized,
            np.zeros((gap_between_blocks, right_w, 3), dtype=np.uint8),
            char_row,
            np.zeros((gap_between_blocks, right_w, 3), dtype=np.uint8),
            text_img,
            np.zeros((10, right_w, 3), dtype=np.uint8),
            conf_img
        ])
        #Срез

        right_col = np.pad(right_col, ((border,border),(border,border),(0,0)), mode='constant')
        #добавляем рамку
        right_col[:frame_thickness, :, :] = 0
        right_col[-frame_thickness:, :, :] = 0
        right_col[:, :frame_thickness, :] = 0
        right_col[:, -frame_thickness:, :] = 0

        #выравнивание по высоте
        if left_img.shape[0] < right_col.shape[0]:
            pad_h = right_col.shape[0] - left_img.shape[0]
            left_img = np.pad(left_img, ((0,pad_h),(0,0),(0,0)), mode='constant')
        else:
            pad_h = left_img.shape[0] - right_col.shape[0]
            right_col = np.pad(right_col, ((0,pad_h),(0,0),(0,0)), mode='constant')
        #объединяем левый и правый блоки с небольшим промежутком
        canvas_img = np.hstack([left_img, np.zeros((left_img.shape[0], gap_between_columns,3), dtype=np.uint8), right_col])
        #отображение
        pil_img = Image.fromarray(canvas_img)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.canvas.config(width=canvas_img.shape[1], height=canvas_img.shape[0])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()