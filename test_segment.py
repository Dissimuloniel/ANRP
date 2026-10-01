import cv2
from segmantation import segment_characters

IMAGE_PATH = "D:\\Shlagbaum\\aligned_plate.png"

from detect import detect_plate

def test_segmentation(image_path):
    plates = detect_plate(image_path)

    if not plates:
        print("Номер не найден")
        return

    for p_idx, plate in enumerate(plates):
        cv2.imshow(f"Plate_{p_idx}", plate)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 3
        )
        cv2.imshow(f"Thresh_{p_idx}", thresh)
        chars = segment_characters(plate)
        print(f"Найдено символов: {len(chars)}")
        debug_img = plate.copy()
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        h_img, w_img = plate.shape[:2]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (h > h_img * 0.5 and
                h < h_img * 0.95 and
                w > w_img * 0.02 and
                w < w_img * 0.2):
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow(f"Boxes_{p_idx}", debug_img)
        for i, ch in enumerate(chars):
            cv2.imshow(f"char_{p_idx}_{i}", ch)
            cv2.imwrite(f"debug_char_{p_idx}_{i}.jpg", ch)
    print("Нажмите любую клавишу...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_segmentation(IMAGE_PATH)