import cv2
from detect_cor import detect_plate

# Путь к тестовому изображению
image_path = "D:\\vaz-lada_2110__634859402bx.jpg"
plates = detect_plate(image_path)
if not plates:
    print("Номера не обнаружены!")
else:
    for i, plate in enumerate(plates):
        cv2.imshow(f"Plate {i+1}", plate)
        cv2.imwrite(f"test_images/plate_{i+1}.jpg", plate)
cv2.waitKey(0)
cv2.destroyAllWindows()