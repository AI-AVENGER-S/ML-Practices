import cv2
import dlib
import numpy as np
import os

def create_face_skin_mask(image_path, output_path, predictor_path):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("Лица на изображении не обнаружены.")
        return

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

        jawline = points[0:17]
        eyebrows = points[17:27][::-1]
        face_contour = np.concatenate((jawline, eyebrows), axis=0)
        
        cv2.fillPoly(mask, [face_contour], 255)

        left_eye = points[36:42]
        right_eye = points[42:48]
        mouth = points[48:60]
        cv2.fillPoly(mask, [left_eye], 0)
        cv2.fillPoly(mask, [right_eye], 0)
        cv2.fillPoly(mask, [mouth], 0)

    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite(output_path, result)
    print(f"Успешно! Результат сохранен в: {output_path}")

if __name__ == "__main__":
    INPUT_IMAGE = "input.jpg"
    OUTPUT_IMAGE = "output_mask.jpg"
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

    create_face_skin_mask(INPUT_IMAGE, OUTPUT_IMAGE, PREDICTOR_PATH)