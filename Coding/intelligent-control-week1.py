import cv2
import numpy as np

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---------------------- Warna Merah ----------------------
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = mask_red1 + mask_red2

    # ---------------------- Warna Hijau ----------------------
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # ---------------------- Warna Biru ----------------------
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Gabungkan semua mask
    masks = [("Merah", mask_red, (0, 0, 255)),
             ("Hijau", mask_green, (0, 255, 0)),
             ("Biru", mask_blue, (255, 0, 0))]

    for color_name, mask, box_color in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:  # hanya objek besar
                # Cek bentuk kontur
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if len(approx) == 4:  # hanya kotak/persegi panjang
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
                    cv2.putText(frame, f"{color_name} (Kotak)", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    cv2.imshow("Deteksi Warna Kotak", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
