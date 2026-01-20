import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

# 1. MediaPipe El Takibi Ayarları (Yeni API - 0.10.x)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model dosyasını indir (ilk çalıştırmada gerekli)
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Model indiriliyor...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model indirildi!")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

hand_landmarker = HandLandmarker.create_from_options(options)

# 2. Hazırlanan Kareleri Hafızaya Yükle
frame_folder = "frames"
if not os.path.exists(frame_folder) or len(os.listdir(frame_folder)) == 0:
    print("Hata: 'frames' klasörü boş! Önce hazirlik.py kodunu çalıştırın.")
    exit()

frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])
flower_images = [cv2.imread(os.path.join(frame_folder, f), cv2.IMREAD_UNCHANGED) for f in frame_files]

def overlay_png(bg, img, pos):
    """Şeffaf PNG'yi arka plana yerleştiren yardımcı fonksiyon."""
    x, y = pos
    h, w = img.shape[:2]
    
    # Ekran sınır kontrolü
    if y < 0 or y + h > bg.shape[0] or x < 0 or x + w > bg.shape[1]:
        return bg

    # Alpha kanalı işlemleri (Şeffaflık için)
    overlay_rgb = img[:, :, :3]
    alpha = img[:, :, 3] / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    # Arka planla görseli birleştir
    bg_section = bg[y:y+h, x:x+w]
    composite = (bg_section * (1 - alpha) + overlay_rgb * alpha).astype(np.uint8)
    bg[y:y+h, x:x+w] = composite
    return bg

# 3. Kamera Başlatma
cap = cv2.VideoCapture(0)
smooth_idx = 0.0

print("Kamera açılıyor... Kapatmak için 'q' tuşuna basın.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1) # Ayna görüntüsü
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Yeni API ile el tespiti
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = hand_landmarker.detect(mp_image)

    if results.hand_landmarks:
        for hand_lms in results.hand_landmarks:
            h_img, w_img, _ = frame.shape
            # Çiçeğin merkezi (Orta parmak kökü - nokta 9)
            cx = int(hand_lms[9].x * w_img)
            cy = int(hand_lms[9].y * h_img)

            # El açıklığını hesapla - Parmak uçları ile bilek arasındaki mesafe
            # Bilek (0), Parmak uçları: İşaret(8), Orta(12), Yüzük(16), Serçe(20)
            wrist = np.array([hand_lms[0].x, hand_lms[0].y])
            
            # Tüm parmak uçlarının bilekten uzaklığının ortalaması
            finger_tips = [8, 12, 16, 20]
            total_dist = 0
            for tip_idx in finger_tips:
                tip = np.array([hand_lms[tip_idx].x, hand_lms[tip_idx].y])
                total_dist += np.linalg.norm(tip - wrist)
            avg_dist = total_dist / len(finger_tips)

            # Mesafeyi 0-1 arasına çek (Kalibrasyon - geniş aralık)
            # El kapalı (~0.15) -> yaprak kapalı, El tam açık (~0.50) -> yaprak tam açık (bloom)
            target_ratio = np.clip((avg_dist - 0.15) / 0.35, 0, 1)
            
            # Debug: Display hand openness on screen (optional, for testing)
            cv2.putText(frame, f"Openness: {target_ratio:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # El hızını hesapla (önceki frame ile karşılaştır)
            target_frame = target_ratio * (len(flower_images) - 1)
            frame_diff = abs(target_frame - smooth_idx)
            
            # Hız adaptasyonu: hızlı hareket = hızlı animasyon, yavaş hareket = yavaş animasyon
            # Minimum 0.15, maksimum 0.8 yumuşatma faktörü
            speed_factor = np.clip(frame_diff / 10, 0.15, 0.8)
            
            # Animasyon geçişini el hızına göre yumuşat
            smooth_idx += (target_frame - smooth_idx) * speed_factor
            frame_index = int(np.clip(smooth_idx, 0, len(flower_images) - 1))
            current_frame = flower_images[frame_index]

            # Çiçek boyutu (300 piksel idealdir, istersen değiştirebilirsin)
            side = 300 
            resized_flower = cv2.resize(current_frame, (side, side))

            # Çiçeği elin üzerine yerleştir
            frame = overlay_png(frame, resized_flower, (cx - side//2, cy - side//2))

    cv2.imshow("Cicek Animasyonu", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
hand_landmarker.close()
cv2.destroyAllWindows()