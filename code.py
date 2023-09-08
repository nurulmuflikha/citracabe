import cv2
import numpy as np

# Gambar belum di resize
citraRGB = cv2.imread('cabe55.jpg')

# Buka gambar
img = cv2.imread('cabe55.jpg')

# Resize gambar
img_resized = cv2.resize(img, (400, 500))  # Ubah ukuran gambar menjadi 400x500

citra_asli = img_resized.copy()

# Konversi gambar ke hsv
hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

# Nilai hsv warna merah
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

# Kombinasi mask
mask = cv2.bitwise_or(mask1, mask2)

# Operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)

# Hitung nilai rata-rata RGB dari area yang ter-mask
mean_color_bgr = cv2.mean(img_masked, mask=mask)[:3]

# Konversi nilai rata-rata RGB ke HSV
mean_color_hsv = cv2.cvtColor(np.uint8([[mean_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

# Contour
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Bounding box deteksi objek
jumlah_cabe = 0
total_area = 0
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 10 and h > 30:  # Hanya gambar kotak jika ukurannya cukup besar
        jumlah_cabe += 1
        cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area = cv2.contourArea(contour)
        total_area += area
        
print("Jumlah cabe: ", jumlah_cabe)
print("Nilai RGB dari area yang ter-mask: ", mean_color_bgr)
print("Nilai HSV dari area yang ter-mask: ", mean_color_hsv)

cv2.putText(img_resized, "Jumlah cabe terdeteksi: " + str(jumlah_cabe), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(img_resized, "Luas total area cabe: " + str(total_area), (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(img_masked, "Jumlah cabe terdeteksi: " + str(jumlah_cabe), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Menampilkan gambar asli yang belum di resize
# cv2.imshow('Gambar Asli', citraRGB)

cv2.imshow('Gambar Asli Resized', citra_asli)
cv2.imshow('Gambar Termask', img_masked)
cv2.imshow('Hasil deteksi', img_resized)
cv2.imshow('hsv', hsv)
cv2.imshow('mask1', mask1)
cv2.imshow('mask2', mask2)
cv2.imshow('masking gabungan', mask)
cv2.imshow('operasi morphologi', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
