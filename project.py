# memanggil library yang dibutuhkan
import os
import cv2 # memanggil library opencv
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
# memanggil fungsi img_to_array dari keras
from keras.utils import img_to_array
from keras.models import load_model

# from keras.preprocessing.image import load_img, img_to_array 
import matplotlib.pyplot as plt

# Model yang di save tadi di load kembali dengan memanggil fungsi load_model()
model = load_model("best_model_last.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # untuk mendeteksi wajah bagian depan

cap = cv2.VideoCapture(0) # digunakan untuk mengambil gambar kita dari webcam

while True: # Kita menggunakan while True agar program berjalan terus menerus
    ret, test_img = cap.read() # untuk membaca input stream dari webcam
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) # fungsi untuk melakukan konversi color space

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 204, 0), thickness=7) # fungsi Rectangle untuk membuat objek berupa persegi 4
        roi_gray = gray_img[y:y + w, x:x + h]  # memotong wilayah yang diinginkan yaitu area wajah dari gambar
        roi_gray = cv2.resize(roi_gray, (224, 224)) #224
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # temukan array indeks maksimum
        max_index = np.argmax(predictions[0])

        # emotions = ['Marah','Jijik','Takut','Senang','Biasa','Sedih','Terkejut']
        emotions = ['Bohong','Jujur','Bohong','Jujur','Biasa', 'Jujur', 'Bohong']

        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # cv2.putText(img, teks, (x,y), font, font_size, (B,G,R), tebal, type_line)

    resized_img = cv2.resize(test_img, (900, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):  # tunggu sampai tombol 'q' ditekan
        break

cap.release() # untuk menghentikan stream pada webcam
cv2.destroyAllWindows # untuk menutup seluruh tampilan hasil stream dari webcam