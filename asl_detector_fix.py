import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
from PIL import ImageFont, ImageDraw, Image, ImageTk
import tkinter as tk
from tkinter import ttk
import threading

# Inisialisasi pygame untuk pemutaran audio
pygame.mixer.init()

# Load model dan data lainnya
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

# Load the face detection cascade for overlay
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parameters for overlay
mf = 0.7
m = 40
n = m // 2
p = 80

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels for sign language
labels_normal = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
labels_fruits = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
labels_animals = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
voice_files_normal = {
    'A': './voice/a.mp3',
    'B': './voice/b.mp3',
    'C': './voice/c.mp3',
    'D': './voice/d.mp3',
    'E': './voice/e.mp3',
}
voice_files_fruits = {
    'A': './voice/af.mp3',
    'B': './voice/bf.mp3',
    'C': './voice/cf.mp3',
    'D': './voice/df.mp3',
    'E': './voice/ef.mp3',
}
voice_files_animals = {
    'A': './voice/aa.mp3',
    'B': './voice/ba.mp3',
    'C': './voice/ca.mp3',
    'D': './voice/da.mp3',
    'E': './voice/ea.mp3',
}

# Initialize start time and last detected gesture
start_time = None
last_detected = None
overlay_display_time = None  # Timer for overlay display
current_overlay_image = None  # Store the current overlay image

# Font setup for displaying detected letters
font_path = "./Poppins-Regular.ttf"
font_poppins = ("Poppins", 12)
font_poppins_pil = ImageFont.truetype(font_path, 40)

labels_dict = labels_normal
voice_files_dict = voice_files_normal

# Play audio on detection in a separate thread to avoid blocking
def play_audio(file_path):
    def _play():
        if os.path.exists(file_path):
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
    threading.Thread(target=_play, daemon=True).start()
    
# Initialize tkinter window for displaying video
root = tk.Tk()
root.title("ASL Detection")

root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
    
# Fungsi untuk mengganti mode
def change_mode(event):
    global labels_dict, voice_files_dict, current_overlay_image

    selected_mode = options_combobox.get()
    if selected_mode == "Normal":
        labels_dict = labels_normal
        voice_files_dict = voice_files_normal
        current_overlay_image = None  # Reset overlay image
        secondary_combobox.pack_forget()  # Sembunyikan combobox kedua
        secondary_label.pack_forget()
    elif selected_mode == "Interactive":
        secondary_label.pack(side="left", padx=20)
        secondary_combobox.pack(side="left")  # Tampilkan combobox kedua
        secondary_combobox.current(0)
        labels_dict = labels_fruits
        voice_files_dict = voice_files_fruits

# Fungsi untuk mengganti kategori dalam mode Interactive
def change_category(event):
    global labels_dict, voice_files_dict

    selected_category = secondary_combobox.get()
    if selected_category == "Fruit":
        labels_dict = labels_fruits
        voice_files_dict = voice_files_fruits
    elif selected_category == "Animal":
        labels_dict = labels_animals
        voice_files_dict = voice_files_animals

# Membuat frame untuk combobox utama dan kedua
options_frame = tk.Frame(root)
options_frame.pack(pady=15, anchor='w', padx=180)

# Label untuk combobox utama
options_label = tk.Label(options_frame, text="Mode: ", font=font_poppins)
options_label.pack(side="left", padx=10)

# Combobox utama
options_combobox = ttk.Combobox(options_frame, values=["Normal", "Interactive"], state="readonly", font=font_poppins)
options_combobox.current(0)
options_combobox.bind("<<ComboboxSelected>>", change_mode)
options_combobox.pack(side="left", padx=10)

# Label untuk combobox kedua
secondary_label = tk.Label(options_frame, text="Category: ", font=font_poppins)
secondary_label.pack(side="left")

# Combobox kedua (untuk kategori dalam mode Interactive)
secondary_combobox = ttk.Combobox(options_frame, values=["Fruit", "Animal"], state="readonly", font=font_poppins)
secondary_combobox.bind("<<ComboboxSelected>>", change_category)
secondary_combobox.pack_forget() 
secondary_label.pack_forget() # Sembunyikan combobox kedua saat awal


label = tk.Label(root)
label.pack()

# Function for overlaying image on face
def overlay_image(face_img, overlay_img, x, y, w, h, scale_factor=0.5, x_offset=30):
    if overlay_img is None or overlay_img.shape[2] != 4:  # Check if overlay image is valid
        return face_img
    
    # Resize overlay image with scale_factor
    overlay_img_resized = cv2.resize(overlay_img, (int(w * scale_factor), int(h * scale_factor)))  
    alpha_mask = overlay_img_resized[:, :, 3] / 255.0  # Alpha channel mask
    
    # Check if the face region has valid dimensions
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return face_img  # Return the image without overlay if invalid dimensions

    y_offset = y - int(1.3 * h * scale_factor)  # Adjust position above head
    if y_offset < 0:
        y_offset = 0

    overlay_height, overlay_width = overlay_img_resized.shape[:2]
    y_start = max(0, y_offset)
    y_end = min(y_offset + overlay_height, face_img.shape[0])
    x_start = max(0, x + x_offset)
    x_end = min(x + x_offset + overlay_width, face_img.shape[1])

    overlay_crop = overlay_img_resized[0:y_end - y_start, 0:x_end - x_start]
    alpha_crop = alpha_mask[0:y_end - y_start, 0:x_end - x_start]

    for c in range(3):
        face_img[y_start:y_end, x_start:x_end, c] = (
            (1 - alpha_crop) * face_img[y_start:y_end, x_start:x_end, c] +
            alpha_crop * overlay_crop[:, :, c]
        )
    
    return face_img

# Update frame for video capture and hand gesture detection
def update_frame():
    global last_detected, start_time, overlay_display_time, current_overlay_image

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Jika ada tangan yang terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        if len(data_aux) < 84:
            data_aux.extend([0] * (84 - len(data_aux)))
        
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Prediksi gesture tangan
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Jika gesture sama seperti sebelumnya
        if predicted_character == last_detected:
            if start_time and time.time() - start_time >= 2.5:  # Gesture stabil selama 2.5 detik
                if overlay_display_time is None:  # Perbarui gambar dan suara hanya sekali
                    overlay_display_time = time.time()
                    play_audio(voice_files_dict.get(predicted_character, ''))
                    
                    selected_mode = options_combobox.get()
                    selected_category = secondary_combobox.get()

                    if selected_mode == "Normal":
                        # Reset gambar overlay untuk mode Normal
                        current_overlay_image = cv2.imread(f'./images/normal/{predicted_character}.png', cv2.IMREAD_UNCHANGED)
                    elif selected_mode == "Interactive":
                        if selected_category == "Fruit":
                            current_overlay_image = cv2.imread(f'./images/fruit/{predicted_character}.png', cv2.IMREAD_UNCHANGED)
                        elif selected_category == "Animal":
                            current_overlay_image = cv2.imread(f'./images/animal/{predicted_character}.png', cv2.IMREAD_UNCHANGED)
           
        else:
            # Gesture berubah
            last_detected = predicted_character
            start_time = time.time()
            overlay_display_time = None  # Reset overlay timer
            
        # Menggambar teks dan border langsung di frame asli
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((x1 + 75, y1 - 50), predicted_character, font=font_poppins_pil, fill=(204, 0, 0))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    else:
        # Jika tidak ada tangan terdeteksi, jangan ubah overlay yang ada
        pass
    
    # Tampilkan gambar overlay terakhir jika ada
    if current_overlay_image is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue
            frame = overlay_image(frame, current_overlay_image, x, y, w, h, scale_factor=0.7)

    # Resize frame untuk ditampilkan
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    aspect_ratio = W / H

    if screen_width / aspect_ratio <= screen_height:
        new_width = screen_width
        new_height = int(screen_width / aspect_ratio)
    else:
        new_height = screen_height
        new_width = int(screen_height * aspect_ratio)

    frame = cv2.resize(frame, (new_width, new_height))

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
pygame.mixer.quit()