import os, cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'faces')

y_labels = []
x_train = []
current_id = 0
index = 0
label_ids = {}

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

print(IMG_DIR)

for root, dirs, files in os.walk(IMG_DIR):
    for file in files:
        if file.endswith('jpg') or file.endswith('png'):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-').lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            index = label_ids[label]

            pil_img = Image.open(path).convert('L')
            gray = np.array(pil_img, 'uint8')

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) != 0:
                x, y, w, h = faces[0]

                roi = gray[y: y + h, x: x + w]
                roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_LINEAR)

                roi = cv2.equalizeHist(roi)

                x_train.append(roi)
                y_labels.append(index)

with open("labels.dat", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")