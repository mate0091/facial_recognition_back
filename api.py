from django.http import JsonResponse
import json
import base64
import cv2
import pickle


def image(request):
    if request.method == "POST":
        data = request.body

        image1 = base64.decodebytes(data)

        with open('image.png', 'wb') as f:
            f.write(image1)
        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read("trainer.yml")

        labels = {}
        temp_labels = {}
        with open("labels.dat", 'rb') as f:
            temp_labels = pickle.load(f)
            labels = {v: k for k, v in temp_labels.items()}
        #conv to grayscale
        image = cv2.imread('image.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        if len(faces) != 0:
            x, y, w, h = faces[0]

            roi = gray[y: y + h, x: x + w]

            roi = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_LINEAR)

            ind, conf = recognizer.predict(roi)

            print(labels[ind], conf)

            if 40 <= conf <= 65:
                #print(labels[ind])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[ind]
                print(name)
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(image, name, (x, y - 25), font, 1, color, stroke, cv2.LINE_AA)

            color = (255, 255, 255)
            stroke = 2

            cv2.rectangle(image, (x, y), (x + w, y + h), color, stroke)

        cv2.imwrite('webcam_detected.png', image)


        return JsonResponse({'post method': 'successful'}, safe=False)
    else:
        return JsonResponse({'get method': 'invalid'}, safe=False)