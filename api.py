from django.http import JsonResponse
import json
import base64
import cv2
import pickle


def image(request):
    if request.method == "POST":
        data = json.loads(request.body)
        f = open('uname2auth.json')
        uname2auth = json.load(f)
        f.close()

        image1 = base64.b64decode(data['imgbase64'])

        with open('image.png', 'wb') as f:
            f.write(image1)
        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read("trainer.yml")

        labels = {}
        temp_labels = {}
        with open("labels.dat", 'rb') as f:
            temp_labels = pickle.load(f)
            labels = {v: k for k, v in temp_labels.items()}
        # conv to grayscale
        image = cv2.imread('image.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        name = None
        conf = None

        if len(faces) != 0:
            x, y, w, h = faces[0]

            roi = gray[y: y + h, x: x + w]

            roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_LINEAR)
            roi = cv2.equalizeHist(roi)

            ind, conf = recognizer.predict(roi)

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

        if name == data['person']:
            #load name2auth
            return JsonResponse(uname2auth[name], safe=False)
        else:
            return JsonResponse(uname2auth['unauthorized'], safe=False)

    else:
        return JsonResponse({'get method': 'invalid'}, safe=False)
