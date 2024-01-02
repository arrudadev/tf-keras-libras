import os
import shutil
import cv2
import pickle
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if os.path.exists('./data'):
  shutil.rmtree('./data')

if not os.path.exists('./data'):
  os.makedirs('./data')

dataset_size = 100
signals = [
    'A',
    'B',
    'C',
    'D',
    'E'
]

print(f'Collecting images')

for index, signal in enumerate(signals):
  if not os.path.exists(os.path.join('./data', str(index))):
    os.makedirs(os.path.join('./data', str(index)))

  print(f'Collecting the image from the signal of {signal}')

  done = False
  while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
      for hand in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Press "Q" to start and make the signal of "{signal}"', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) == ord('q'):
      break

  time.sleep(1)

  counter = 0
  while counter < dataset_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join('./data', str(
        index), '{}.jpg'.format(counter)), frame)

    counter += 1

cap.release()
cv2.destroyAllWindows()

print(f'Collecting landmarks')

data = []
labels = []
for dir_ in os.listdir('./data'):
  for img_path in os.listdir(os.path.join('./data', dir_)):
    data_aux = []

    x_ = []
    y_ = []

    img = cv2.imread(os.path.join('./data', dir_, img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
          x = hand_landmarks.landmark[i].x
          y = hand_landmarks.landmark[i].y

          x_.append(x)
          y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
          x = hand_landmarks.landmark[i].x
          y = hand_landmarks.landmark[i].y
          data_aux.append(x - min(x_))
          data_aux.append(y - min(y_))

      data.append(data_aux)
      labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
