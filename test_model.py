import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

signals = [
    'A',
    'E',
    'I',
    'O',
    'U'
]

while True:
  ret, frame = cap.read()
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  results = hands.process(frame_rgb)
  coordinates = []

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          frame,  # image to draw
          hand_landmarks,  # model output
          mp_hands.HAND_CONNECTIONS)

      for landmark in hand_landmarks.landmark:
        coordinates.append([landmark.x, landmark.y])

    prediction = model.predict([coordinates])

    print('======================')
    print(signals[np.argmax(prediction, axis=1)[0]])
    print('======================')

  cv2.imshow('frame', frame)
  if cv2.waitKey(25) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
