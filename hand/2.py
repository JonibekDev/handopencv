import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from enum import Enum
import pyautogui
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
hands_haar_cascade = cv2.CascadeClassifier(IMAGE_FILES)

class Moves(Enum):
    LEFT = 0
    TOP = 1
    RIGHT = 2
    DOWN = 3

moves_queue = deque([0]*4, maxlen=4)
last_vertical, last_horizontal = 0, 0

while True:
    _r, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hands_haar_cascade.detectMultiScale(gray, 1.1, 3)

    if len(hands):
        for x, y, w, h in hands:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

            if last_vertical or last_horizontal:
                diff_vertical = y - last_vertical
                diff_horizontal = x - last_horizontal

                if abs(diff_vertical) > abs(diff_horizontal):
                    if diff_vertical < 0:
                        moves_queue.appendleft(Moves.TOP)
                    else:
                        moves_queue.appendleft(Moves.DOWN)
                else:
                    if diff_horizontal > 0:
                        moves_queue.appendleft(Moves.LEFT)
                    else:
                        moves_queue.appendleft(Moves.RIGHT)

            last_vertical = y
            last_horizontal = x

            if all(m == Moves.LEFT for m in moves_queue):
                print("Left")
                # pyautogui.hotkey('alt', 'f4')
            elif all(m == Moves.TOP for m in moves_queue):
                print('Up')
                # pyautogui.keyUp('up')
            elif all(m == Moves.RIGHT for m in moves_queue):
                print('Right')
               # pyautogui.press('q')
            elif all(m == Moves.DOWN for m in moves_queue):
                print('Down')
                # pyautogui.keyDown('down')

    cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break