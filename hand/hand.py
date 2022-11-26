import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
from enum import Enum
import pyautogui

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 4:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    hands_haar_cascade = results.multi_hand_landmarks


    class Moves(Enum):
        LEFT = 0
        TOP = 1
        RIGHT = 2
        DOWN = 3


    moves_queue = deque([0] * 4, maxlen=4)
    last_vertical, last_horizontal = 0, 0

    while True:
        _r, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hands = hands_haar_cascade.results.multi_hand_landmarks

        if len(hands):
            for x, y, w, h in hands:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

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

    cv2.imshow("Image", img)
    cv2.waitKey(1)
