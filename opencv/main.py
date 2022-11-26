import cv2
import numpy as np
from collections import deque
from enum import Enum
import pyautogui

video = cv2.VideoCapture(0)
hands_haar_cascade = cv2.CascadeClassifier("rpalm.xml")

class Moves(Enum):
    LEFT = 0
    TOP = 1
    RIGHT = 2
    DOWN = 3

moves_queue = deque([0]*4, maxlen=4)
last_vertical, last_horizontal = 0, 0

while True:
    _r, frame = video.read()
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
                pyautogui.hotkey("end")
            elif all(m == Moves.TOP for m in moves_queue):
                pyautogui.keyUp('up')
            elif all(m == Moves.RIGHT for m in moves_queue):
               pyautogui.press('home')
            elif all(m == Moves.DOWN for m in moves_queue):
                pyautogui.keyDown('down')

    cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
