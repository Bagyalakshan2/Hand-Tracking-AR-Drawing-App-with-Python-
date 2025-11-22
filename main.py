import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)


canvas = None


draw_color = (0, 0, 255)

prev_x, prev_y = 0, 0


palette = {
    "red": (0, 0, 80, 80),
    "green": (80, 0, 160, 80),
    "blue": (160, 0, 240, 80),
    "erase": (240, 0, 320, 80)
}

def fingers_up(hand_landmarks):
    finger_status = []

    
    thumb_up = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
    finger_status.append(thumb_up)

    
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        finger_status.append(
            hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
        )

    return finger_status  


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    cv2.rectangle(frame, palette["red"][:2], palette["red"][2:], (0, 0, 255), -1)
    cv2.rectangle(frame, palette["green"][:2], palette["green"][2:], (0, 255, 0), -1)
    cv2.rectangle(frame, palette["blue"][:2], palette["blue"][2:], (255, 0, 0), -1)
    cv2.rectangle(frame, palette["erase"][:2], palette["erase"][2:], (255, 255, 255), -1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            status = fingers_up(hand_landmarks)
            thumb, index, middle, _, _ = status

            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            if y < 80:
                if 0 < x < 80:
                    draw_color = (0, 0, 255)
                elif 80 < x < 160:
                    draw_color = (0, 255, 0)
                elif 160 < x < 240:
                    draw_color = (255, 0, 0)
                elif 240 < x < 320:
                    draw_color = (0, 0, 0)  
                continue

            if middle and not index:
                prev_x, prev_y = 0, 0
                cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
                continue

            if index and not middle:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 7)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

            if status == [False, False, False, False, False]:
                cv2.imwrite("saved_drawing.png", canvas)
                print("Saved drawing!")

    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)

    cv2.imshow("AR Hand Drawing App", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
