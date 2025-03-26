import cv2
import mediapipe as mp
import time
import numpy as np
from pynput import keyboard, mouse
import screeninfo
from threading import Thread
import queue
import sys
import pyautogui
import os

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

# Custom drawing specs for better skeleton visibility
LANDMARK_COLOR = (255, 0, 0)  # Bright red for landmarks
CONNECTION_COLOR = (0, 255, 0)  # Bright green for connections
LANDMARK_THICKNESS = 4  # Thicker dots
CONNECTION_THICKNESS = 2  # Thicker lines

# Initialize pynput controllers
kb_controller = keyboard.Controller()
mouse_controller = mouse.Controller()

# Screen size for cursor mapping
try:
    screen = screeninfo.get_monitors()[0]
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.width, screen.height
except Exception as e:
    print(f"Error getting screen info: {e}")
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080  # Fallback resolution

# Queue for async command processing
command_queue = queue.Queue()

# Gesture recognition class with advanced features
class GestureController:
    def __init__(self):
        self.prev_frame_time = time.time()
        self.hand_data = {}
        self.debounce_time = {}
        self.debounce_delay = 0.3
        self.prev_cursor_pos = None

    def calculate_fps(self):
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
        self.prev_frame_time = current_time
        return fps

    def is_finger_open(self, tip, pip, mcp):
        return tip.y < pip.y and tip.y < mcp.y

    def get_hand_velocity(self, curr_pos, prev_pos, delta_time):
        if prev_pos is None or delta_time == 0:
            return 0, 0
        dx = (curr_pos[0] - prev_pos[0]) / delta_time
        dy = (curr_pos[1] - prev_pos[1]) / delta_time
        return dx, dy

    def move_cursor(self, hand_x, hand_y, frame_width, frame_height):
        """Move cursor with enhanced smoothing for stability."""
        target_x = int(np.interp(hand_x, [0, frame_width], [-50, SCREEN_WIDTH + 50]))
        target_y = int(np.interp(hand_y, [0, frame_height], [-50, SCREEN_HEIGHT + 50]))

        smoothed_x = target_x
        smoothed_y = target_y

        if self.prev_cursor_pos is not None:
            curr_x, curr_y = self.prev_cursor_pos
            # Stronger smoothing: 80% previous position, 20% new position
            smoothed_x = int(curr_x * 0.8 + target_x * 0.2)
            smoothed_y = int(curr_y * 0.8 + target_y * 0.2)

        self.prev_cursor_pos = (smoothed_x, smoothed_y)

        final_x = min(max(smoothed_x, 0), SCREEN_WIDTH - 1)
        final_y = min(max(smoothed_y, 0), SCREEN_HEIGHT - 1)
        mouse_controller.position = (final_x, final_y)

    def calculate_pinch_distance(self, thumb_tip, index_tip, frame_width):
        return np.hypot(
            (thumb_tip.x - index_tip.x) * frame_width,
            (thumb_tip.y - index_tip.y) * frame_width
        )

    def process_commands(self):
        while True:
            try:
                hand_id, command = command_queue.get(timeout=0.1)
                if command == "Click":
                    mouse_controller.click(mouse.Button.left, 1)
                elif command == "Pinch In":
                    kb_controller.press(keyboard.Key.ctrl)
                    mouse_controller.scroll(0, -1)
                    kb_controller.release(keyboard.Key.ctrl)
                elif command == "Pinch Out":
                    kb_controller.press(keyboard.Key.ctrl)
                    mouse_controller.scroll(0, 1)
                    kb_controller.release(keyboard.Key.ctrl)
                elif command == "Swipe Left":
                    kb_controller.press(keyboard.Key.left)
                    kb_controller.release(keyboard.Key.left)
                elif command == "Swipe Right":
                    kb_controller.press(keyboard.Key.right)
                    kb_controller.release(keyboard.Key.right)
                elif command == "Volume Up":
                    kb_controller.press(keyboard.Key.media_volume_up)
                    kb_controller.release(keyboard.Key.media_volume_up)
                elif command == "Volume Down":
                    kb_controller.press(keyboard.Key.media_volume_down)
                    kb_controller.release(keyboard.Key.media_volume_down)
                elif command == "App Switch":
                    kb_controller.press(keyboard.Key.alt)
                    kb_controller.press(keyboard.Key.tab)
                    kb_controller.release(keyboard.Key.tab)
                    kb_controller.release(keyboard.Key.alt)
                elif command == "Maximize Window":
                    kb_controller.press(keyboard.Key.alt)
                    kb_controller.press(keyboard.Key.space)
                    kb_controller.release(keyboard.Key.space)
                    kb_controller.press('x')
                    kb_controller.release('x')
                    kb_controller.release(keyboard.Key.alt)
                elif command == "Minimize Window":
                    kb_controller.press(keyboard.Key.alt)
                    kb_controller.press(keyboard.Key.space)
                    kb_controller.release(keyboard.Key.space)
                    kb_controller.press('n')
                    kb_controller.release('n')
                    kb_controller.release(keyboard.Key.alt)
                elif command == "Play/Pause":
                    kb_controller.press(keyboard.Key.media_play_pause)
                    kb_controller.release(keyboard.Key.media_play_pause)
                elif command == "Next Track":
                    kb_controller.press(keyboard.Key.media_next)
                    kb_controller.release(keyboard.Key.media_next)
                elif command == "Screenshot":
                    pyautogui.screenshot().save(f"screenshot_{int(time.time())}.png")
                command_queue.task_done()
            except queue.Empty:
                pass

    def recognize_gesture(self, hand_landmarks, frame_shape, hand_id):
        landmarks = hand_landmarks.landmark
        thumb_tip, thumb_ip = landmarks[4], landmarks[3]
        index_tip, index_pip, index_mcp = landmarks[8], landmarks[6], landmarks[5]
        middle_tip, middle_pip, middle_mcp = landmarks[12], landmarks[10], landmarks[9]
        ring_tip, ring_pip, ring_mcp = landmarks[16], landmarks[14], landmarks[13]
        pinky_tip, pinky_pip, pinky_mcp = landmarks[20], landmarks[18], landmarks[17]
        wrist = landmarks[0]

        finger_states = [
            1 if thumb_tip.x < thumb_ip.x else 0,
            1 if self.is_finger_open(index_tip, index_pip, index_mcp) else 0,
            1 if self.is_finger_open(middle_tip, middle_pip, middle_mcp) else 0,
            1 if self.is_finger_open(ring_tip, ring_pip, ring_mcp) else 0,
            1 if self.is_finger_open(pinky_tip, pinky_pip, pinky_mcp) else 0
        ]

        curr_pos = (int(wrist.x * frame_shape[1]), int(wrist.y * frame_shape[0]))
        prev_pos = self.hand_data.get(hand_id, {}).get('prev_pos', None)
        delta_time = time.time() - self.prev_frame_time
        dx, dy = self.get_hand_velocity(curr_pos, prev_pos, delta_time)

        self.hand_data[hand_id] = {
            'fingers': finger_states,
            'pos': curr_pos,
            'prev_pos': curr_pos,
            'gesture': "Tracking"
        }

        current_time = time.time()
        debounce = hand_id not in self.debounce_time or (current_time - self.debounce_time.get(hand_id, 0)) > self.debounce_delay

        if finger_states == [0, 1, 0, 0, 0]:  # Index finger up (Tracking only)
            self.hand_data[hand_id]['gesture'] = "Tracking"
            self.move_cursor(curr_pos[0], curr_pos[1], frame_shape[1], frame_shape[0])

        elif finger_states == [0, 1, 1, 0, 0]:  # Index + Middle (Click)
            self.hand_data[hand_id]['gesture'] = "Click"
            if debounce:
                command_queue.put((hand_id, "Click"))
                self.debounce_time[hand_id] = current_time

        elif finger_states == [1, 1, 0, 0, 0]:  # Pinch
            pinch_dist = self.calculate_pinch_distance(thumb_tip, index_tip, frame_shape[1])
            if pinch_dist < 50:
                self.hand_data[hand_id]['gesture'] = "Pinch In"
                if debounce:
                    command_queue.put((hand_id, "Pinch In"))
                    self.debounce_time[hand_id] = current_time
            elif pinch_dist > 100:
                self.hand_data[hand_id]['gesture'] = "Pinch Out"
                if debounce:
                    command_queue.put((hand_id, "Pinch Out"))
                    self.debounce_time[hand_id] = current_time

        elif finger_states == [0, 0, 1, 0, 0] and abs(dx) > 50:  # Swipe
            self.hand_data[hand_id]['gesture'] = "Swipe Left" if dx < 0 else "Swipe Right"
            if debounce:
                command_queue.put((hand_id, "Swipe Left" if dx < 0 else "Swipe Right"))
                self.debounce_time[hand_id] = current_time

        elif finger_states == [1, 1, 1, 1, 1]:  # Volume Up
            self.hand_data[hand_id]['gesture'] = "Volume Up"
            if debounce:
                command_queue.put((hand_id, "Volume Up"))
                self.debounce_time[hand_id] = current_time

        elif finger_states == [0, 0, 0, 0, 0]:  # Volume Down
            self.hand_data[hand_id]['gesture'] = "Volume Down"
            if debounce:
                command_queue.put((hand_id, "Volume Down"))
                self.debounce_time[hand_id] = current_time

        elif finger_states == [0, 0, 0, 0, 1]:  # App Switch
            self.hand_data[hand_id]['gesture'] = "App Switch"
            if debounce:
                command_queue.put((hand_id, "App Switch"))
                self.debounce_time[hand_id] = current_time

        elif finger_states == [0, 1, 0, 0, 1]:  # Maximize Window
            self.hand_data[hand_id]['gesture'] = "Maximize Window"
            if debounce:
                command_queue.put((hand_id, "Maximize Window"))
                self.debounce_time[hand_id] = current_time

        elif finger_states == [0, 0, 1, 1, 0]:  # Minimize Window
            self.hand_data[hand_id]['gesture'] = "Minimize Window"
            if debounce:
                command_queue.put((hand_id, "Minimize Window"))
                self.debounce_time[hand_id] = current_time

        elif finger_states == [1, 0, 1, 0, 0]:  # Play/Pause
            self.hand_data[hand_id]['gesture'] = "Play/Pause"
            if debounce:
                command_queue.put((hand_id, "Play/Pause"))
                self.debounce_time[hand_id] = current_time

        elif finger_states == [1, 0, 0, 1, 0]:  # Next Track
            self.hand_data[hand_id]['gesture'] = "Next Track"
            if debounce:
                command_queue.put((hand_id, "Next Track"))
                self.debounce_time[hand_id] = current_time

        elif finger_states == [0, 1, 1, 1, 0]:  # Screenshot
            self.hand_data[hand_id]['gesture'] = "Screenshot"
            if debounce:
                command_queue.put((hand_id, "Screenshot"))
                self.debounce_time[hand_id] = current_time

        return self.hand_data[hand_id]['gesture']

# Main function
def main():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)

        controller = GestureController()
        Thread(target=controller.process_commands, daemon=True).start()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_id = idx
                    # Draw landmarks with custom colors and thickness
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=LANDMARK_COLOR, thickness=LANDMARK_THICKNESS, circle_radius=4),
                        mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=CONNECTION_THICKNESS)
                    )
                    gesture = controller.recognize_gesture(hand_landmarks, frame.shape, hand_id)
                    
                    pos = controller.hand_data[hand_id]['pos']
                    fingers = controller.hand_data[hand_id]['fingers']
                    cv2.circle(frame, pos, 10, (255, 0, 0) if hand_id == 0 else (0, 255, 0), -1)
                    cv2.putText(frame, f"Hand {hand_id}: {gesture} {fingers}", (pos[0] - 50, pos[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            fps = controller.calculate_fps()
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Hand Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()
