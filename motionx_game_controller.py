import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pyautogui

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class VoiceGuide:
    def __init__(self):
        self.enabled = pyttsx3 is not None
        self.engine = None
        self.last_message = ""
        self.last_spoken_time = 0
        self.cooldown = 1.8

        if self.enabled:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 165)

    def speak(self, message):
        if not self.enabled:
            return

        now = time.time()
        if message == self.last_message and now - self.last_spoken_time < self.cooldown:
            return

        if now - self.last_spoken_time < self.cooldown:
            return

        self.last_message = message
        self.last_spoken_time = now

        threading.Thread(target=self._speak_thread, args=(message,), daemon=True).start()

    def _speak_thread(self, message):
        try:
            self.engine.say(message)
            self.engine.runAndWait()
        except RuntimeError:
            pass


class SkeletonViewer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.mirror_camera = True
        self.voice = VoiceGuide()

        self.last_action_time = 0
        self.action_cooldown = 0.35
        self.target_center_y = None
        self.upper_tolerance_y = 0.08  # jump / UP
        self.lower_tolerance_y = 0.13  # crouch / DOWN
        self.y_lane_follow_speed = 0.09

        # Center/tolerance zone settings
        self.target_center_x = 0.50
        self.center_tolerance_x = 0.10  # fallback until shoulders are detected
        self.x_lane_width_scale = 0.70  # lane total width ≈ shoulder width

        self.extra_connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
        ]

    def landmark_to_point(self, landmark, width, height):
        return int(landmark.x * width), int(landmark.y * height)

    def press_key(self, key):
        threading.Thread(
            target=self._press_key_thread,
            args=(key,),
            daemon=True
        ).start()

    def _press_key_thread(self, key):
        pyautogui.keyDown(key)
        time.sleep(0.025)
        pyautogui.keyUp(key)
        print(f"Pressed: {key.upper()}")

    def draw_connection(self, canvas, landmarks, start, end, width, height, color, thickness):
        start_lm = landmarks[start.value]
        end_lm = landmarks[end.value]

        if start_lm.visibility < 0.45 or end_lm.visibility < 0.45:
            return

        cv2.line(
            canvas,
            self.landmark_to_point(start_lm, width, height),
            self.landmark_to_point(end_lm, width, height),
            color,
            thickness
        )

    def get_body_center(self, landmarks):
        points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
        ]

        visible_points = []
        for point in points:
            lm = landmarks[point.value]
            if lm.visibility >= 0.45:
                visible_points.append((lm.x, lm.y))

        if len(visible_points) < 3:
            return None

        center_x = sum(p[0] for p in visible_points) / len(visible_points)
        center_y = sum(p[1] for p in visible_points) / len(visible_points)
        return center_x, center_y

    def guide_user_to_center(self, center_x, center_y):
        x_diff = center_x - self.target_center_x
        now = time.time()

        if abs(x_diff) <= self.center_tolerance_x:
            self.voice.speak("Perfect. Stay there.")
            return "CENTERED"

        if now - self.last_action_time < self.action_cooldown:
            return "RECENTERING..."

        if x_diff < -self.center_tolerance_x:
            print("Action detected: LEFT_ARROW")
            self.press_key("a")

            # Move center lane to user's new torso position
            self.target_center_x = center_x

            self.last_action_time = now
            return "RIGHT_ARROW -> NEW CENTER SET"

        if x_diff > self.center_tolerance_x:
            print("Action detected: RIGHT_ARROW")
            self.press_key("d")

            # Move center lane to user's new torso position
            self.target_center_x = center_x

            self.last_action_time = now
            return "LEFT_ARROW -> NEW CENTER SET"

        return "ADJUST POSITION"

    def update_x_tolerance_from_shoulders(self, landmarks):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        if left_shoulder.visibility < 0.45 or right_shoulder.visibility < 0.45:
            return

        shoulder_width = abs(right_shoulder.x - left_shoulder.x)

        # center_tolerance_x is half-lane width
        self.center_tolerance_x = shoulder_width * self.x_lane_width_scale

    def draw_center_zone(self, canvas, width, height):
        center_x1 = int((self.target_center_x - self.center_tolerance_x) * width)
        center_x2 = int((self.target_center_x + self.center_tolerance_x) * width)

        overlay = canvas.copy()

        # Center lane only
        cv2.rectangle(overlay, (center_x1, 0), (center_x2, height), (255, 180, 0), -1)
        cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)

        cv2.rectangle(canvas, (center_x1, 0), (center_x2, height), (255, 180, 0), 2)

        cv2.line(
            canvas,
            (int(self.target_center_x * width), 0),
            (int(self.target_center_x * width), height),
            (255, 220, 120),
            1
        )

        cv2.putText(
            canvas,
            "Center Lane",
            (center_x1 + 8, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 120),
            2
        )

    def draw_horizontal_center_zone(self, canvas, width, height):
        if self.target_center_y is None:
            return

        center_y1 = int((self.target_center_y - self.upper_tolerance_y) * height)
        center_y2 = int((self.target_center_y + self.lower_tolerance_y) * height)

        overlay = canvas.copy()

        # Horizontal torso lane only
        cv2.rectangle(overlay, (0, center_y1), (width, center_y2), (255, 180, 0), -1)
        cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)

        cv2.rectangle(canvas, (0, center_y1), (width, center_y2), (255, 180, 0), 2)

        cv2.line(
            canvas,
            (0, int(self.target_center_y * height)),
            (width, int(self.target_center_y * height)),
            (255, 220, 120),
            1
        )

        cv2.putText(
            canvas,
            "Torso Height Lane",
            (30, center_y1 - 10 if center_y1 > 30 else center_y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 120),
            2
        )

    def guide_user_vertical(self, center_y):
        if self.target_center_y is None:
            return None

        y_diff = center_y - self.target_center_y
        now = time.time()

        if -self.upper_tolerance_y <= y_diff <= self.lower_tolerance_y:
            return None

        if now - self.last_action_time < self.action_cooldown:
            return "VERTICAL RECENTERING..."

        # In image coordinates:
        # smaller y = body moved upward
        # larger y = body moved downward
        if y_diff < -self.upper_tolerance_y:
            print("Action detected: UP_ARROW")
            self.press_key("w")
            self.last_action_time = now
            return "UP_ARROW"

        if y_diff > self.lower_tolerance_y:
            print("Action detected: DOWN_ARROW")
            self.press_key("s")
            self.last_action_time = now
            return "DOWN_ARROW"

        return None

    def draw_skeleton(self, frame):
        if self.mirror_camera:
            frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.draw_center_zone(canvas, width, height)
        self.draw_horizontal_center_zone(canvas, width, height)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            self.voice.speak("Step into the camera view")
            cv2.putText(canvas, "No person detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return canvas

        landmarks = results.pose_landmarks.landmark
        self.update_x_tolerance_from_shoulders(landmarks)
        body_center = self.get_body_center(landmarks)

        guide_text = "Finding body center"
        if body_center:
            center_x, center_y = body_center
            if self.target_center_y is None:
                self.target_center_y = center_y
            else:
                self.target_center_y = (
                        (1 - self.y_lane_follow_speed) * self.target_center_y
                        + self.y_lane_follow_speed * center_y
                )
                vertical_action = self.guide_user_vertical(center_y)
                if vertical_action:
                    guide_text = vertical_action

            guide_text = self.guide_user_to_center(center_x, center_y)
            cv2.circle(canvas, (int(center_x * width), int(center_y * height)), 9, (255, 0, 255), -1)
            cv2.line(
                canvas,
                (int(center_x * width), int(center_y * height)),
                (int(self.target_center_x * width), int(center_y * height)),
                (255, 0, 255),
                2
            )

        for start, end in self.mp_pose.POSE_CONNECTIONS:
            start_lm = landmarks[start]
            end_lm = landmarks[end]

            if start_lm.visibility < 0.45 or end_lm.visibility < 0.45:
                continue

            cv2.line(
                canvas,
                self.landmark_to_point(start_lm, width, height),
                self.landmark_to_point(end_lm, width, height),
                (255, 210, 80),
                3
            )

        for start, end in self.extra_connections:
            self.draw_connection(canvas, landmarks, start, end, width, height, (255, 230, 140), 2)

        for landmark in landmarks:
            if landmark.visibility < 0.45:
                continue
            cv2.circle(canvas, self.landmark_to_point(landmark, width, height), 5, (255, 255, 255), -1)

        cv2.putText(canvas, guide_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
        cv2.putText(canvas, "Q: Quit | M: Mirror toggle", (30, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return canvas


def main():
    viewer = SkeletonViewer()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found")
        return

    print("MotionX skeleton voice guide started")
    print("Install voice package if needed: pip install pyttsx3")
    print("Q = quit | M = mirror toggle")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read camera frame")
            break

        skeleton_frame = viewer.draw_skeleton(frame)
        cv2.imshow("MotionX Skeleton Voice Guide", skeleton_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("m"):
            viewer.mirror_camera = not viewer.mirror_camera
            print(f"Mirror camera: {viewer.mirror_camera}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
