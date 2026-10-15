import cv2
import mediapipe as mp
import numpy as np


class LandmarkExtractor:

  def __init__(self):
    self.mp_holistic = mp.solutions.holistic
    self.holistic = self.mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    self.mp_drawing = mp.solutions.drawing_utils

  def process_frame(self, frame):
    """Converts frame to RGB and processes landmarks."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = self.holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

  def extract_keypoints(self, results):
    """Flattens pose, left hand, and right hand keypoints into a 1D vector."""
    pose = (
        np.array([
            [res.x, res.y, res.z, res.visibility]
            for res in results.pose_landmarks.landmark
        ]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )

    lh = (
        np.array([
            [res.x, res.y, res.z]
            for res in results.left_hand_landmarks.landmark
        ]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    rh = (
        np.array([
            [res.x, res.y, res.z]
            for res in results.right_hand_landmarks.landmark
        ]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([pose, lh, rh])

  def draw_landmarks(self, image, results):
    """Visualizes landmarks on frame."""
    if results.left_hand_landmarks:
      self.mp_drawing.draw_landmarks(
          image,
          results.left_hand_landmarks,
          self.mp_holistic.HAND_CONNECTIONS,
      )
    if results.right_hand_landmarks:
      self.mp_drawing.draw_landmarks(
          image,
          results.right_hand_landmarks,
          self.mp_holistic.HAND_CONNECTIONS,
      )
    if results.pose_landmarks:
      self.mp_drawing.draw_landmarks(
          image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
      )