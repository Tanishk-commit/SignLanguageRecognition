import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sklearn
import matplotlib

print("🎉 All libraries imported successfully!")

print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)

# Safer way to get Matplotlib version
from importlib.metadata import version
print("Matplotlib:", version("matplotlib"))