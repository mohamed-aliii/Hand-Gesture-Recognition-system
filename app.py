import streamlit as st
import cv2
import numpy as np
from src.config import settings
from src.preprocessing import HandPreprocessor
from src.inference import GestureClassifier

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1100px;
    }
    .stButton > button { width: 100%; }
    .stMarkdown pre { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)
st.markdown("### Hand Gesture Recognition", unsafe_allow_html=True)

# Initialize components
preprocessor = HandPreprocessor()
classifier = GestureClassifier()

import time

if 'run_camera' not in st.session_state:
    st.session_state['run_camera'] = False

controls_col, video_col, prediction_col = st.columns([1,3,2])

with controls_col:
    st.markdown("##### Controls")
    if st.button('Start Camera', key='start'):
        st.session_state['run_camera'] = True
    if st.button('Stop Camera', key='stop'):
        st.session_state['run_camera'] = False

with video_col:
    st.markdown("##### Camera Feed")
    FRAME_WINDOW = st.empty()
    status_text = st.empty()

with prediction_col:
    st.markdown("##### Prediction")
    prediction_text = st.empty()

FEED_WIDTH = 500  # px, adjust as needed for smaller feed

if st.session_state['run_camera']:
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
    status_text.info('Camera started. Showing live predictions.')
    while st.session_state['run_camera'] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            status_text.error('Failed to read from camera.')
            break
        # Process frame
        landmarks, vis_frame = preprocessor.process_frame(frame)
        prediction = "No Gesture"
        confidence = 0.0
        if landmarks is not None:
            gesture, confidence = classifier.predict(landmarks)
            if gesture is not None:
                prediction = f"Prediction: {gesture} ({confidence:.2f})"
            else:
                prediction = "No Gesture Detected"
        else:
            prediction = "No Gesture Detected"
        # Show processed frame and prediction (make feed smaller)
        FRAME_WINDOW.image(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB), width=FEED_WIDTH)
        prediction_text.markdown(f"**{prediction}**")
        time.sleep(0.05)  # ~20 FPS
    cap.release()
    status_text.info('Camera stopped.')
else:
    status_text.warning('Camera is off. Click "Start Camera" to begin.')
    FRAME_WINDOW.empty()
    prediction_text.empty()

