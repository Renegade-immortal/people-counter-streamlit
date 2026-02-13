import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import tempfile
from pathlib import Path

# Load YOLO model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

st.title("Beyondbug INC")
st.markdown("Customer Demographics.")

# --- Sidebar ---
st.sidebar.header("Source Selection")

# Store Buttons
st.sidebar.subheader("Live Store Feeds")
col1, col2 = st.sidebar.columns(2)
if col1.button("CLD"):
    st.session_state['source'] = "rtsp://admin:Netwo@1116@192.168.1.101:554/h264_stream"  # Placeholder IP
    st.session_state['source_type'] = "stream"
if col2.button("CLT"):
    st.session_state['source'] = "rtsp://admin:Netwo@1116@192.168.1.102:554/h264_stream"  # Placeholder IP
    st.session_state['source_type'] = "stream"
col3, col4 = st.sidebar.columns(2)
if col3.button("COK"):
    st.session_state['source'] = "rtsp://admin:Netwo@1116@192.168.1.103:554/h264_stream"  # Placeholder IP
    st.session_state['source_type'] = "stream"
if col4.button("STM"):
    st.session_state['source'] = "rtsp://admin:Netwo@1116@192.168.1.104:554/h264_stream"  # Placeholder IP
    st.session_state['source_type'] = "stream"

if st.sidebar.button("Stop Stream"):
    st.session_state['source'] = None
    st.session_state['source_type'] = None

# File Uploader
st.sidebar.subheader("Upload Media")
uploaded_file = st.sidebar.file_uploader("Choose an image or video file", type=["jpg", "png", "mp4"])

if uploaded_file:
    st.session_state['source'] = uploaded_file
    st.session_state['source_type'] = "file"


# --- Main Logic ---
source = st.session_state.get('source')
source_type = st.session_state.get('source_type')

if source is not None:
    if source_type == "file":
        suffix = Path(source.name).suffix.lower()
        if suffix in [".jpg", ".png"]:
            image = Image.open(source).convert('RGB')
            results = model(image)
            st.image(np.squeeze(results.render()), caption='Detected Image', use_column_width=True)
            
            # Optional: Display detected object counts
            label_counts = results.pandas().xyxy[0]['name'].value_counts()
            st.write("### Detected Counts:")
            st.write(label_counts.to_dict())

        elif suffix == ".mp4":
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(source.read())
            cap = cv2.VideoCapture(tfile.name)
            
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                frame = np.squeeze(results.render())
                stframe.image(frame, channels='BGR', use_column_width=True)
            cap.release()
            st.success("Video processing completed.")

    elif source_type == "stream":
        st.write(f"**Playing Stream:** {source}")
        cap = cv2.VideoCapture(source)
        stframe = st.empty()
        stop_button = st.button("Stop Current Stream")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read stream or stream ended.")
                break
            
            results = model(frame)
            frame = np.squeeze(results.render())
            stframe.image(frame, channels='BGR', use_column_width=True)
        
        cap.release()
