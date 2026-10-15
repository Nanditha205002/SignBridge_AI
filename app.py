import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from src.landmark_extractor import LandmarkExtractor
from src.llm_refiner import SentenceBuilder
from src.tts_engine import AudioSpeaker

st.set_page_config(page_title="SignBridge AI", layout="wide")
st.title("SignBridge AI: Communication Assistant")

# Sidebar settings
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
target_lang = st.sidebar.selectbox("Output Language", ["English", "Hindi", "Kannada"])

ACTIONS = np.array(['hello', 'hospital', 'doctor', 'yesterday', 'go'])

@st.cache_resource
def load_lstm_model():
    return tf.keras.models.load_model('models/signbridge_lstm.h5')

try:
    model = load_lstm_model()
except Exception as e:
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

extractor = LandmarkExtractor()
speaker = AudioSpeaker()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Real-Time Camera Feed")
    run_cam = st.checkbox("Turn On Camera")
    frame_window = st.image([])

with col2:
    st.subheader("Detected Signs Sequence")
    sequence_placeholder = st.empty()
    st.subheader("Generative AI Sentence")
    sentence_placeholder = st.empty()

sequence = []
sentence_tokens = []
predictions = []
threshold = 0.8

cap = None
if run_cam:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image, results = extractor.process_frame(frame)
        extractor.draw_landmarks(image, results)
        
        keypoints = extractor.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(res)
            
            if res[predicted_idx] > threshold:
                predicted_action = ACTIONS[predicted_idx]
                
                if len(sentence_tokens) == 0 or predicted_action != sentence_tokens[-1]:
                    sentence_tokens.append(predicted_action)
                    sequence_placeholder.markdown(f"**Tokens:** `{' -> '.join(sentence_tokens)}`")
                    
        frame_window.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
    cap.release()

if st.button("Generate & Speak Sentence"):
    if len(sentence_tokens) > 0 and api_key:
        refiner = SentenceBuilder(api_key=api_key)
        final_sentence = refiner.construct_sentence(sentence_tokens, target_language=target_lang)
        sentence_placeholder.success(f"**Result:** {final_sentence}")
        
        lang_code_map = {"English": "en", "Hindi": "hi", "Kannada": "kn"}
        speaker.speak(final_sentence, lang_code=lang_code_map.get(target_lang, "en"))
    else:
        st.warning("Ensure tokens are detected and Gemini API key is provided.")