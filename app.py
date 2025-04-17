# Import necessary libraries
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from PIL import Image
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import os
import time
import warnings
from indic_transliteration.sanscript import transliterate
from indic_transliteration import sanscript

# Suppress any warning messages to keep the UI clean
warnings.filterwarnings("ignore")

# Constants used in the model and processing
MAX_LENGTH = 34  # Maximum length of generated caption sequence
IMG_SIZE = 224   # Size to which input image will be resized
MODEL_PATH = "model.keras"  # Path to the caption generation model
FEATURE_EXTRACTOR_PATH = "feature_extractor.keras"  # Path to CNN feature extractor model
TOKENIZER_PATH = "tokenizer.pkl"  # Path to saved tokenizer object

# Load the trained caption generation model (cached for performance)
@st.cache_resource
def load_caption_model():
    return load_model(MODEL_PATH)

# Load the CNN feature extractor model (e.g., InceptionV3, ResNet, etc.)
@st.cache_resource
def load_feature_extractor():
    return load_model(FEATURE_EXTRACTOR_PATH)

# Load the tokenizer used to map words to integers
@st.cache_data
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

# Generate a caption for a given image using the model and tokenizer
def generate_caption(image, model, feature_extractor, tokenizer):
    # Resize and normalize the image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Extract image features using CNN
    features = feature_extractor.predict(img, verbose=0)

    # Start generating the caption with the start token
    in_text = "startseq"
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)

        # Predict the next word
        yhat = model.predict([features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index)

        # Stop if word is None or we reach the end token
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    # Remove start and end tokens from the final caption
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption

# Convert given text into speech using gTTS and return path to temp mp3 file
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts.save(tmpfile.name)
        tmpfile.close()
        return tmpfile.name

# Streamlit app configuration and title
st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator")
st.markdown("Upload an image and let the AI generate a caption for it!")

# File uploader for users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Once an image is uploaded
if uploaded_file:
    # Load and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate a caption using the models
    with st.spinner("Generating caption..."):
        model = load_caption_model()
        feature_extractor = load_feature_extractor()
        tokenizer = load_tokenizer()
        caption = generate_caption(image, model, feature_extractor, tokenizer)

    # Display the generated English caption
    st.markdown("### 📝 Generated Caption:")
    st.success(caption)

    # Translate the caption to Telugu
    translated_caption = GoogleTranslator(source='en', target='te').translate(caption)
    st.markdown("### 🇮🇳 Translated Caption (Telugu):")
    st.success(translated_caption)

# Transliterate Telugu script into Roman script for pronunciation help
def romanize_telugu(telugu_text):
    try:
        return transliterate(telugu_text, sanscript.TELUGU, sanscript.ITRANS)
    except Exception:
        return "Could not transliterate the Telugu caption."

# Generate the romanized Telugu version and show it
romanized_caption = romanize_telugu(translated_caption)
st.markdown("### 🗣️ Romanized Telugu (Pronunciation):")
st.info(romanized_caption)

# Add two buttons side-by-side for audio playback of captions
col1, col2 = st.columns(2)

# Speak English caption using TTS
with col1:
    if st.button("🔊 Speak English Caption"):
        audio_file_path = text_to_speech(caption, lang='en')
        with open(audio_file_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")
        time.sleep(1)
        os.remove(audio_file_path)

# Speak Telugu caption using TTS
with col2:
    if st.button("🔊 Speak Telugu Caption"):
        audio_file_path = text_to_speech(translated_caption, lang='te')
        with open(audio_file_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")
        time.sleep(1)
        os.remove(audio_file_path)
