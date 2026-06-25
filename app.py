import streamlit as st
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import queue
import re
import pickle
from transformers import pipeline
from spellchecker import SpellChecker

# Page Configuration Setup
st.set_page_config(page_title="AI Speech Suite", layout="wide", page_icon="🎙️")

st.title("🎙️ AI-Powered Hybrid Speech Suite")
st.markdown("### Developed by: **Kashaf (Institute of Space Technology)**")
st.markdown("---")

# ==========================================
# 1. LOADING PRE-TRAINED MODELS INSTANTLY
# ==========================================

@st.cache_resource
def load_all_pretrained_engines():
    # A. Real Dictionary Load Karna
    spell = SpellChecker()
    full_vocab = set(spell.word_frequency.dictionary.keys())
    
    # B. Whisper Model Load Karna
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    
    # C. Load Pickle Models (No Training Loop!)
    with open('svm_model.pkl', 'rb') as f: svm = pickle.load(f)
    with open('knn_model.pkl', 'rb') as f: knn = pickle.load(f)
    with open('scaler_model.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f: kmeans = pickle.load(f)
    
    return transcriber, full_vocab, svm, knn, scaler, kmeans

# Quick Loading Spinner
try:
    with st.spinner("⏳ Loading Pre-trained Models into System Memory..."):
        transcriber, full_vocabulary, svm_model, knn_model, scaler_model, kmeans_model = load_all_pretrained_engines()
    st.success("🎉 All Systems Online! Pre-trained Models Loaded Instantly.")
except FileNotFoundError:
    st.error("❌ Error: Pickle files nahi milin! Pehle Jupyter mein step 1 wala code run karo.")

# ==========================================
# 2. CORE HELPER ALGORITHMS
# ==========================================

def calculate_heuristic(current_str, local_candidates):
    if not local_candidates: return 999
    return min([sum(1 for a, b in zip(current_str, w) if a != b) + abs(len(current_str) - len(w)) for w in local_candidates])

def a_star_library_spell_check(typo_word, full_vocab):
    local_candidates = [w for w in full_vocab if abs(len(w) - len(typo_word)) <= 1]
    local_candidates_set = set(local_candidates)
    
    pq = queue.PriorityQueue()
    h_0 = calculate_heuristic(typo_word, local_candidates)
    pq.put((0 + h_0, 0, typo_word, [typo_word]))
    
    visited = set()
    letters = 'abcdefghijklmnopqrstuvwxyz'
    
    while not pq.empty():
        f_n, g_n, current_str, path = pq.get()
        if current_str in local_candidates_set:
            return current_str, path, len(visited)
        if current_str not in visited:
            visited.add(current_str)
            for i in range(len(current_str)):
                for l in letters:
                    neighbor = current_str[:i] + l + current_str[i+1:]
                    if neighbor not in visited:
                        new_g = g_n + 1
                        new_h = calculate_heuristic(neighbor, local_candidates)
                        pq.put((new_g + new_h, new_g, neighbor, path + [neighbor]))
    return typo_word, [typo_word], len(visited)

# ==========================================
# 3. USER INTERACTION & DASHBOARD LAYOUT
# ==========================================

st.sidebar.header("📁 Audio Input Panel")
uploaded_file = st.sidebar.file_uploader("Upload your recording (.wav format)", type=["wav"])

if uploaded_file is not None:
    with open("temp_app_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.sidebar.audio(uploaded_file, format="audio/wav")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Stage 1: Acoustic Analysis & ML Models")
        y_audio, sr_audio = librosa.load("temp_app_audio.wav", sr=16000)
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        librosa.display.waveshow(y_audio, sr=sr_audio, ax=ax, color='dodgerblue')
        ax.set_title("Raw Audio Waveform (Heuristic Analysis)")
        st.pyplot(fig)
        
        mfccs_audio = librosa.feature.mfcc(y=y_audio, sr=sr_audio, n_mfcc=13)
        processed_features = np.mean(mfccs_audio.T, axis=0).reshape(1, -1)
        scaled_features = scaler_model.transform(processed_features)
        
        svm_res = "Male 👦" if svm_model.predict(scaled_features)[0] == 0 else "Female 👧"
        knn_res = "Male 👦" if knn_model.predict(scaled_features)[0] == 0 else "Female 👧"
        kmeans_res = "Cluster 0 (Male Group)" if kmeans_model.predict(scaled_features)[0] == 0 else "Cluster 1 (Female Group)"
        
        st.info(f"**Supervised SVM Prediction:** {svm_res}")
        st.info(f"**Supervised KNN Prediction:** {knn_res}")
        st.warning(f"**Unsupervised K-Means Cluster Assignment:** {kmeans_res}")
        
    with col2:
        st.subheader("📝 Stage 2: Deep Learning Transcription")
        with st.spinner("🎙️ Whisper Model is Decoding Speech to Text..."):
            trans_res = transcriber({"array": y_audio, "sampling_rate": sr_audio})
            raw_text = trans_res['text']
            
        st.text_area("Original Whisper Raw Output:", value=raw_text, height=80)
        st.subheader("🚀 Stage 3: Post-Transcription Graph Optimization (A*)")
        
        words = raw_text.split()
        clean_words = []
        error_logs = []
        
        for word in words:
            clean_word = word.strip(",.?!").lower()
            punc = "".join(re.findall(r'[,.?!]+$', word))
            
            if clean_word not in full_vocabulary and clean_word.isalpha():
                corrected, path, nodes = a_star_library_spell_check(clean_word, full_vocabulary)
                error_logs.append(f"❌ **Typo Found:** '{clean_word}' ➡️ Corrected to **'{corrected}'** (Nodes: {nodes})")
                if word[0].isupper(): corrected = corrected.capitalize()
                clean_words.append(corrected + punc)
            else:
                clean_words.append(word)
                
        final_text = " ".join(clean_words)
        
        if error_logs:
            for log in error_logs: st.markdown(log)
        else:
            st.write("✨ No acoustic typos detected by the A* Layer.")
            
        st.success(f"**Final Clean Optimized Transcript:**\n\n \"{final_text}\"")
else:
    st.info("👈 Please upload a voice clip in the sidebar to initiate the AI Pipeline.")