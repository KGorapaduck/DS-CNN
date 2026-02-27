import tensorflow as tf
import numpy as np
import librosa
import glob
import os
import json
import collections
import models

# --- 1. 설정 (Settings) ---
CHECKPOINT_DIR = "./work/ds_cnn_korean/best"
AUDIO_DIR = "./speech_dataset/for_validation_from_AIHUB"

SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 40
WINDOW_STRIDE_MS = 20
DCT_COEFFICIENT_COUNT = 10
CLIP_DURATION_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_MS / 1000)
CHUNK_SIZE = int(SAMPLE_RATE * 0.25) # 0.25초 (4000 픽셀)

LABELS = ['_silence_', '_unknown_', 'understand', 'quiz']

# --- 모델 초기화 함수 ---
def build_model(sess):
    model_settings = models.prepare_model_settings(
        label_count=len(LABELS),
        sample_rate=SAMPLE_RATE,
        clip_duration_ms=CLIP_DURATION_MS,
        window_size_ms=WINDOW_SIZE_MS,
        window_stride_ms=WINDOW_STRIDE_MS,
        dct_coefficient_count=DCT_COEFFICIENT_COUNT,
        activations_bits=32
    )

    wav_placeholder = tf.placeholder(tf.float32, [CLIP_DURATION_SAMPLES, 1], name='wav_data')
    
    # 1. MFCC 추출 그래프 구성
    from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
    spectrogram = contrib_audio.audio_spectrogram(
        wav_placeholder,
        window_size=int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000),
        stride=int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000),
        magnitude_squared=True)
    mfcc = contrib_audio.mfcc(
        spectrogram,
        SAMPLE_RATE,
        dct_coefficient_count=DCT_COEFFICIENT_COUNT)
    mfcc_flatten = tf.reshape(mfcc, [-1])

    # 2. DS-CNN 추론 그래프 구성
    fingerprint_size = model_settings['fingerprint_size']
    fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')
    model_size_info = [5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1]

    logits = models.create_model(
        fingerprint_input=fingerprint_input,
        model_settings=model_settings,
        model_architecture="ds_cnn",
        model_size_info=model_size_info,
        is_training=False)
        
    probabilities_op = tf.nn.softmax(logits, name='labels_softmax')

    # 체크포인트 로드
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_state = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if not checkpoint_state or not checkpoint_state.model_checkpoint_path:
        print("No checkpoint found in", CHECKPOINT_DIR)
        exit(1)
    print(f"Loading checkpoint: {checkpoint_state.model_checkpoint_path}")
    saver.restore(sess, checkpoint_state.model_checkpoint_path)

    return wav_placeholder, mfcc_flatten, fingerprint_input, probabilities_op

# --- 단일 오디오 파일 평가 함수 ---
def evaluate_audio_clip(sess, wav_placeholder, mfcc_flatten, fingerprint_input, probabilities_op, wav_path, target_keyword):
    # 타겟 단어가 나온 파일만 로딩
    try:
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return 0.0

    # 0.25초 단위로 Sliding Window하며 최대 점수 탐색
    audio_buffer = np.zeros(CLIP_DURATION_SAMPLES, dtype=np.float32)
    window_history = collections.deque(maxlen=2)
    max_score = 0.0

    # 패딩 등을 고려하여 전체 오디오를 청크 단위로 분할 처리
    num_chunks = int(np.ceil(len(audio) / CHUNK_SIZE))
    padded_audio = np.pad(audio, (0, num_chunks * CHUNK_SIZE - len(audio)))

    for i in range(num_chunks):
        audio_chunk = padded_audio[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
        
        audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
        audio_buffer[-CHUNK_SIZE:] = audio_chunk
        
        # MFCC
        mfcc_feat = sess.run(mfcc_flatten, feed_dict={wav_placeholder: audio_buffer.reshape(-1, 1)})
        
        # Probabilities
        probs = sess.run(probabilities_op, feed_dict={fingerprint_input: [mfcc_feat]})[0]
        window_history.append(probs)
        
        if len(window_history) == window_history.maxlen:
            smoothed_output = np.mean(window_history, axis=0)
            
            # target keyword's score in the current window
            target_index = LABELS.index(target_keyword)
            score = smoothed_output[target_index]
            
            # prediction for volume tracking logic (optional here since we sweep all)
            top_index = np.argmax(smoothed_output)
            prediction = LABELS[top_index]
            
            # 볼륨 체크 (실시간과 동일)
            volume = np.max(np.abs(audio_chunk))
            if volume >= 0.02:
                if score > max_score:
                    max_score = score

    return max_score

# --- 메인 로직 ---
def main():
    sess = tf.compat.v1.Session()
    wav_placeholder, mfcc_flatten, fingerprint_input, probabilities_op = build_model(sess)

    # 1. JSON 파일 검색을 통해 '퀴즈'와 '이해하셨나요'가 포함된 오디오 리스트 색인
    quiz_files = set()
    understand_files = set()
    
    print("\nScanning JSON files for target keywords...")
    for json_path in glob.glob(os.path.join(AUDIO_DIR, "*.json")):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            wav_path = json_path.replace(".json", ".wav")
            if not os.path.exists(wav_path):
                continue

            for dialog in data.get("dialogs", []):
                text = dialog.get("text", "")
                if "퀴즈" in text:
                    quiz_files.add(wav_path)
                if "이해" in text:  # '이해하셨나요'의 포괄적 탐색
                    understand_files.add(wav_path)
        except Exception as e:
            pass

    print(f"Found {len(quiz_files)} files containing 'quiz'")
    print(f"Found {len(understand_files)} files containing 'understand' (or similar forms)")

    # 2. 평가 진행
    print("\n--- Evaluating 'quiz' Sensitivity ---")
    quiz_scores = []
    for count, wav_path in enumerate(list(quiz_files)):
        score = evaluate_audio_clip(sess, wav_placeholder, mfcc_flatten, fingerprint_input, probabilities_op, wav_path, "quiz")
        quiz_scores.append(score)
        print(f"[{count+1}/{len(quiz_files)}] {os.path.basename(wav_path)} -> Max Match Score: {score:.4f}")
        
    print("\n--- Evaluating 'understand' Sensitivity ---")
    understand_scores = []
    for count, wav_path in enumerate(list(understand_files)):
        score = evaluate_audio_clip(sess, wav_placeholder, mfcc_flatten, fingerprint_input, probabilities_op, wav_path, "understand")
        understand_scores.append(score)
        print(f"[{count+1}/{len(understand_files)}] {os.path.basename(wav_path)} -> Max Match Score: {score:.4f}")

    # 3. 통계 요약
    print("\n==============================================")
    print("📈 Threshold Evaluation Summary (Maximum Confidences)")
    print("==============================================")
    
    if quiz_scores:
        q_arr = np.array(quiz_scores)
        print(f"[QUIZ]\n - Average Max Score: {np.mean(q_arr):.4f}\n - Median Max Score : {np.median(q_arr):.4f}\n - Lowest Max Score : {np.min(q_arr):.4f}\n - Highest Max Score: {np.max(q_arr):.4f}")
    if understand_scores:
        u_arr = np.array(understand_scores)
        print(f"[UNDERSTAND]\n - Average Max Score: {np.mean(u_arr):.4f}\n - Median Max Score : {np.median(u_arr):.4f}\n - Lowest Max Score : {np.min(u_arr):.4f}\n - Highest Max Score: {np.max(u_arr):.4f}")
    
    sess.close()

if __name__ == "__main__":
    main()
