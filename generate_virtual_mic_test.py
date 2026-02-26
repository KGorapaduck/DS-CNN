import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf

# 설정 파라미터
SAMPLE_RATE = 16000
TARGET_DURATION_SEC = 120  # 총 2분 길이의 오디오 생성
OUTPUT_FILE = "korean_virtual_mic_test.wav"

# 데이터셋 경로
DATA_DIR = "./speech_dataset"
QUIZ_DIR = os.path.join(DATA_DIR, "quiz")
UNDERSTAND_DIR = os.path.join(DATA_DIR, "understand")
UNKNOWN_DIR = os.path.join(DATA_DIR, "unknown")
NOISE_DIR = os.path.join(DATA_DIR, "_background_noise_")

def load_audio(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])

def main():
    print("🎙️ 가상 마이크 테스트 시나리오 오디오 생성 시작 (총 2분 예상)...")
    
    # 1. 사용 가능한 파일 리스트업
    quiz_files = glob.glob(os.path.join(QUIZ_DIR, "*.wav"))
    understand_files = glob.glob(os.path.join(UNDERSTAND_DIR, "*.wav"))
    unknown_files = glob.glob(os.path.join(UNKNOWN_DIR, "*.wav"))
    noise_files = glob.glob(os.path.join(NOISE_DIR, "*.wav"))
    
    if not quiz_files or not understand_files or not unknown_files or not noise_files:
        print("❌ 필수 데이터(quiz, understand, unknown, noise)가 충분하지 않습니다!")
        return

    # 2. 오디오 조각 믹싱
    final_audio = []
    current_length_samples = 0
    target_samples = TARGET_DURATION_SEC * SAMPLE_RATE
    
    print("조각 모음 중...")
    
    transcript = []
    
    def add_to_transcript(label, start_samples, duration_samples):
        start_ms = (start_samples / SAMPLE_RATE) * 1000
        end_ms = ((start_samples + duration_samples) / SAMPLE_RATE) * 1000
        start_m = int(start_ms // 60000)
        start_s = int((start_ms % 60000) // 1000)
        end_m = int(end_ms // 60000)
        end_s = int((end_ms % 60000) // 1000)
        
        time_str = f"[{start_m:02d}:{start_s:02d} ~ {end_m:02d}:{end_s:02d}]"
        
        if label in ['quiz', 'understand']:
            transcript.append(f"{time_str} 🎯 타겟 단어: '{label}'")
        elif label == 'unknown':
            transcript.append(f"{time_str} 💬 타단어/일반대화 (False Positive 테스트)")
        elif label == 'noise':
            transcript.append(f"{time_str} 🔊 백그라운드 소음")
    
    # 맨 처음 2초는 노이즈로 시작 (안정화 버퍼)
    initial_noise_file = random.choice(noise_files)
    noise_audio = load_audio(initial_noise_file)
    if len(noise_audio) > SAMPLE_RATE * 2:
        start_idx = random.randint(0, len(noise_audio) - SAMPLE_RATE * 2)
        chunk = noise_audio[start_idx : start_idx + SAMPLE_RATE * 2]
        final_audio.append(chunk)
        add_to_transcript('noise', current_length_samples, len(chunk))
        current_length_samples += len(chunk)

    # 랜덤하게 조각 이어붙이기
    while current_length_samples < target_samples:
        choice = random.choices(
            ['quiz', 'understand', 'unknown', 'noise'], 
            weights=[10, 10, 50, 30]  # 혼동하기 쉬운 일반대화(unknown) 50%, 타겟 단어 20%, 노이즈 30%
        )[0]
        
        audio_chunk = []
        if choice == 'quiz':
            audio_chunk = load_audio(random.choice(quiz_files))
            audio_chunk = audio_chunk * 0.8 
            add_to_transcript('quiz', current_length_samples, len(audio_chunk))
        elif choice == 'understand':
            audio_chunk = load_audio(random.choice(understand_files))
            audio_chunk = audio_chunk * 0.8
            add_to_transcript('understand', current_length_samples, len(audio_chunk))
        elif choice == 'unknown':
            audio_chunk = load_audio(random.choice(unknown_files))
            audio_chunk = audio_chunk * random.uniform(0.5, 0.9)
            add_to_transcript('unknown', current_length_samples, len(audio_chunk))
        elif choice == 'noise':
            n_file = load_audio(random.choice(noise_files))
            if len(n_file) > SAMPLE_RATE * 1.5:
                dur = random.randint(int(SAMPLE_RATE * 1.0), int(SAMPLE_RATE * 2.0))
                start_idx = random.randint(0, len(n_file) - dur)
                audio_chunk = n_file[start_idx : start_idx + dur]
            audio_chunk = audio_chunk * random.uniform(0.1, 0.3)
            add_to_transcript('noise', current_length_samples, len(audio_chunk))
            
        if len(audio_chunk) > 0:
            final_audio.append(audio_chunk)
            current_length_samples += len(audio_chunk)
            
            # 단어와 단어 사이에 무작위 짧은 묵음(0.2~0.8초) 삽입
            silence_len = random.randint(int(SAMPLE_RATE * 0.2), int(SAMPLE_RATE * 0.8))
            final_audio.append(np.zeros(silence_len))
            current_length_samples += silence_len

    # 1D numpy 배열로 결합
    final_audio_concat = np.concatenate(final_audio)
    
    # 3. 목표 길이(2분)에 맞춰 자르기
    final_audio_concat = final_audio_concat[:target_samples]
    
    # 4. WAV 및 스크립트 파일 저장
    print(f"✅ 파일 생성 완료: {OUTPUT_FILE}")
    sf.write(OUTPUT_FILE, final_audio_concat, SAMPLE_RATE, subtype='PCM_16')
    
    script_file = "korean_virtual_mic_test_script.txt"
    with open(script_file, "w", encoding="utf-8") as f:
        f.write("==== 🎙️ 라즈베리파이 가상 마이크 테스트 (2분 극한 시나리오) ====\n")
        f.write("💡 목적: 타단어/일반대화 및 소음 구간에서는 무응답('silence' 또는 'unknown')을 유지하고,\n")
        f.write("         오직 '🎯 타겟 단어' 구간에서만 알림이 뜨는지 검증합니다.\n")
        f.write("----------------------------------------------------------------------\n\n")
        
        for line in transcript:
            # 타겟 샘플 범위를 넘어서는 건 무시
            time_str = line[1:6]
            m, s = map(int, time_str.split(':'))
            if m * 60 + s <= TARGET_DURATION_SEC:
                # 🎯 타겟 단어일 경우 시각적으로 강조
                if "🎯" in line:
                    f.write("\n" + "="*50 + "\n")
                    f.write(line + "  <-- 🚨 모델 예측 반응 필수!\n")
                    f.write("="*50 + "\n\n")
                else:
                    f.write(line + "\n")
                    
    print(f"✅ 상세 대본 생성 완료: {script_file}")

if __name__ == '__main__':
    main()
