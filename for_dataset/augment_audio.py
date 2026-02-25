import os
import random
import librosa
import soundfile as sf
import numpy as np
import time
from scipy.signal import butter, lfilter
import random
import librosa
import soundfile as sf
import numpy as np
import time

def add_noise(data, noise_factor):
    # 백색 소음 (White Noise) 추가
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def change_pitch(data, sr, n_steps):
    # 피치 변경 (음정 높낮이 조절: 남성/여성/어린이 목소리 시뮬레이션)
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def change_speed(data, speed_factor):
    # 재생 속도 변경 (말하는 속도 조절)
    return librosa.effects.time_stretch(y=data, rate=speed_factor)

def bandpass_filter(data, sr, lowcut=80.0, highcut=10000.0, order=5):
    # [Architecture-First] 사람 목소리 및 실제 마이크 특성을 반영한 대역통과 필터
    # 80Hz 이하 저음역 및 10kHz 이상 고주파 대역 차단
    nyquist = 0.5 * sr
    
    # 목표 주파수가 Nyquist 주파수보다 높으면 Nyquist 한계로 조정 (클리핑 방지)
    high = min(highcut, nyquist * 0.99)
    low = lowcut / nyquist
    high = high / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def main():
    base_dir = r"c:\Users\User\KWS-DS-CNN-for-embedded\for_dataset\recordings"
    input_dir = os.path.join(base_dir, "quiz")
    output_dir = os.path.join(base_dir, "quiz_augmented1")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 원본 파일 목록 불러오기
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    if not files:
        print("[오류] quiz 폴더에 원본 파일이 없습니다.")
        return
        
    print(f"원본 파일 수: {len(files)}개")
    
    target_count = 2500
    current_count = 0
    
    print(f"\n데이터 증강 (Data Augmentation) 시작... 목표: {target_count}개")
    start_time = time.time()
    
    for i in range(target_count):
        # 무작위로 원본 파일 중 하나 선택
        original_file = random.choice(files)
        file_path = os.path.join(input_dir, original_file)
        
        # 오디오 로드 (샘플률 유지를 위해 sr=None 설정)
        data, sr = librosa.load(file_path, sr=None)
        augmented_data = data.copy()
        
        # 1. 50% 확률로 피치(Pitch) 변경 (-4 ~ +4 반음. 마이너스=낮은목소리/남성, 플러스=높은목소리/여성,어린이)
        if random.random() < 0.5:
            n_steps = random.uniform(-4, 4)
            augmented_data = change_pitch(augmented_data, sr, n_steps)
            
        # 2. 50% 확률로 속도(Speed) 변경 (0.8배 ~ 1.2배 속도)
        if random.random() < 0.5:
            speed_factor = random.uniform(0.8, 1.2)
            augmented_data = change_speed(augmented_data, speed_factor)
            
        # 3. 30% 확률로 노이즈(Noise) 추가 (배경 소음 시뮬레이션)
        if random.random() < 0.3:
            noise_factor = random.uniform(0.001, 0.015)
            augmented_data = add_noise(augmented_data, noise_factor)
            
        # 4. [중요] 사람 목소리 대역(80Hz ~ 10kHz) 필터 적용하여 실제 마이크 환경 모사
        augmented_data = bandpass_filter(augmented_data, sr)
            
        # 데이터 길이 정규화 방지 및 클리핑 (소리가 깨지는 것 방지)
        augmented_data = np.clip(augmented_data, -1.0, 1.0)
            
        # 결과 저장
        new_filename = f"aug_{i:04d}_{original_file}"
        output_path = os.path.join(output_dir, new_filename)
        
        sf.write(output_path, augmented_data, sr)
        current_count += 1
        
        if current_count % 100 == 0:
            print(f"진행 상황: {current_count} / {target_count} 완료...")
            
    end_time = time.time()
    print(f"\n✅ 데이터 증강 완료! 총 {target_count}개의 파일이 '{output_dir}'에 생성되었습니다.")
    print(f"소요 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    main()
