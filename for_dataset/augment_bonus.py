import os
import random
import numpy as np
import librosa
import soundfile as sf

def generate_custom_augmented_audio(input_files, output_folder, target_count=400, fn_type='normal', sr=16000, duration=1.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    target_length = int(sr * duration)
    print(f'Starting {fn_type} augmentation for {len(input_files)} files. Target count: {target_count}')

    for i in range(target_count):
        # 1. Randomly pick an input file
        in_path = random.choice(input_files)
        filename = os.path.basename(in_path)
        
        # 2. Load audio
        y, _ = librosa.load(in_path, sr=sr, mono=True)
        
        # 3. Apply Time Stretch (Speed) - 0.8x to 1.2x
        stretch_rate = random.uniform(0.8, 1.2)
        y_mod = librosa.effects.time_stretch(y, rate=stretch_rate)
        
        # 4. Apply Pitch Shift
        if fn_type == 'female':
            # 여자의 목소리처럼 높이려면 피치를 크게 올림 (+4 ~ +8 반음)
            pitch_shift = random.uniform(4.0, 8.0)
        else:
            # 일반 증강 (-2 ~ +2 반음)
            pitch_shift = random.uniform(-2.0, 2.0)
            
        y_mod = librosa.effects.pitch_shift(y_mod, sr=sr, n_steps=pitch_shift)
        
        # 5. Length Normalization (Zero Padding or Truncate)
        if len(y_mod) < target_length:
            padding = np.zeros(target_length - len(y_mod))
            y_final = np.concatenate((y_mod, padding))
        else:
            y_final = y_mod[:target_length]
            
        # 6. Add Background Noise (White Noise)
        noise_level = random.uniform(0.001, 0.015) 
        noise = np.random.randn(len(y_final))
        y_final = y_final + noise_level * noise
        y_final = np.clip(y_final, -1.0, 1.0) # 클리핑 방지
        
        # 7. Save output
        prefix = 'aug_female' if fn_type == 'female' else 'aug_self'
        out_filename = f'{prefix}_{i:04d}_{filename}'
        out_path = os.path.join(output_folder, out_filename)
        
        sf.write(out_path, y_final, sr, subtype='PCM_16')
        
        if (i+1) % 100 == 0:
            print(f'Processed {i+1}/{target_count} ({fn_type}) files...')
            
    print(f'Done! Successfully generated {target_count} ({fn_type}) files.')

if __name__ == "__main__":
    base_dir = r'c:\Users\User\KWS-DS-CNN-for-embedded\for_dataset\recordings\quiz'
    out_dir = r'c:\Users\User\KWS-DS-CNN-for-embedded\for_dataset\recordings\quiz_augmented_v2'

    target_files = [
        os.path.join(base_dir, '녹음 (2).wav'),
        os.path.join(base_dir, '녹음 (3).wav'),
        os.path.join(base_dir, '녹음 (4).wav'),
        os.path.join(base_dir, '녹음 (5).wav'),
        os.path.join(base_dir, '녹음 (6).wav')
    ]

    # 1. 일반 사용자 본인 목소리 증강 400개
    generate_custom_augmented_audio(target_files, out_dir, target_count=400, fn_type='normal')

    # 2. 피치업(여성 목소리화) 증강 200개
    generate_custom_augmented_audio(target_files, out_dir, target_count=200, fn_type='female')
