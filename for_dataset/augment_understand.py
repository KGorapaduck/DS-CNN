import os
import random
import numpy as np
import librosa
import soundfile as sf

def generate_augmented_audio(input_folder, output_folder, target_count=2500, sr=16000, duration=1.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    target_length = int(sr * duration)
    
    files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    if not files:
        print('No wav files in input directory!')
        return

    print(f'Found {len(files)} files. Starting augmentation to generate {target_count} samples...')

    for i in range(target_count):
        # 1. Randomly pick an input file
        filename = random.choice(files)
        in_path = os.path.join(input_folder, filename)
        
        # 2. Load audio (librosa automatically resamples to sr and converts to mono)
        y, _ = librosa.load(in_path, sr=sr, mono=True)
        
        # 3. Apply Time Stretch (Speed)
        # "이해하셨나요"는 길기 때문에 살짝 빠르게(0.9x)에서 매우 빠르게(1.3x) 사이로 왜곡
        stretch_rate = random.uniform(0.9, 1.3)
        y_fast = librosa.effects.time_stretch(y, rate=stretch_rate)
        
        # 4. Apply Pitch Shift - -2 to +2 semitones
        pitch_shift = random.uniform(-2.0, 2.0)
        y_pitch = librosa.effects.pitch_shift(y_fast, sr=sr, n_steps=pitch_shift)
        
        # 5. Length Normalization (1.0 sec = 16000 frames) -> Zero Padding or Truncate
        if len(y_pitch) < target_length:
            # Pad with zeros
            padding = np.zeros(target_length - len(y_pitch))
            y_final = np.concatenate((y_pitch, padding))
        else:
            # Truncate (앞 1초 구간만)
            y_final = y_pitch[:target_length]
            
        # 6. Add Background Noise (White Noise)
        # SNR roughly 10dB to 30dB
        noise_level = random.uniform(0.001, 0.015) 
        noise = np.random.randn(len(y_final))
        y_final = y_final + noise_level * noise
        
        # Ensure we don't clip
        y_final = np.clip(y_final, -1.0, 1.0)
        
        # 7. Save output
        out_filename = f'aug_{i:04d}_{filename}'
        out_path = os.path.join(output_folder, out_filename)
        
        # soundfile automatically converts back to 16bit PCM based on subtype
        sf.write(out_path, y_final, sr, subtype='PCM_16')
        
        if (i+1) % 500 == 0:
            print(f'Processed {i+1}/{target_count} files...')
            
    print(f'Successfully generated {target_count} augmented 1-sec files in {output_folder}')

if __name__ == '__main__':
    understand_in = r'c:\Users\User\KWS-DS-CNN-for-embedded\for_dataset\recordings\understand'
    understand_out = r'c:\Users\User\KWS-DS-CNN-for-embedded\for_dataset\recordings\understand_augmented'
    generate_augmented_audio(understand_in, understand_out, target_count=2500)
