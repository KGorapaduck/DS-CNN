import os
import json
import random
import glob
import numpy as np
import soundfile as sf
import librosa
import re

def create_unknown_dataset(search_dirs, target_count, output_dir, keywords_to_avoid, sr=16000, duration=1.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'Starting extraction of {target_count} "unknown" audio samples.')
    print(f'Omitting files containing keywords: {keywords_to_avoid}')
    
    avoid_pattern = re.compile('|'.join(keywords_to_avoid))
    target_length = int(sr * duration)
    
    # Collect all candidate files (valid json + audio pairs)
    candidate_pairs = []
    
    for base_dir in search_dirs:
        label_dir = os.path.join(base_dir, '라벨링데이터_0825_add')
        audio_dir = os.path.join(base_dir, '원천데이터_0825_add')

        if not os.path.exists(label_dir):
            continue

        for root, _, files in os.walk(label_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue
                
                json_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, label_dir)
                audio_rel_path = rel_path.replace('TL', 'TS').replace('VL', 'VS')
                audio_filename = file.replace('.json', '.pcm')
                audio_path = os.path.join(audio_dir, audio_rel_path, audio_filename)
                
                if os.path.exists(audio_path):
                    candidate_pairs.append((json_path, audio_path))

    print(f'Total candidates found: {len(candidate_pairs)}')
    random.shuffle(candidate_pairs)
    
    extracted_count = 0
    
    for json_path, audio_path in candidate_pairs:
        if extracted_count >= target_count:
            break
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
            
        # Check if the file contains forbidden keywords
        has_forbidden_keyword = False
        dialogs = data.get('dialogs', [])
        for dialog in dialogs:
            text = dialog.get('text', '')
            if avoid_pattern.search(text):
                has_forbidden_keyword = True
                break
                
        if has_forbidden_keyword:
            continue
            
        # File is safe, select a random dialog piece to extract audio from
        if not dialogs:
            continue
            
        # Find a dialog part that is long enough (we relax the duration check to rely on padding if too short)
        valid_dialogs = []
        for d in dialogs:
            try:
                start = float(d.get('startTime', 0))
                end = float(d.get('endTime', 0))
                if end - start >= float(duration * 0.4): # Relax duration: accept any utterance longer than 0.4s and just pad the rest
                    valid_dialogs.append((start, end))
            except (ValueError, TypeError):
                continue
                
        if not valid_dialogs:
            continue
            
        # Read PCM and process audio
        try:
            with open(audio_path, 'rb') as f:
                pcm_data = f.read()
            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Choose a random valid dialog interval
            start_sec, end_sec = random.choice(valid_dialogs)
            
            # We want exactly 1 second (duration). Pick a random start within [start_sec, end_sec - duration]
            max_start = end_sec - duration
            if max_start <= start_sec:
                crop_start_sec = start_sec
            else:
                crop_start_sec = random.uniform(start_sec, max_start)
                
            start_idx = int(crop_start_sec * sr)
            end_idx = start_idx + target_length
            
            cropped_audio = audio_array[start_idx:end_idx]
            
            # Padding if slightly short due to end of file boundaries
            if len(cropped_audio) < target_length:
                padding = np.zeros(target_length - len(cropped_audio), dtype=np.float32)
                cropped_audio = np.concatenate((cropped_audio, padding))
            else:
                cropped_audio = cropped_audio[:target_length]
            
            out_filename = f'unknown_{extracted_count:04d}_{os.path.basename(audio_path).replace(".pcm",".wav")}'
            out_path = os.path.join(output_dir, out_filename)
            
            sf.write(out_path, cropped_audio, sr, subtype='PCM_16')
            extracted_count += 1
            
            if extracted_count % 100 == 0:
                print(f'Extracted {extracted_count}/{target_count} unknown samples...')
                
        except Exception as e:
            # Skip corrupted audio files
            continue

    print(f'\nJob finished. Successfully generated {extracted_count} 1-second WAV files in {output_dir}')


if __name__ == '__main__':
    dirs = [
        r'c:\Users\User\KWS-DS-CNN-for-embedded\speech_dataset\AI_hub_Training',
        r'c:\Users\User\KWS-DS-CNN-for-embedded\speech_dataset\AI_hub_Validation'
    ]
    # We omit any file that even remotely looks like our targets
    forbidden_kws = ['퀴즈', '이해했어요', '이해했나요', '이해됐나요', '이해'] 
    
    # We aim for ~4000 unknown samples (slightly more than the 3100 target samples we have to prevent false positives)
    out_dir = r'c:\Users\User\KWS-DS-CNN-for-embedded\speech_dataset\unknown'
    create_unknown_dataset(dirs, 4000, out_dir, forbidden_kws)
