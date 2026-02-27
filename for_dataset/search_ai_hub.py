import os
import json
import shutil
import re

def search_and_extract(search_dirs, keywords, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'Starting search for keywords: {keywords}')
    pattern = re.compile('|'.join(keywords))
    found_count = 0

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
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f'Error reading {json_path}: {e}')
                    continue
                
                has_keyword = False
                matched_sentences = []
                for dialog in data.get('dialogs', []):
                    text = dialog.get('text', '')
                    if pattern.search(text):
                        has_keyword = True
                        matched_sentences.append(text)
                
                if has_keyword:
                    # Find matching audio
                    rel_path = os.path.relpath(root, label_dir)
                    # For audio, rename TL14 to TS14, VL14 to VS14
                    audio_rel_path = rel_path.replace('TL', 'TS').replace('VL', 'VS')
                    audio_filename = file.replace('.json', '.pcm')
                    audio_path = os.path.join(audio_dir, audio_rel_path, audio_filename)
                    
                    if os.path.exists(audio_path):
                        # Copy
                        dest_json = os.path.join(output_dir, file)
                        dest_wav = os.path.join(output_dir, audio_filename)
                        shutil.copy2(json_path, dest_json)
                        shutil.copy2(audio_path, dest_wav)
                        print(f'==== Found & Copied: {file} ====')
                        for s in matched_sentences:
                            print(f'  -> {s}')
                        found_count += 1
                    else:
                        print(f'==== Found JSON but missing audio: {audio_path} ====')

    print(f'\nTotal {found_count} matching audio files successfully separated into {output_dir}')

if __name__ == '__main__':
    dirs = [
        r'c:\Users\User\KWS-DS-CNN-for-embedded\speech_dataset\AI_hub_Training',
        r'c:\Users\User\KWS-DS-CNN-for-embedded\speech_dataset\AI_hub_Validation'
    ]
    kws = ['퀴즈', '이해했어요', '이해했나요', '이해됐나요']
    out_dir = r'c:\Users\User\KWS-DS-CNN-for-embedded\speech_dataset\extracted_keywords'
    search_and_extract(dirs, kws, out_dir)
