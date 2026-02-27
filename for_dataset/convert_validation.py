import os
import shutil
import glob
import numpy as np
import soundfile as sf

def convert_pcm_to_wav(pcm_path, wav_path, sr=16000, channels=1):
    # AI Hub PCM format is typically 16-bit integer, 16000Hz, Mono
    with open(pcm_path, 'rb') as f:
        pcm_data = f.read()
    
    # Convert binary raw PCM to numpy array
    audio_data = np.frombuffer(pcm_data, dtype=np.int16)
    
    # Write to WAV using soundfile
    sf.write(wav_path, audio_data, sr, subtype='PCM_16')

if __name__ == '__main__':
    base_dir = r'c:\Users\User\KWS-DS-CNN-for-embedded\speech_dataset'
    old_folder = os.path.join(base_dir, 'extracted_keywords')
    new_folder = os.path.join(base_dir, 'for_validation_from_AIHUB')
    
    # Rename folder if it exists under the old name
    if os.path.exists(old_folder):
        os.rename(old_folder, new_folder)
        print(f"Renamed folder to: {new_folder}")
    elif not os.path.exists(new_folder):
        print(f"Error: Neither {old_folder} nor {new_folder} exists.")
        exit(1)
        
    pcm_files = glob.glob(os.path.join(new_folder, '*.pcm'))
    print(f"Found {len(pcm_files)} PCM files to convert.")
    
    for pcm_path in pcm_files:
        wav_path = pcm_path.replace('.pcm', '.wav')
        
        try:
            convert_pcm_to_wav(pcm_path, wav_path)
            # Remove the original PCM file after successful conversion
            os.remove(pcm_path)
        except Exception as e:
            print(f"Failed to convert {pcm_path}: {e}")
            
    print("PCM to WAV conversion completed successfully.")
