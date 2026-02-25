import soundcard as sc
import soundfile as sf
import keyboard
import numpy as np
import os
import time
from datetime import datetime

def main():
    # 현재 스크립트 파일 위치를 기준으로 저장 폴더 생성 (Create save folder based on current path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "recordings")
    os.makedirs(output_dir, exist_ok=True)

    print("[시스템 안내] 윈도우 루프백(Windows Loopback) 스피커 출력 캡처를 준비합니다.")

    # 기본 스피커 장치 및 스피커 루프백(Loopback) 마이크 객체 생성
    default_speaker = sc.default_speaker()
    
    try:
        # 시스템에서 나오는 소리를 그대로 캡처하는 루프백 설정
        loopback_mic = sc.get_microphone(default_speaker.id, include_loopback=True)
    except Exception as e:
        print(f"[오류 발생] 루프백 장치를 찾을 수 없습니다: {e}")
        return

    sample_rate = 48000 # 고음질 (High Quality) PCM 표준 샘플링 레이트
    
    print("="*50)
    print("🎙️ 웹 오디오 확장 녹음기 준비 완료!")
    print(f"📂 저장 위치: {output_dir}")
    print("▶️ [F9] 키: 녹음 시작 및 중지 (Toggle)")
    print("⏹️ [ESC] 키: 프로그램 종료")
    print("="*50)

    while True:
        # F9 또는 ESC 입력 대기
        while not keyboard.is_pressed('f9'):
            if keyboard.is_pressed('esc'):
                print("프로그램을 안전하게 종료합니다.")
                return
            time.sleep(0.01)
            
        # 디바운스 (Debounce): 키보드를 꾹 누르고 있을 때 여러 번 인식되는 것 방지
        while keyboard.is_pressed('f9'):
            pass

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"web_record_{now}.wav")
        print(f"\n🔴 녹음 중... (저장 예정 파일: {os.path.basename(filename)})")
        print("   종료하려면 [F9] 키를 다시 누르세요.")

        record_data = [] # 오디오 데이터를 담을 리스트 (List to store audio chunks)
        
        # 루프백 마이크를 통해 시스템 소리 녹음 진행
        with loopback_mic.recorder(samplerate=sample_rate) as mic:
            while True:
                # 0.1초 단위(sample_rate // 10)로 데이터 캡처하여 키보드 응답성 확보
                data = mic.record(numframes=sample_rate // 10)
                record_data.append(data)
                
                if keyboard.is_pressed('f9'):
                    while keyboard.is_pressed('f9'):
                        pass
                    break # 녹음 중지
                
                if keyboard.is_pressed('esc'):
                    print("\n[경고] 프로그램을 강제 종료합니다. (현재 진행된 녹음은 저장되지 않습니다)")
                    return

        print("데이터 정리 중... 잠시만 기다려주세요.")
        
        # 저장된 데이터 조각들을 하나의 numpy 배열로 병합 (Merge array)
        audio_data = np.concatenate(record_data, axis=0)
        
        # 비압축 고음질 WAV 파일로 저장 (Save as uncompressed high-quality WAV)
        sf.write(filename, audio_data, sample_rate)
        
        print(f"✅ 저장 완료: {filename}")
        print("▶️ 새로운 녹음을 하려면 다시 [F9] 키를 누르세요. (완전 종료: [ESC])")

if __name__ == "__main__":
    main()
