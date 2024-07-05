import os
import librosa
import soundfile as sf
from spleeter.separator import Separator
from tqdm import tqdm
from pydub import AudioSegment
import torch
from demucs import pretrained
from demucs.apply import apply_model
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# OpenMP 경고 억제
os.environ['OMP_NUM_THREADS'] = '1'

# 경로 설정
test_data_folder = r"/Users/iseongjun/Downloads/open/test_split"
output_folder = r"/Users/iseongjun/Downloads/open/test_split/separated_output"

# Wav2Vec 모델 로드
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def convert_ogg_to_wav(input_path, output_path):
    audio = AudioSegment.from_ogg(input_path)
    audio.export(output_path, format="wav")

def demucs_separate(input_path):
    model = pretrained.get_model('mdx_extra')
    model.cpu()
    wav, sr = librosa.load(input_path, sr=44100, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    wav = torch.tensor(wav, dtype=torch.float32)
    sources = apply_model(model, wav[None, :], device='cpu')[0]
    vocals = sources[0].numpy()
    accompaniment = sources[1:].sum(0).numpy()
    return vocals, accompaniment, sr

def detect_voices(audio, sr):
    input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    words = transcription.split()
    if len(words) == 0:
        return 0
    elif len(words) < 10:
        return 1
    else:
        return 2

def separate_and_save_audio(test_data_folder, output_folder):
    separator = Separator('spleeter:2stems')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(test_data_folder), desc="Separating audio"):
        if filename.endswith('.ogg'):
            try:
                file_path = os.path.join(test_data_folder, filename)
                wav_path = file_path.replace('.ogg', '.wav')
                convert_ogg_to_wav(file_path, wav_path)

                # Demucs separation
                demucs_vocals, demucs_accompaniment, sr_demucs = demucs_separate(wav_path)

                # Spleeter separation
                separator.separate_to_file(wav_path, output_folder)
                spleeter_vocals_path = os.path.join(output_folder, filename.split('.')[0], 'vocals.wav')
                spleeter_accompaniment_path = os.path.join(output_folder, filename.split('.')[0], 'accompaniment.wav')
                spleeter_vocals, sr_spleeter = librosa.load(spleeter_vocals_path, sr=None)
                spleeter_accompaniment, _ = librosa.load(spleeter_accompaniment_path, sr=None)

                # Combine results (weighted average)
                vocals_combined = 0.7 * demucs_vocals[0] + 0.3 * spleeter_vocals
                accompaniment_combined = 0.7 * demucs_accompaniment[0] + 0.3 * spleeter_accompaniment

                # Detect number of voices using Wav2Vec
                num_voices = detect_voices(vocals_combined, sr_spleeter)

                # Save results based on number of detected voices
                if num_voices == 0:  # 음성이 없는 경우
                    sf.write(os.path.join(output_folder, f"{filename.split('.')[0]}_no_voice.wav"), vocals_combined, sr_spleeter)
                elif num_voices == 1:  # 음성이 하나인 경우 (더블로 처리)
                    mid_point = len(vocals_combined) // 2
                    sf.write(os.path.join(output_folder, f"{filename.split('.')[0]}_voice_1.wav"), vocals_combined[:mid_point], sr_spleeter)
                    sf.write(os.path.join(output_folder, f"{filename.split('.')[0]}_voice_2.wav"), vocals_combined[mid_point:], sr_spleeter)
                else:  # 음성이 두 개 이상인 경우 (싱글로 처리)
                    sf.write(os.path.join(output_folder, f"{filename.split('.')[0]}_single_voice.wav"), vocals_combined, sr_spleeter)

                # 소음 및 음성 파일 저장
                sf.write(os.path.join(output_folder, f"{filename.split('.')[0]}_vocals_combined.wav"), vocals_combined, sr_spleeter)
                sf.write(os.path.join(output_folder, f"{filename.split('.')[0]}_accompaniment_combined.wav"), accompaniment_combined, sr_spleeter)

                print(f"Processed {filename} - Detected {num_voices} voice(s)")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

if __name__ == "__main__":
    separate_and_save_audio(test_data_folder, output_folder)
