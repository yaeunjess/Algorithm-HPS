import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)
file = "./Am_AcusticPlug26_2.wav"

sig, sr = librosa.load(file, sr=22050)

# STFT -> spectrogram
hop_length = 512  # 전체 frame 수, hop 길이는 프레임 사이의 거리를 의미
n_fft = 2048  # frame 하나당 sample 수, STFT에서 사용할 FFT 윈도우 크기, FFT 윈도우 크기는 한 번에 계산할 샘플의 수를 의미

# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/sr
n_fft_duration = float(n_fft)/sr

# STFT
stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)

# 복소공간 값 절댓값 취하기
magnitude = np.abs(stft)


# 진폭이 30dB 이상이고, 82Hz보다 크고 500Hz보다 작은 주파수 출력
threshold_db = 30
threshold_amp = librosa.db_to_amplitude(threshold_db)
frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# 중복되지 않는 정수 주파수와 해당 주파수의 진폭을 저장할 변수
unique_int_frequencies = set()

for i, frame in enumerate(magnitude.T):
    # 프레임에서 최대 진폭 값 계산
    max_amp = np.max(frame)

    # 최대 진폭 값이 지정한 threshold_amp 이상인 경우 처리
    if max_amp >= threshold_amp:
        # threshold_amp 이상인 값의 인덱스 추출
        max_indices = np.where(frame >= threshold_amp)[0]

        # 인덱스에 해당하는 주파수 및 진폭 확인
        for idx in max_indices:
            frequency = frequencies[idx]

            # 주파수가 82Hz보다 크고 500Hz보다 작은 경우 처리
            if 82 < frequency < 500:
                integer_part = int(frequency)
                amplitude = frame[idx]

                # 중복되지 않는 정수 주파수인 경우 출력
                if integer_part not in unique_int_frequencies:
                    unique_int_frequencies.add(integer_part)
                    print(f"주파수: {integer_part} Hz, 진폭: {amplitude} dB")



# magnitude > Decibels 
log_spectrogram = librosa.amplitude_to_db(magnitude)

# display spectrogram
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")

# 주파수 눈금 조정
frequencies = np.arange(0, 500, 100)  # 0부터 Nyquist 주파수까지 1000Hz 간격으로
frequency_labels = [f"{freq/1000} kHz" for freq in frequencies]  # 레이블을 kHz 단위로 표시
plt.yticks(frequencies, frequency_labels)

plt.show()