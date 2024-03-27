import librosa
import librosa.display
import matplotlib.pyplot as plt

# 오디오 파일 로드
y, sr = librosa.load('./F_AcusticPlug26_1.wav')

# MFCC 계산
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # n_mfcc는 추출할 MFCC 계수의 개수

# 시각화
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()