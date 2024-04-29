# Project name: Polyphonic note detector using Harmonic Product Spectrum
# Date:         2021.05.19
# Author:       João Nuno Carvalho
# Description:  This is my implementation of a polyphonic note detector using
#               the Harmonic Product Spectrum method.
#               The input is a mono WAV file.
#               The output are the corresponding notes in time. 
# License: MIT Open Source License

import numpy as np
import matplotlib.pyplot as plt
import wave
import math

path     = "./"
filename = 'G-DDDD.wav' 

note_threshold = 5_000.0    # 120   # 50_000.0   #  3_000.0

# Parameters
sample_rate  = 44100                     # Sampling Frequency
fft_len      = 10000  # 22050   # 2048            # Length of the FFT window
overlap      = 0.5                       # Hop overlap percentage between windows
hop_length   = int(fft_len*(1-overlap))  # Number of samples between successive frames

# For the calculations of the music scale.
TWELVE_ROOT_OF_2 = math.pow(2, 1.0 / 12)

## wav 파일 읽은 후, sample_rate와 input_buffer 반환
# (sample_rate : int, input_buffer : NDArray[Any]) 반환, NDArray[Any]는 실수값의 numpy 배열을 의미
def read_wav_file(path, filename): 
    # Reads the input WAV file from HDD disc.
    wav_handler = wave.open(path + filename,'rb')    # 지정된 경로에 wav 파일을 읽기 전용 모드로 연다.
    num_frames = wav_handler.getnframes()            # 파일에서 sample의 총 개수를 얻는다. 44100*(wav 길이 예로 4초) = 176400개
    sample_rate = wav_handler.getframerate()         # 파일의 sample_rate를 얻는다. 
                                                     # sampling rate는 초당 샘플링 횟수를 의미하며, 단위는 Hz이다. wav파일의 헤더 부분에 이 정보가 포함되어 있다.
    wav_frames = wav_handler.readframes(num_frames)  # 모든 frame을 읽는다. wav_frames는 num_frames의 두 배이다. 각 샘플이 2바이트로 표현되기 때문. wav_frames의 바이트 배열 길이는 176400*2 = 352800 바이트이다.

    # Loads the file into a NumPy contiguous array.
    # WAV 파일의 프레임을 numpy 배열로 변환합니다.
    # Convert Int16 into float64 in the range of [-1, 1].
    # This means that the sound pressure values are mapped to integer values that can range from -2^15 to (2^15)-1.
    # We can convert our sound array to floating point values ranging from -1 to 1 as follows.
    signal_temp = np.frombuffer(wav_frames, np.int16) # 읽은 wav_frame 데이터를 numpy 배열로 변환한다. 데이터 타입은 int16이다. 
    signal_array = np.zeros(len(signal_temp), float) # wav_frames로부터 생성된 numpy 배열이다. 신호를 저장할 float 타입의 numpy 배열을 생성한다.

    for i in range(0, len(signal_temp)):
        signal_array[i] = signal_temp[i] / (2.0**15) # int16 타입의 값을 [-1, 1] 범위의 float64 타입으로 변환합니다.

    print("------------------------------")
    print("file_name: " + str(filename))
    print("sample_rate: " + str(sample_rate) + " Hz")
    print("input_buffer.size(sample의 총 갯수): " + str(len(signal_array)) + " 개")
    print("seconds(input_buffer.size/sample_rate): " + to_str_f4(len(signal_array)/sample_rate) + " s")
    print("type [-1, 1]: " + str(signal_array.dtype))
    print("min: " + to_str_f4(np.min(signal_array)) + " max: " + to_str_f4(np.max(signal_array))  )

    # sample_rate int형 숫자와 input_buffer numpy float 배열 반환
    return sample_rate, signal_array 

## chunk 나눈 후, 나누어진 chunk의 리스트 반환
# (나누어진 청크들의 리스트 : list[NDArray]) 반환
# buffer의 총 sample 수 = wav 파일 seconds * fft_len
def divide_buffer_into_non_overlapping_chunks(buffer, max_len): # max_len -> fft_len
    buffer_len = len(buffer)                  # input_buffer의 길이 계산, buffer에 총 몇개의 sample이 있는지를 반환
    chunks = int(buffer_len / max_len)        # input_buffer 길이를 fft_len으로 나누어 몇 개의 chunk로 나눌 수 있는지 계산

    division_pts_list = []                    # chunk를 나눌 지점을 저장할 리스트
    for i in range(1, chunks):
        division_pts_list.append(i * max_len) # 각 청크의 시작 지점을 계산하여 리스트에 추가, fft_len의 배수가 리스트에 추가됨 -> [22050, 44100, 66150, ..]
    splitted_array_view = np.split(buffer, division_pts_list, axis=0) # 계산된 지점을 기준으로 버퍼를 나눈다.
    
    print("------------------------------")
    print("buffers_num: " + str(chunks))      # 총 chunks의 개수를 출력
    print("나누어진 chunk들에 대한 리스트 :", splitted_array_view)
    # 나누어진 청크들의 리스트를 반환, list[NDArray]
    return splitted_array_view                

## fft 연산 후, frequency 배열과 magnitude 배열과 frequency 개수(대칭적인 rfft 이용) 반환
# (frequency 배열 : NDArray[floating[Any]], magnitude 배열 : NDArray[Any], frequency 개수 : int) 반환
def getFFT(data, rate):
    # Returns fft_freq and fft, fft_res_len.
    len_data = len(data)                # 입력 데이터의 길이 계산
    data = data * np.hamming(len_data)  # 입력 데이터에 hamming_window를 적용하여 스펙트럼의 누설을 감소

    # fft 연산 후 magnitude 배열
    fft = np.fft.rfft(data)             # 입력 데이터에 대해 실수 fft를 수행한다. fft의 결과로 복소수 numpy 배열을 반환한다.
                                        # rfft는 수행하면 양수, 음수의 대칭적이므로 양수만을 출력한다.
    fft = np.abs(fft)                   # fft 결과의 절대값을 취하여 magnitue를 얻는다. 실수값의 numpy 배열을 반환한다.

    # fft 연산 후 frequency 개수 
    ret_len_FFT = len(fft)              # fft 결과의 길이를 저장한다. 배열의 원소 개수, 즉 fft 변환을 통해 분석된 주파수 성분의 개수를 반환한다.
    
    # fft 연산 후 frequency 배열
    freq = np.fft.rfftfreq(len_data, 1.0 / sample_rate) # fft 결과에 대응하는 주파수 배열을 계산한다.
    # return ( freq[:int(len(freq) / 2)], fft[:int(ret_len_FFT / 2)], ret_len_FFT )
   
    print("--------------getFFT() 거친 후----------------")
    print("fft 연산 후 magnitude 배열 :", fft)
    print("fft 연산 후 frequency 개수 :", ret_len_FFT)
    print("fft 연산 후 frequency 배열 :", freq)
    # (frequency 배열 : NDArray[floating[Any]], magnitude 배열 : NDArray[Any], frequency 개수 : int) 반환
    return (freq, fft, ret_len_FFT) 

## fft 결과에 DC offset을 제거한 magnitude를 반환
def remove_dc_offset(fft_res):
    # Removes the DC offset from the FFT (First bin's)
    fft_res[0] = 0.0
    fft_res[1] = 0.0
    fft_res[2] = 0.0
    return fft_res

def freq_for_note(base_note, note_index):
    # See Physics of Music - Notes
    #     https://pages.mtu.edu/~suits/NoteFreqCalcs.html
    
    A4 = 440.0

    base_notes_freq = {"A2" : A4 / 4,   # 110.0 Hz
                       "A3" : A4 / 2,   # 220.0 Hz
                       "A4" : A4,       # 440.0 Hz
                       "A5" : A4 * 2,   # 880.0 Hz
                       "A6" : A4 * 4 }  # 1760.0 Hz  

    scale_notes = { "C"  : -9.0,
                    "C#" : -8.0,
                    "D"  : -7.0,
                    "D#" : -6.0,
                    "E"  : -5.0,
                    "F"  : -4.0,
                    "F#" : -3.0,
                    "G"  : -2.0,
                    "G#" : -1.0,
                    "A"  :  1.0,
                    "A#" :  2.0,
                    "B"  :  3.0,
                    "Cn" :  4.0}

    scale_notes_index = list(range(-9, 5)) # Has one more note.
    note_index_value = scale_notes_index[note_index]
    freq_0 = base_notes_freq[base_note]
    freq = freq_0 * math.pow(TWELVE_ROOT_OF_2, note_index_value) 
    return freq

def get_all_notes_freq():
    ordered_note_freq = []
    ordered_notes = ["C",
                     "C#",
                     "D",
                     "D#",
                     "E",
                     "F",
                     "F#",
                     "G",
                     "G#",
                     "A",
                     "A#",
                     "B"]
    for octave_index in range(2, 7):
        base_note  = "A" + str(octave_index)
        # note_index = 0  # C2
        # note_index = 12  # C3
        for note_index in range(0, 12):
            note_freq = freq_for_note(base_note, note_index)
            note_name = ordered_notes[note_index] + "_" + str(octave_index)
            ordered_note_freq.append((note_name, note_freq))
    return ordered_note_freq

def find_nearest_note(ordered_note_freq, freq):
    final_note_name = 'note_not_found'
    last_dist = 1_000_000.0
    for note_name, note_freq in ordered_note_freq:
        curr_dist = abs(note_freq - freq)
        if curr_dist < last_dist:
            last_dist = curr_dist
            final_note_name = note_name
        elif curr_dist > last_dist:
            break    
    return final_note_name

# 기타 조에 대한 딕셔너리 생성
def get_all_key_freq():
    keys_freq = {
        "A": ['A_2', 'E_3', 'E4'],
        "B": ['F#_2', 'B_2', 'F#_3', 'F#_4'],
        "C": ['C_3', 'E_3', 'C_4', 'E_4'],
        "D": ['D_3', 'A_3'],
        "E": ['E_2', 'B_2', 'B_3', 'E_4'],
        "F": ['F_2', 'C_3', 'C_4', 'F_4'],
        "G": ['G_2', 'B_2', 'D_3', 'G_3', 'B_3']
    }
    return keys_freq

# 기타 코드에 대한 딕셔너리 생성
def get_all_guitar_chords_freq():
    guitar_chords_freq = {
        "A": [110.00, 164.81, 220.00, 277.18, 329.63],
        "Am": [110.00, 164.81, 220.00, 261.63, 329.63],
        "A7": [110.00, 164.81, 196.00, 277.18, 329.63],
        "B": [92.50, 123.47, 185.00, 246.94, 311.13, 369.99],
        "Bm": [92.50, 123.47, 185.00, 246.94, 293.66, 369.99],
        "B7": [92.50, 123.47, 185.00, 220.00, 311.13, 369.99],
        "C": [130.81, 164.81, 196.00, 261.63, 329.63],
        "C7": [130.81, 164.81, 233.08, 261.63, 329.63],
        "D": [146.83, 220.00, 293.66, 369.99],
        "Dm": [146.83, 220.00, 293.66, 349.23],
        "D7": [146.83, 220.00, 261.63, 369.99],
        "E": [82.41, 123.47, 164.81, 207.65, 246.94, 329.63],
        "Em": [82.41, 123.47, 164.81, 196.00, 246.94, 329.63],
        "E7": [82.41, 123.47, 146.83, 207.65, 246.94, 329.63],
        "F": [87.31, 130.81, 174.61, 220.00, 261.63, 349.23],
        "Fm": [87.31, 130.81, 174.61, 207.65, 261.63, 349.23],
        "F7": [87.31, 130.81, 155.56, 220.00, 261.63, 349.23],
        "G": [98.00, 123.47, 146.83, 196.00, 246.94, 392.00],
        "G7": [98.00, 123.47, 146.83, 196.00, 246.94, 349.23]
    }
    return guitar_chords_freq 

# 기타 조 판단에 중요한 list 생성
def get_unique_key():
    unique_notes = ['A_2', 'E_3', 'E_4', 'F#2', 
                    'B_2', 'F#_3', 'F#_4', 'C_3', 
                    'C_4', 'D_3', 'A_3', 'E_2', 
                    'B_3', 'F_2', 'F_4', 'G_2', 'D_3', 'G_3']
    return unique_notes

# chunk별 상위 top_n개 주파수 뽑기
def get_top_frequencies(frequencies, top_n):
    top_freqs = sorted(frequencies, key=lambda x: x[1], reverse=True)[:top_n]
    return top_freqs

# 기타 조 판단에 해당하는 음 저장하기
def find_unique_notes(ordered_note_freq, top_freqs, unique_notes):
    found_notes = []
    
    # 주어진 상위 주파수들 중 unique_notes에 해당하는 음 찾기
    for freq in top_freqs:
        note_name = find_nearest_note(ordered_note_freq, freq[0])  # 주파수로부터 가장 가까운 음 찾기
        if note_name in unique_notes and note_name not in found_notes:  # unique_notes 목록에 해당하는지 확인
            found_notes.append(note_name)
    
    return found_notes

# 기타 조 추정
def find_nearest_key(found_notes, keys_freq):
    # found_notes가 비어있으면, 'null' 반환
    if not found_notes:
        return 'null'
    
    # 각 조와 found_notes 간의 일치도 계산
    best_match = None
    best_match_score = -1  # 일치하는 음의 개수를 저장할 변수

    for key, notes in keys_freq.items():
        match_score = sum(note in found_notes for note in notes)  # found_notes에 포함된 음의 개수를 계산
        
        if match_score > best_match_score:  # 현재 조가 이전 조보다 더 많은 일치를 가지면
            best_match = key  # 현재 조를 최고 일치로 업데이트
            best_match_score = match_score  # 최고 일치 점수 업데이트
    
    return best_match  # 가장 일치율이 높은 조 반환

# 기타 코드 추정 
def find_nearest_chord(guitar_chords_freq):
    return 0

def PitchSpectralHps(X, freq_buckets, f_s, buffer_rms):

    """
    NOTE: This function is from the book Audio Content Analysis repository
    https://www.audiocontentanalysis.org/code/pitch-tracking/hps-2/
    The license is MIT Open Source License.
    And I have modified it. Go to the link to see the original.

    computes the maximum of the Harmonic Product Spectrum

    Args:
        X: spectrogram (dimension FFTLength X Observations)
        f_s: sample rate of audio data

    Returns:
        f HPS maximum location (in Hz)
    """
    print("fft_res, HPS 들어간 후", "(",X.shape, ")",X)

    # initialize
    iOrder = 4
    f_min = 65.41   # C2      300
    # f = np.zeros(X.shape[1])
    f = np.zeros(len(X))

    iLen = int((X.shape[0] - 1) / iOrder)
    afHps = X[np.arange(0, iLen)]
    k_min = int(round(f_min / f_s * 2 * (X.shape[0] - 1)))

    # compute the HPS
    for j in range(1, iOrder):
        X_d = X[::(j + 1)]
        afHps *= X_d[np.arange(0, iLen)]

    ## Uncomment to show the original algorithm for a single frequency or note. 
    # f = np.argmax(afHps[np.arange(k_min, afHps.shape[0])], axis=0)
    ## find max index and convert to Hz
    # freq_out = (f + k_min) / (X.shape[0] - 1) * f_s / 2

    note_threshold = note_threshold_scaled_by_RMS(buffer_rms)

    all_freq = np.argwhere(afHps[np.arange(k_min, afHps.shape[0])] > note_threshold)
    # find max index and convert to Hz
    freqs_out = (all_freq + k_min) / (X.shape[0] - 1) * f_s / 2

    
    x = afHps[np.arange(k_min, afHps.shape[0])]
    freq_indexes_out = np.where( x > note_threshold)
    freq_values_out = x[freq_indexes_out]

    # print("\n##### x: " + str(x))
    # print("\n##### freq_values_out: " + str(freq_values_out))

    max_value = np.max(afHps[np.arange(k_min, afHps.shape[0])])
    max_index = np.argmax(afHps[np.arange(k_min, afHps.shape[0])])
    
    ## Uncomment to print the values: buffer_RMS, max_value, min_value
    ## and note_threshold.    
    print("--------------PitchSpectralHps() 거친 후----------------")
    print("buffer_rms: " + to_str_f4(buffer_rms) )
    print("max_value : " + to_str_f(max_value) + "  max_index : " + to_str_f(max_index) )
    print("note_threshold : " + to_str_f(note_threshold) )

    ## Uncomment to show the graph of the result of the 
    ## Harmonic Product Spectrum. 
    # fig, ax = plt.subplots()
    # yr_tmp = afHps[np.arange(k_min, afHps.shape[0])]
    # xr_tmp = (np.arange(k_min, afHps.shape[0]) + k_min) / (X.shape[0] - 1) * f_s / 2
    # ax.plot(xr_tmp, yr_tmp)
    # plt.show()

    # Turns 2 level list into a one level list.
    freqs_out_tmp = []
    for freq, value  in zip(freqs_out, freq_values_out):
        freqs_out_tmp.append((freq[0], value))
    
    return freqs_out_tmp

def note_threshold_scaled_by_RMS(buffer_rms):
    note_threshold = 1000.0 * (4 / 0.090) * buffer_rms
    return note_threshold

def normalize(arr):
    # Note: Do not use.
    # Normalize array between -1 and 1.
    # Only works if the signal is larger then the final signal and if the positive
    # value is grater in absolute value them the negative value.
    ar_res = (arr / (np.max(arr) / 2)) - 1  
    return ar_res

def to_str_f(value):
    # Returns a string with a float without decimals.
    return "{0:.0f}".format(value)

def to_str_f4(value):
    # Returns a string with a float without decimals.
    return "{0:.4f}".format(value)


def main():
    print("\nPolyphonic note detector\n")
    
    unique_notes = get_unique_key()
    keys_freq = get_all_key_freq()
    guitar_chords_freq = get_all_guitar_chords_freq()
    ordered_note_freq = get_all_notes_freq()
    # print(ordered_note_freq)

    sample_rate_file, input_buffer = read_wav_file(path, filename)
    buffer_chunks = divide_buffer_into_non_overlapping_chunks(input_buffer, fft_len)
    # The buffer chunk at n seconds:

    count = 0

    ## Uncomment to process a single chunk os a limited number os sequential chunks. 
    for chunk in buffer_chunks[0: 60]:
        print("\nChunk", str(count+1))
                
        fft_freq, fft_res, fft_res_len = getFFT(chunk, len(chunk))
        ### print("fft_res, getFFT 직후", "(",fft_res.shape, ")", fft_res)
        fft_res = remove_dc_offset(fft_res)
        ### print("fft_res, de offset 제거 직후","(",fft_res.shape, ")", fft_res)

        # Calculate Root Mean Square of the signal buffer, as a scale factor to the threshold.
        buffer_rms = np.sqrt(np.mean(chunk**2))

        all_freqs = PitchSpectralHps(fft_res, fft_freq, sample_rate_file, buffer_rms)
        # print(all_freqs)

        # get_top_frequencies 함수를 사용하여 상위 6개의 주파수를 선택
        top_freqs = get_top_frequencies(all_freqs, 6)
        print("top_freqs :", top_freqs)
        for freq in top_freqs:
            note_name = find_nearest_note(ordered_note_freq, freq[0])
            print("=> freq: " + to_str_f(freq[0]) + " Hz  value: " + to_str_f(freq[1]) + " note_name: " + note_name)

        # print("------------------------------")
        for freq in all_freqs:
            note_name = find_nearest_note(ordered_note_freq, freq[0])
            # print("=> freq: " + to_str_f(freq[0]) + " Hz  value: " + to_str_f(freq[1]) + " note_name: " + note_name)

        # 상위 6개의 주파수를 이용하여 가장 가까운 조를 찾기
        found_notes = find_unique_notes(ordered_note_freq, top_freqs, unique_notes)
        print(found_notes)

        nearest_key = find_nearest_key(found_notes, keys_freq)
        print(nearest_key)

        # 각 주파수에 대해 해당하는 기타 조 찾기
        # for freq, _ in top_freqs:
        #     key = find_key_for_freq(freq, keys_freq)
        #     print(f"freq: {freq:.2f} Hz -> key: {key}")

        count += 1

if __name__ == "__main__":
    main()