"""Audio File Dataset / Dataloader"""
import os
import math
import random
import numpy as np
import scipy.signal
import librosa
import librosa.display as dispaly
import json
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

def load_audio(path):
    sound = np.memmap(path, dtype='h', mode='r')
    # print("sound: ", sound.shape)#: (17430, ) |(17942, ) 길이별로 정리해놓음 (사운드의 길이) 최대 50만 정도까지 존재
    # np.memmap: 디스크에 binary file로 저장된 array 에 대한 memory-map 생성 / 'h': short = int16 int 범위일치
    # 원래 하면 (220, ) 에서 'h' 추가 시 (110, ) 즉 1/2배

    sound = sound.astype('float32') / 32767
    # 32767: int16 범위 (-32768, 32767)
    # normalized 목적[-1.0, 1.0]
    # -1~1 사이로 옮김

    assert len(sound) # 17430 | 17942

    sound = torch.from_numpy(sound).view(-1, 1).type(torch.FloatTensor)
    # from_numpy: ndarray -> tensor (ndarray: 다차원 array)
    # print("sound(tensor): ", sound.size()): [length, 1]

    sound = sound.numpy() # todo 왜 Tensor로 가서 view하고 numpy로 다시 오는 지??
    # print("sound.shape: ", sound.shape): [length, 1]
    if len(sound.shape) > 1: # 2
        if sound.shape[1] == 1:
            sound = sound.squeeze()
            # print("squeeze", sound.shape) # 다시 원래대로 복구(76430, ) todo 왜 다시 원래 사이즈로 옮기는지
        else:
            sound = sound.mean(axis=1) # 여러 채널 평균
            print("sound.mean: ", sound) # todo 이런경우는 아직 없는것 같음
            # numpy 에서 axis=1
            # [[3,2,1],
            #  [4,5,6]] ==> [2., 5.] 으로 나옴

    return sound

class SpectrogramDataset(Dataset):

    def __init__(self, audio_conf, dataset_path, data_list, char2index, sos_id, eos_id, normalize=False, mode='train'):
        """
        Dataset 은 wav_name, transcripts, speaker_id 가 dictionary 로 담겨져있는 list으로부터 data 를 load
        :param audio_conf: Sample rate, window, window size나 length, stride 설정
        :param data_list: dictionary . key: 'wav', 'text', 'speaker_id'
        :param char2index: character 에서 index 로 mapping 된 Dictionary
        :param normalize: Normalized by instance-wise standardazation
        """
        super(SpectrogramDataset, self).__init__()
        self.audio_conf = audio_conf # dict{sample rate, window_size, window_stride}
        self.data_list = data_list # [{"wav": , "text": , "speaker_id": "}]
        self.size = len(self.data_list) # 59662
        self.char2index = char2index
        self.sos_id = sos_id # 2001
        self.eos_id = eos_id # 2002
        self.PAD = 0
        self.normalize = normalize # Train: True
        self.dataset_path = dataset_path # data/wavs_train
        self.mode = mode

    def __getitem__(self, index):
        wav_name = self.data_list[index]['wav']
        # print("wav: " , wav_name): 41_0607_213_1_08139_05.wav
        audio_path = os.path.join(self.dataset_path, wav_name)
        # print("audio_path: ", audio_path): data/wavs_train/41_0607_213_1_08139_05.wav
        transcript = self.data_list[index]['text']
        # print("text: ", transcript): 예약 받나요?

        spect = self.parse_audio(audio_path)
        # print("spect: ", spect.size()): Tensor[161, Frame] 161: 1+ n_fft/2 | Frame: length / stride
        transcript = self.parse_transcript(transcript)
        # print("text: ", transcript): [2001, 22, 3, ..., 2002] 각 단어의 인덱스 2001(sos)

        return spect, transcript

    def parse_audio(self, audio_path):
        # print(audio_path)
        y = load_audio(audio_path)
        # print("y shape", y.shape)
        # plt.figure(figsize=(15, 10))
        # plt.xlabel("Index")
        # plt.ylabel("Amp")
        # plt.title("Original")
        # plt.plot(y)
        # plt.show()
        ############### NOISE INJECTION################
        # noise = np.random.random(y.shape) / 75
        # y = y + noise
        ##############################################

        ############### Pitch_Shift####################
        # y = librosa.effects.pitch_shift(y, sr=16000, n_steps=4)
        ###############################################

        ############### Changing_speed ################
        # y = librosa.effects.time_stretch(y, 2.0)
        ###############################################

        # print("y: ", y.shape) # numpy 형식 (13980, ) (12121, )
        # write("test.wav", 16000, y)
        # plt.figure(figsize=(15, 10))
        # plt.xlabel("Index")
        # plt.ylabel("Amp")
        # plt.title("Noise")
        # plt.plot(y)
        # plt.show()


        n_fft = int(self.audio_conf['sample_rate'] * self.audio_conf['window_size'])
        # print("n_fft: ", n_fft) = 320
        window_size = n_fft
        stride_size = int(self.audio_conf['sample_rate'] * self.audio_conf['window_stride'])
        # print("stride_size: ", stride_size) = 160

        # STFT (Short Time Fourier Transform)
        # 음원을 특정 시간 주기(window, 또는 fream)로 쪼갠 뒤, 해당 주기별로 FFT 진행하면 해당 주기만큼 주파수 분석 그래프 얻음
        # 이를 다시 시간 단위로 배열하면 3차원 칼라맵 나옴
        D = librosa.stft(y, n_fft=n_fft, hop_length=stride_size, win_length=n_fft, window=scipy.signal.windows.hamming)
        # print("D.shape: ", D.shape) # (161, 100 | 138 다양하게찍힘)
        # (161, 100 | 138) => (1 + n_fft/2, n_frames) n_frampes 는 auido length 에서 stride_size만큼 나눠줌
        # https://kaen2891.tistory.com/39
        """
        n_fft: 음성의 길이를 얼마 만큼으로 자를 것인가(window)
        e.g) 16kHz에 n_fft=512 이면 1개의 n_fft는 16000/512 = 약 32 총 음성 데이터 길이가 500이면 32씩 1칸으로자름
        보유한 음성 데이터의 sampling rate와 n_fft 사용시 아래 공식사용
        n_Fft = int(sampling rate * window_length(size))
        hop_length: 음성을 얼만큼 겹친 상태로 잘라서 칸으로 나타낼 것인지? 즉 stride임 얼마나 이동할지
        즉, window_length - stride 라고 보면 됌
        why: 초기신호를 window_length 만큼 쪼개기 때문에 당연히 freq resolution 악화 반대로 window_ length를
        늘리면 time resolution 이 악화 되는 trage off 관계를 가짐
        이를 조금 개선 하기 위해 overlap 을 적용
        """
        # plt.figure(figsize=(15, 10))
        # magnitude = np.abs(D)
        # magnitude_dB = librosa.amplitude_to_db(magnitude)
        # img = librosa.display.specshow(magnitude_dB, sr=self.audio_conf['sample_rate'], hop_length=stride_size,
        #                          x_axis='time', y_axis='log')
        # plt.title("41_0518_474_0_09115_01.wav")
        # plt.colorbar(format="%+2.f dB")
        # plt.show()

        spect, phase = librosa.magphase(D)
        # print("spect1: ", spect.shape)
        #if self.mode == 'train':
        #    spect = spec_augment(spect)
        # print("spect2", spect.shape)

        # y가 (313366, ) ,D.shape(161, 1959) 이면
        # print("spect: ", spect) # (161, 1959)
        # print("Phase: ", phase.shape) # (161, 1959) 허수가 대부분
        # plt.figure(figsize=(15, 10))
        # spect_1 = np.abs(spect)
        # db = librosa.amplitude_to_db(spect_1)
        # img = librosa.display.specshow(db, sr=self.audio_conf['sample_rate'], hop_length=stride_size,
        #                                x_axis='time', y_axis='log')
        # plt.colorbar(format="%+2.f dB")
        # plt.show()


        # S = log(S+1) 내 생각으로는 log scale로 바꿔주고 또한 zero point를 0으로 옮겨주기 위해서
        spect = np.log1p(spect)
        # print("spect_log1p: ", spect.shape)
        if self.normalize:
            mean = np.mean(spect)
            std = np.std(spect)
            spect -= mean
            spect /= std
            # 이는 원래 normalize 식을 적용

        # plt.figure(figsize=(15, 10))
        # spect_1 = np.abs(spect)
        # db = librosa.amplitude_to_db(spect_1)
        # img = librosa.display.specshow(db, sr=self.audio_conf['sample_rate'], hop_length=stride_size,
        #                                x_axis='time', y_axis='log')
        # plt.title(audio_path)
        # plt.colorbar(format="%+2.f dB")
        # plt.show()

        spect = torch.FloatTensor(spect)

        return spect

    def parse_transcript(self, transcript):
        # print(list(transcript))
        # ['아', '기', '랑', ' ', '같', '이', ' ', '갈', '건', '데', '요', ',', ' ', '아', '기', '가', ' ', '먹', '을', ' ', '수', ' ', '있', '는', '것', '도', ' ', '있', '나', '요', '?']
        # ['매', '장', ' ', '전', '용', ' ', '주', '차', '장', '이', ' ', '있', '나', '요', '?']
        # ['카', '드', ' ', '할', '인', '은', ' ', '신', '용', '카', '드', '만', ' ', '되', '나', '요', '?']
        # ['미', '리', ' ', '예', '약', '하', '려', '고', ' ', '하', '는', '데', '요', '.']

        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        # filter(조건, 순횐 가능한 데이터): char2index 의 key 에 없는 것(None) 다 삭제 해버림
        # print("transcript: ", transcript):[49, 153, 4, 85, 63, 24, 129, 5, 4, 47, 601, 64, 4, 137, 55, 126]

        transcript = [self.sos_id] + transcript + [self.eos_id]
        # [2001, 49, 153, 4, 85, 63, 24, 129, 5, 4, 47, 601, 64, 4, 137, 55, 126, 2002]

        return transcript

    def __len__(self):
        return self.size # 59662



def _collate_fn(batch):
    # print("batch[list 형식]: ", np.array(batch).shape) # (16, 2) 16=batch, 2=Tensor + transcript
    # print("batch[list 형식]: ", batch[1][0].size()) # (161, Freame)

    def seq_length_(p): # todo 용도
        return p[0].size(1)
    def target_length_(p): # todo 용도
        return len(p[1])

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    # e.g) Tensor([3,2,1]) 이여도 for s in Tensor ==> batch 수로 순환
    seq_lengths = [s[0].size(1) for s in batch]
    # print("seq_length: ", seq_lengths)
    #  [74, 74, 74, 74, 74, 74, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] 16개 batch만큼 반환
    #print("length: ", [s[0][1] for s in batch]) : Tensor 값

    target_lengths = [len(s[1]) for s in batch]
    # print("target_length: ", target_lengths) # [8, 7, 7, 7, 6, 8, 7, 8, 6, 7, 6, 7, 8, 7, 8, 7]
    # 한 문장에 index 수( 단어 수 )
    # print("target: ", [s[1] for s in batch]) # target index들

    max_seq_size = max(seq_lengths)
    # print("max_seq_size: ", max_seq_size)
    max_target_size = max(target_lengths)
    # print("max_target_size: ", max_target_size)

    feat_size = batch[0][0].size(0)
    # print("feat_size: ", feat_size) # 161 : 1+ n_fft/2
    batch_size = len(batch)
    # print("batch_size: ", batch_size) # 16

    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        # print("sample: ", sample)
        tensor = sample[0]
        # print("tensor: ", x, tensor.size()) # [161, Frame]
        target = sample[1]
        # print("target: ", target) : transcript (index 번호들)
        seq_length = tensor.size(1)
        # print(tensor.size(1))
        seqs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        # print("seq: ", x,seqs[x][0].size()) # [161, length]
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
        # print("target: ", targets[x].size())

    seq_lengths = torch.IntTensor(seq_lengths) # [16]

    return seqs, targets, seq_lengths, target_lengths

class AudioDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

class BucketingSampler(Sampler):

    def __init__(self, data_source, batch_size=1):
        """
        비슷한 크기의 samples과 함께 순서대로 배치
        data_source: Dataset
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        # print("data_source: ", len(data_source)) = 59662
        ids = list(range(0, len(data_source))) # idx 만듬
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        # batch_size 만큼 쪼개짐
        # e.g) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] 와 batch_size=3
        # -> [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19]]
        # print("bins: ", len(self.bins)) batch:16 ==> bin: 3729

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


###################################
# Label_loader
###################################
def load_label_json(labels_path):
    with open(labels_path, encoding='utf-8') as label_file:
        labels = json.load(label_file)
        char2index = dict()
        index2char = dict()

        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char

        return char2index, index2char # todo labels 형태

def load_label_index(label_path):
    char2index = dict()
    index2char = dict()
    print(label_path)
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for no, line in enumerate(f):
            if line[0] == '#':
                continue

            index, char, freq = line.strip().split('\t') # strip 양쪽 공백 제거
            char = char.strip()

            if len(char) == 0:
                char = ' '

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char



def spec_augment(feat, T = 20, F = 27, time_mask_num = 2, freq_mask_num = 2):
    feat_size = feat.shape[1] # length
    seq_len = feat.shape[0] # 161
    # print(feat_size)
    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(0, seq_len - f)
        feat[f0 : f0 + f, :] = 0

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(0, feat_size - t)
        feat[:, t0 : t0 + t] = 0

    return feat




























