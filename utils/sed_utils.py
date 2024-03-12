import librosa
import pandas as pd
import numpy as np

import time

def get_tf_transform(name, n_mels, sample_rate, normalize=False):
    bioacoustic_conf = get_bioacoustic_pcen_conf()
    speech_conf      = get_speech_pcen_conf()
    tf_transforms = {
        'decibel'           : lambda x: wav_to_mel(x - (np.sum(x)/np.size(x)), sample_rate, n_mels=n_mels, normalize=normalize),
        'pcen_biodiversity' : lambda x: wav_to_pcen(x - (np.sum(x)/np.size(x)), sample_rate, bioacoustic_conf, n_mels=n_mels, normalize=normalize),
        'pcen_speech'       : lambda x: wav_to_pcen(x - (np.sum(x)/np.size(x)), sample_rate, speech_conf, n_mels=n_mels, normalize=normalize),
        'spectrogram'       : lambda x: wav_to_spec(x - (np.sum(x)/np.size(x)), sample_rate, n_bins=n_mels, normalize=normalize),
        'stacked'           : lambda x: wav_to_stacked(x - (np.sum(x)/np.size(x)), sample_rate, n_mels=n_mels, normalize=normalize),
    }
    tf_transform = tf_transforms[name]
    return tf_transform

def my_frames_to_time(frames, sample_rate):
    if sample_rate == 11025:
        window_size = 256 # ~25 ms
        hop_size    = 128 # 
        n_fft       = 256 # 
    elif sample_rate == 22050:
        window_size = 2*256 # ~25 ms
        hop_size    = 2*128 # 
        n_fft       = 2*256 # 
    elif sample_rate > 44100:
        window_size = 4*256 # ~25 ms
        hop_size    = 4*128 # 
        n_fft       = 4*256 # 
    else:
        raise ValueError("undefined for sample rate: {}".format(sample_rate))

    return librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_size, n_fft=n_fft)

def get_bioacoustic_pcen_conf():
    return {
	'gain' : 0.8,
	'bias' : 10,
	'power' : 0.25,
	'time_constant' : 0.06,
	'eps' : 1e-6    
    }
def get_speech_pcen_conf():
    return {
	'gain' : 0.98,
	'bias' : 2,
	'power' : 0.5,
	'time_constant' : 0.4,
	'eps' : 1e-6    
    }

def wav_to_spec(wav, sample_rate, n_bins, normalize=False):
    if sample_rate == 11025:
        window_size = 256 # ~25 ms
        hop_size    = 128 # 
        n_fft       = 256 # 
    elif sample_rate == 22050:
        window_size = 2*256 # ~25 ms
        hop_size    = 2*128 # 
        n_fft       = 2*256 # 
    elif sample_rate > 44100:
        window_size = 4*256 # ~25 ms
        hop_size    = 4*128 # 
        n_fft       = 4*256 # 
    else:
        raise ValueError("undefined for sample rate: {}".format(sample_rate))

    D = librosa.stft(
        wav,
        n_fft = n_fft,
        hop_length = hop_size,
        win_length = window_size
    )

    #if normalize:
    #    D = D / np.sum(D)

    S = np.power(np.abs(D), 2)
    S_spec = librosa.power_to_db(np.abs(D), ref=np.max)

    if normalize:
        S_spec = S_spec / np.sum(S_spec)

    return S_spec

def wav_to_pcen(wav, sample_rate, conf, n_mels=40, normalize=False):
    if sample_rate == 11025:
        window_size = 256 # ~25 ms
        hop_size    = 128 # 
        n_fft       = 256 # 
    elif sample_rate == 22050:
        window_size = 2*256 # ~25 ms
        hop_size    = 2*128 # 
        n_fft       = 2*256 # 
    elif sample_rate > 44100:
        window_size = 4*256 # ~25 ms
        hop_size    = 4*128 # 
        n_fft       = 4*256 # 
    else:
        raise ValueError("undefined for sample rate: {}".format(sample_rate))

    D = librosa.feature.melspectrogram(
        wav, 
        sr=sample_rate,
        win_length=window_size,
        n_fft=n_fft,
        hop_length=hop_size,
        n_mels=n_mels     # used to derive default params for PCEN
    )

    #if normalize:
    #    D = D / np.sum(D)

    S_pcen = librosa.core.pcen(
        D, 
        sr=sample_rate,
        gain=conf['gain'],
        bias=conf['bias'],
        power=conf['power'],
        time_constant=conf['time_constant'],
        eps=conf['eps']
    )
    if normalize:
        S_pcen = S_pcen / np.sum(S_pcen)

    return np.expand_dims(S_pcen, axis=0)

def wav_to_mel(wav, sample_rate, n_mels=40, normalize=False):
    if sample_rate == 11025:
        window_size = 256 # ~25 ms
        hop_size    = 128 
        n_fft       = 256 
    elif sample_rate == 22050:
        window_size = 2*256 # ~ 25 ms
        hop_size    = 2*128 
        n_fft       = 2*256 
    elif sample_rate >= 44100:
        window_size = 4*256 # ~25 ms
        hop_size    = 4*128 # 
        n_fft       = 4*256 # 
    else:
        raise ValueError("undefined for sample rate: {}".format(sample_rate))

    D = librosa.feature.melspectrogram(
        wav, 
        sr=sample_rate,
        win_length=window_size,
        n_fft=n_fft,
        hop_length=hop_size,
        n_mels=n_mels
    )

    #if normalize:
    #    D = D / np.sum(D)

    S_db = librosa.power_to_db(np.abs(D), ref=np.max)
    if normalize:
        S_db = S_db / np.sum(S_db)
    return np.expand_dims(S_db, axis=0)

def wav_to_stacked(wav, sample_rate, n_mels=40, normalize=False):
    bio_conf    = get_bioacoustic_pcen_conf()
    speech_conf = get_speech_pcen_conf()

    S_mel = wav_to_mel(wav, sample_rate, n_mels=n_mels, normalize=normalize)
    S_pcen_bio = wav_to_pcen(wav, sample_rate, bio_conf, n_mels=n_mels, normalize=normalize)
    S_pcen_speeh = wav_to_pcen(wav, sample_rate, speech_conf, n_mels=n_mels, normalize=normalize)

    S = np.concatenate((S_mel, S_pcen_bio, S_pcen_speeh), axis=0)

    return S

# TODO: Write a test for this
# TODO: What if we store the indices instead of the audio?
def split_into_segments(wave, sample_rate, hop_size, window_size):

    N = len(wave)
    M = int(np.floor((N-window_size)/hop_size))
    index = np.array([np.arange(window_size) + hop_size*i for i in range(M+1)])
    
    # split the wave into the segments
    segments = wave[index]
    
    # compute the start and end time for each segment
    time_intervals = [(np.min(x)/sample_rate, np.max(x)/sample_rate) for x in index]
    
    return segments, time_intervals

# TODO: Write a test for this
def compute_interval_intersection(i1, i2):
    (a_s, a_e) = i1
    (b_s, b_e) = i2
    if b_s > a_e or a_s > b_e:
        return 0
    else:
        o_s = max(a_s, b_s)
        o_e = min(a_e, b_e)
        return o_e - o_s

# TODO: write a test for this
def compute_interval_union(i1, i2):
    (a_s, a_e) = i1
    (b_s, b_e) = i2
    
    o_s = min(a_s, b_s)
    o_e = max(a_e, b_e)
    
    return o_e - o_s

# TODO: Write a test for this
def compute_interval_intersection_over_union(i1, i2):
    
    intersection = compute_interval_intersection(i1, i2)
    union = compute_interval_union(i1, i2)
    
    if union == 0:
        return 0 #print(i1, i2)
    else:
        return intersection/union

def get_segments_and_labels(wave, sample_rate, annotation_df, n_shot, n_background, hop_size, window_size, n_classes, n_time, get_label_fn):
    segments, segment_intervals = split_into_segments(wave, sample_rate, hop_size, window_size)

    # TODO: this needs optimization ...
    annotation_intervals, labels = get_annotation_intervals_and_labels(annotation_df, get_label_fn)
    # take first 5 shots
    annotation_intervals = annotation_intervals[:n_shot]
    labels = labels[:n_shot]

    segment_size = segments.shape[1]
    # special case for n_time == 1
    if n_time == 1:
        scale = segment_size // (n_time*2)
    else:
        scale = segment_size // n_time

    target = np.zeros((len(wave)//scale, n_classes), dtype=int)

    # the reason we scale down like this is memory efficiency, the targets
    # take way too much memory otherwise
    for annotation_interval, label in zip(annotation_intervals, labels):
        start_time = annotation_interval[0] / scale
        end_time   = annotation_interval[1] / scale
        start_idx  = int(np.floor(start_time * sample_rate))
        end_idx    = int(np.ceil(end_time * sample_rate))

        if label < n_classes:
            target[start_idx:end_idx, label] += 1

    target = np.clip(target, 0, 1)
    segment_targets, segment_target_intervals = split_into_segments(target, sample_rate//scale, hop_size//scale, window_size//scale)
    segment_targets = np.transpose(segment_targets, axes=(0,2,1))

    # special case for n_time == 1
    if n_time == 1:
        segment_targets = np.max(segment_targets, axis=2)
        segment_targets = np.expand_dims(segment_targets, axis=2)
    
    # bool_idx for signal and background
    signal_bool_idx     = np.sum(segment_targets[:,0:n_classes,:], axis=(1, 2)) > 0 # sum over class and time dimension
    background_bool_idx = np.sum(segment_targets[:,0:n_classes,:], axis=(1, 2)) == 0 # sum over class and time dimension

    signal_segments = segments[signal_bool_idx,:]
    signal_segment_targets = segment_targets[signal_bool_idx,:,:]
    
    background_segments = segments[background_bool_idx,:]
    background_segment_targets = segment_targets[background_bool_idx,:,:]

    signal_intervals     = np.array(segment_intervals)[signal_bool_idx]
    background_intervals = np.array(segment_intervals)[background_bool_idx]
    t2 = time.time()
    
    if len(background_segments) < len(signal_segments):
        return signal_segments, signal_segment_targets, signal_intervals, background_segments, background_segment_targets, background_intervals
    else:
        # TODO: maybe think a bit more about this. Mainly done to save memory space.
        #n_sample = min(len(background_segments), n_background) # can only sample as many as there is
        #background_random_idx = np.random.choice(np.arange(len(background_segments)), n_sample, replace=False) # sample background signals
        background_segments = background_segments[:n_background] #background_random_idx]
        background_segment_targets = background_segment_targets[:n_background] #background_random_idx]

        background_intervals = background_intervals[:n_background] #background_random_idx]
        
        return signal_segments, signal_segment_targets, signal_intervals, background_segments, background_segment_targets, background_intervals

def load_wave(wav_path):
    wave, sample_rate = librosa.load(wav_path, sr=None)
    wave = wave * (2**31) # rescale according to recommendation for PCEN in librosa
    return wave, sample_rate


def get_annotation_interval(row):
    start_time = row[1][1]
    end_time = row[1][2]
    return (start_time, end_time)

def get_annotation_intervals_and_labels(annotation_df, get_label_fn):
    columns = annotation_df.columns
    annotation_intervals = []
    labels = []
    
    for row in annotation_df.iterrows():
        label = get_label_fn(row, columns)
        annotation_interval = get_annotation_interval(row)
        annotation_intervals.append(annotation_interval)
        labels.append(label)
    
    return annotation_intervals, labels

def plot_spectrogram(ax, audio_segment, sample_rate):
    D = librosa.feature.melspectrogram(audio_segment, sr=sample_rate)
    S_db = librosa.power_to_db(np.abs(D), ref=np.max)
    ax.imshow(np.flip(S_db, axis=0), aspect='auto')
