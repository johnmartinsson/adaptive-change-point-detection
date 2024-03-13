import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def cosine_distance_score(x1, x2):
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def euclidean_distance_score(x1, x2):
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))

def cosine_distance_past_and_future_averages(embeddings, offset=5, M=5):
    ds = distance_past_and_future_averages(embeddings, cosine_distance_score, offset, M)
    return ds

def distance_past_and_future_averages(embeddings, distance_fn, offset=5, M=5):
    ds = np.zeros(len(embeddings))

    for idx in range(M+offset, len(embeddings)-M-offset):
        past_start = idx-M-offset
        past_end   = idx-offset
        future_start = idx+1+offset
        future_end = idx+M+1+offset

        #print("past    : ", past_start, past_end)
        #print("current : ", idx)
        #print("future  : ", future_start, future_end)

        past_mean   = np.mean(embeddings[past_start:past_end,:], axis=0)
        future_mean = np.mean(embeddings[future_start:future_end,:], axis=0)

        cosine_distance = distance_fn(past_mean, future_mean)

        ds[idx] = cosine_distance

    return ds

def change_point_detection_from_embeddings(embeddings, timings, distance_fn=cosine_distance_score, M = 3, prominence=0, n_peaks=5):
    """
        Detects the timing of the change-points for the waveform given the supplied threshold.

        Input:
            embeddings : shape (n, 1024), the BirdNET-Analyser embeddings
            timings    : shape (n, 2)   , the onset and offset of the embeddings

        Output: 
            [t_1, t_2, ..., t_m] : the m change-points
    """

    #emb_hop_length = timings[1][0] - timings[0][0]
    #emb_win_length = timings[0][1] - timings[0][0]

    offset = 0 #int((emb_win_length-emb_hop_length) / emb_hop_length)

    ds = distance_past_and_future_averages(embeddings, distance_fn=distance_fn, offset=offset, M = M)

    peaks = find_peaks(ds, prominence=prominence)

    # sort peaks by prominence
    peak_indices = peaks[0]
    peak_prominences = peaks[1]['prominences']
    xs = sorted(list(zip(peak_indices, peak_prominences)), key=lambda x: x[1], reverse=True)

    # sort by indice
    peak_indices_sorted = sorted([x[0] for x in xs[:n_peaks]])
    #peak_prominences_sorted = [x[1] for x in xs]

    # return most prominent peaks
    peak_timings = np.mean(timings[peak_indices_sorted], axis=1)
    
    return list(peak_timings), ds

def test_change_point_detection():

    embeddings = np.ones((15, 2))
    embeddings[3:5, 0] = 0.1
    embeddings[8:10, 0] = 0
    fig, ax = plt.subplots(1,2)
    ax[0].plot(embeddings[:,0], label='dim 0')
    ax[0].plot(embeddings[:,1], label='dim 1')
    ax[0].set_title('embedding dims over time')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('value')
    ax[0].legend()
    cosine_ds = distance_past_and_future_averages(embeddings, distance_fn=cosine_distance_score, offset=0, M=2)
    euclidean_ds = distance_past_and_future_averages(embeddings, distance_fn=euclidean_distance_score, offset=0, M=2)
    ax[1].plot(cosine_ds, label='cosine distance')
    ax[1].plot(euclidean_ds, label='euclidean distance')
    ax[1].set_title('change-point curve')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('distance between past and future')
    ax[1].legend()
    plt.tight_layout()
    peaks = find_peaks(cosine_ds, prominence=0.1) # Understand prominence ....
    peak_indices = peaks[0]
    peak_prominences = peaks[1]['prominences']

    # sort by prominence
    xs = sorted(list(zip(peak_indices, peak_prominences)), key=lambda x: x[1], reverse=True)
    peak_indices_sorted = [x[0] for x in xs]
    peak_prominences_sorted = [x[1] for x in xs]

    print("peaks (cosine distance)       : ", peak_indices_sorted)
    print("prominences (cosine distance) : ", peak_prominences_sorted)

    return peak_indices_sorted, peak_prominences_sorted
