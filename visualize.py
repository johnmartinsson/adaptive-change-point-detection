import os
import librosa
import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

import evaluate
import warnings
import models
import utils
import datasets
import oracles
import query_strategies as qs
import change_point_detection as cpd

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_soundscape(datasets_dir, sim_dir, soundscape_basename, class_name, n_queries, idx_run, method_name, ax):
    """ The method uses plot_annotations to plot the ground truth annotations, and the ground truth annotations for
    <soundscape_m>.txt in the directory base_dir, and the predicted annotations for <iter_n>_<soundscape_m>.tsv in the in
    the sim_dir
    """
    print(sim_dir)
    print(datasets_dir)
    pred_annotation_dir = os.path.join(sim_dir, '{}/n_queries_{}_noise_0.0/{}/{}/train_annotations/'.format(class_name, n_queries, method_name, idx_run))
    # /mnt/storage_1/john/data/bioacoustic_sed/generated_datasets/me_1.0_0.25s/train_soundscapes_snr_0.0/
    truth_annotation_dir = os.path.join(datasets_dir, '{}_1.0_0.25s/train_soundscapes_snr_0.0/'.format(class_name))

    # find the predicted annotation file (different iter_n) which ends with <soundscape_m>.tsv
    pred_annotation_file = None
    for f in os.listdir(pred_annotation_dir):
        if f.endswith(soundscape_basename + '.tsv'):
            pred_annotation_file = os.path.join(pred_annotation_dir, f)
            break
    if pred_annotation_file is None:
        raise ValueError('could not find predicted annotation file for soundscape {}'.format(soundscape_basename))
    
    ground_truth_annotation_file = os.path.join(truth_annotation_dir, soundscape_basename + '.txt')
    wav_file = os.path.join(truth_annotation_dir, soundscape_basename + '.wav')

    plot_annotations(pred_annotation_file, ground_truth_annotation_file, wav_file, ax)

def plot_annotations(pred_annotation_file, ref_annotation_file, wav_file, ax):
    """There are predicted annotation files with the name <iter_n>_<soundscape_m>.tsv containing annotations of this format:
    onset	offset	event_label
    
    12.857142857142858	17.142857142857142	baby
    25.714285714285715	30.0	baby
    
    and ground truth annotation files with the name <soundscape_m>.txt like this:
    
    12.996099551489078      15.390657374618328      baby
    21.275883446027628      21.398332425619465      baby
    21.407991700390316      22.80708467091186       baby
    
    the files correspond to a soundscape with the name <soundscape_m>.wav

    The following method will load the predicted annotation files, the ground truth annotation files and the corresponding wav files. It will then plot the predicted annotations and the ground truth annotations on top of the spectrogram of the wav file.
    """
    # 1. load predicted annotations
    pred_annotations = []
    with open(pred_annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            d = line.split('\t')
            start_time = float(d[0])
            end_time = float(d[1])
            pred_annotations.append((start_time, end_time))
    
    # 2. load reference annotations
    ref_annotations = []
    with open(ref_annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            d = line.split('\t')
            start_time = float(d[0])
            end_time = float(d[1])
            ref_annotations.append((start_time, end_time))

    # 3. load wav file
    window_length = 0.025
    wave, sr = librosa.load(wav_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wave,
        sr=sr,
        n_fft = utils.next_power_of_2(int(sr * window_length)),
        hop_length = utils.next_power_of_2(int(sr * window_length)) // 2,
    )
    
    # 4. plot the black and white Mel spectrogram
    colormap = plt.get_cmap('gray')

    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram, ref=np.max),
        x_axis='time',
        y_axis='mel',
        sr=sr,
        hop_length = utils.next_power_of_2(int(sr * window_length)) // 2,
        fmin=0.0,
        fmax=22050.0,
        cmap=colormap,
        ax=ax,
    )

    # color bar for ax object to format
    # plt.colorbar(format='%+2.0f dB')
    alpha=0.2
    # 5. plot predicted annotations
    for a in pred_annotations:
        ax.axvspan(a[0], a[1], color='blue', alpha=alpha)
        # highlight the start and end
        ax.axvline(a[0], color='red', alpha=1.0)
        ax.axvline(a[1], color='red', alpha=1.0)
    
    # 6. plot reference annotations
    for a in ref_annotations:
        ax.axvspan(a[0], a[1], color='green', alpha=alpha)
    
    # # 7. save figure
    # if savefile is not None:
    #     plt.savefig(savefile)
    # else:
    #     plt.show()
    

def visualize_query_strategy(query_strategy, query_strategy_name, soundscape_basename, base_dir, n_queries=7, vis_probs=True, vis_queries=True, vis_threshold=True, vis_cpd=True, vis_label=True, vis_peaks=True, vis_true=True, vis_ent=False, vis_embs_label=False, savefile=None, noise_factor=0, normalize=True, emb_win_length=1.0, fp_noise=0.0, fn_noise=0.0, coverage_threshold=0.05, prominence_threshold=0.0):
    
    oracle = oracles.WeakLabelOracle(base_dir, fp_noise=fp_noise, fn_noise=fn_noise, coverage_threshold=coverage_threshold)

    #print("###########################################")
    #print(query_strategy_name)
    #print("###########################################")
    
    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)

    timings, embeddings = datasets.load_timings_and_embeddings(base_dir, soundscape_basename)
    pred_probas = query_strategy.predict_proba(embeddings, noise_factor=noise_factor)
    if query_strategy.fixed_queries or query_strategy.emb_cpd or query_strategy.opt_queries:
        pred_queries = query_strategy.predict_queries(base_dir, soundscape_basename, n_queries, prominence_threshold=prominence_threshold)
    else:
        pred_queries = models.queries_from_probas(pred_probas, timings, n_queries, soundscape_length, prominence_threshold=prominence_threshold)
        pred_queries = sorted(pred_queries, key=lambda x: x[0])
    print("method = {}, n_queries = {}".format(query_strategy_name, len(pred_queries)))

    pred_pos_events = oracle.pos_events_from_queries(pred_queries, soundscape_basename)

    _, _, embs_label = evaluate.get_embeddings_3(pred_pos_events, base_dir, soundscape_basename, emb_win_length)

    ts_probas = np.mean(timings, axis=1)

    opt_queries = qs.optimal_query_strategy(base_dir, soundscape_basename, soundscape_length)
    #print("opt qs: ", opt_queries)
    ref_pos_events = oracle.pos_events_from_queries(opt_queries, soundscape_basename)
    #print("ref pos: ", ref_pos_events)

    # extract Mel spectrogram
    window_length = 0.025
    wave, sr = librosa.load(os.path.join(base_dir, soundscape_basename + ".wav"))
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wave,
        sr=sr,
        n_fft = utils.next_power_of_2(int(sr * window_length)),
        hop_length = utils.next_power_of_2(int(sr * window_length)) // 2,
    )
    
    fig, ax = plt.subplots(2, 1, figsize=(10,3.0))
    ax[0].imshow(np.flip(np.log(mel_spectrogram + 1e-10), axis=0), aspect='auto')
    ax[0].set_title(query_strategy_name + ", prominence = {}, n_queries = {}".format(prominence_threshold, len(pred_queries)))
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])

    probas = pred_probas.reshape((len(pred_probas), 1))
    ds = cpd.distance_past_and_future_averages(
        probas,
        cpd.euclidean_distance_score, offset=0, M=1
    )

    if query_strategy.emb_cpd:
        _, ds = cpd.change_point_detection_from_embeddings(
            embeddings,
            timings,
            M = 1,
            prominence = prominence_threshold,
            n_peaks = n_queries-1,
            distance_fn = cpd.cosine_distance_score,
        )
        #ds = ds/np.max(ds)



    # peaks 
    peaks = find_peaks(ds, prominence=prominence_threshold)
    peak_indices     = peaks[0]
    peak_prominences = peaks[1]['prominences']
    peak_indices = sorted(utils.sort_by_rank(peak_prominences, peak_indices)[:n_queries-1])

    if vis_embs_label:
        ax[1].plot(ts_probas, embs_label, label='emb. label', color=colors[3])

    if vis_peaks:
        ax[1].plot(ts_probas[peak_indices], ds[peak_indices], "x", color="red", label='peaks')

    if vis_cpd:
        ax[1].plot(ts_probas, ds, label='cpd', color=colors[0])
    if vis_probs:
        ax[1].plot(ts_probas, pred_probas, label='probas', color=colors[1])
    if vis_ent:
        ax[1].plot(ts_probas, models.binary_entropy(pred_probas), label='entropy', color=colors[4])
    if vis_threshold:
        ax[1].hlines([0.5], [0], [soundscape_length], color='red', linestyle='dashed')
        ax[1].hlines([1.0], [0], [soundscape_length], color='green', linestyle='dashed')
    ax[1].set_xlim(0, soundscape_length) #ts_probas[-1])
    ax[1].set_ylim(0, 1.2)
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('pseudo-probability')

    def plot_events(ax, events, color, label, ymax=1.0):
        starts = [s for (s, _) in events]
        ends   = [e for (_, e) in events]
        heights = [ymax for _ in range(len(events))]
        ax.vlines(starts + ends, ymin=0, ymax=ymax, color=color, label=label)
        ax.hlines(heights, starts, ends, color=color)

    def plot_queries(ax, queries, color, label):
        # start of all queries
        points = [s for (s, e) in queries]
        # end of last query
        points = points + [queries[-1][1]]
        points[0] += 0.05
        #points = [0.1] + points + [29.9]
        ax.vlines(points, ymin=-0.2, ymax=1.2, color=color, label=label, linestyle='dashed')

    # plot true event onsets and offsets
    if vis_true:
        #print(ref_pos_events)
        plot_events(ax[1], ref_pos_events, color=colors[2], label='reference labels', ymax=0.9)
        #plot_queries(ax[1], opt_queries, color='magenta', label='reference queries')
        #query_centers = [e - ((e - s) / 2) for (s, e) in opt_queries]
        #for idx_q_c, q_c in enumerate(query_centers):
        #    ax[1].text(x=q_c-0.3, y=1.25, s=r'$q_{}$'.format(idx_q_c))
    

    if vis_label:
        plot_events(ax[1], pred_pos_events, color='magenta', label='annotated labels', ymax=0.95)

    if vis_queries:
        assert len(pred_queries) <= n_queries
        if len(pred_queries) < n_queries:
            warnings.warn("not using all queries: {}".format(len(pred_queries)))
        plot_queries(ax[1], pred_queries, color='red', label='predicted queries')

        query_centers = [e - ((e - s) / 2) for (s, e) in pred_queries]
        for idx_q_c, q_c in enumerate(query_centers):
            ax[1].text(x=q_c-0.3, y=1.05, s=r'$q_{}$'.format(idx_q_c))
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')
        #plt.cla()
        #plt.clf()
        #plt.close()
