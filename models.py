import numpy as np

import datasets
import query_strategies as qs
import utils
import oracles

import change_point_detection as cpd

from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.power(x1-x2, 2)))

def weighted_average(xn_mean, n, xm_mean, m):
    return xn_mean * (n / (n + m)) + xm_mean * (m / (n + m))

def binary_entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def queries_from_probas(probas, timings, n_queries):
    probas = probas.reshape((len(probas), 1))
    ds = cpd.distance_past_and_future_averages(
        probas,
        cpd.euclidean_distance_score, offset=0, M=1
    )

    n_peaks = n_queries-1

    # we want to rank all peaks, hence 0 prominence
    peaks = find_peaks(ds, prominence=0)

    # sort peaks by prominence
    peak_indices = peaks[0]
    peak_prominences = peaks[1]['prominences']
    xs = sorted(list(zip(peak_indices, peak_prominences)), key=lambda x: x[1], reverse=True)

    # sort by indice
    peak_indices_sorted = sorted([x[0] for x in xs[:n_peaks]])

    # return most prominent peaks
    peak_timings = list(np.mean(timings[peak_indices_sorted], axis=1))

    # create AL queries
    query_starts  = [0] + peak_timings
    query_ends = peak_timings + [np.mean(timings[-1])]
    AL_queries = list(zip(query_starts, query_ends))

    return AL_queries

class AdaptiveQueryStrategy():
    """
    The base class for an active learning model.
    """
    def __init__(self, base_dir, random_soundscape, fixed_queries):
        self.base_dir          = base_dir
        self.random_soundscape = random_soundscape
        self.fixed_queries     = fixed_queries

        # initial state of prototypes
        self.n_prototype = np.zeros(1024)
        self.n_count     = 0
        self.p_prototype = np.zeros(1024)
        self.p_count     = 0

    def initialize_with_ground_truth_labels(self, soundscape_basename):
        #print("initialize: ", soundscape_basename)
        oracle = oracles.WeakLabelOracle(self.base_dir)
        
        # load initial queries and embeddings
        queries, embeddings = datasets.load_timings_and_embeddings(self.base_dir, soundscape_basename)
        
        # label the embeddings
        labels = []
        for (s, e) in queries:
            c = oracle.query(s, e, soundscape_basename)
            labels.append(c)
        labels = np.array(labels)
        
        n_embeddings = embeddings[labels == 0]
        p_embeddings = embeddings[labels == 1]
        
        # initialize the active learner
        self.update(p_embeddings=p_embeddings, n_embeddings=n_embeddings)

    def predict(self, query_embeddings, threshold, temp=1):
        """
        Hard classification of the input.
        """
        probas = self.predict_probas(query_embeddings, temp=temp)
        pred = probas >= threshold
        return pred.astype(np.int32)

    def predict_probas(self, query_embeddings, temp=1, noise_factor=0):
        """
        Predict pseudo-probabilities of the input belonging to each class.
        """
        probas = np.zeros(len(query_embeddings))
        for idx_query, query_embedding in enumerate(query_embeddings):
            d_n = euclidean_distance(query_embedding, self.n_prototype)
            d_p = euclidean_distance(query_embedding, self.p_prototype)

            proba = np.exp(-d_p / temp) / (np.exp(-d_p/temp) + np.exp(-d_n/temp))
            probas[idx_query] = proba

        # TODO: this noise factor should probably be removed again. Only here
        # to study how the method fails.
        probas = probas + np.random.rand(len(probas)) * noise_factor
        probas = probas / np.max(probas)

        return probas

    def predict_pos_events(self, base_dir, soundscape_basename, threshold=0.5):
        query_timings, query_embeddings = datasets.load_timings_and_embeddings(base_dir, soundscape_basename)

        probas      = self.predict_probas(query_embeddings)
        pos_indices = (probas >= threshold)

        avg_timings = query_timings.mean(axis=1)
        pos_avg_timings = avg_timings[pos_indices]
        hop_length  = avg_timings[1]-avg_timings[0]

        pos_events = []

        idx_timing = 0
        # TODO: maybe improve this a bit? fairly naive as it is
        while idx_timing < len(pos_avg_timings):
            s = pos_avg_timings[idx_timing]
            while idx_timing < len(pos_avg_timings)-1 and (pos_avg_timings[idx_timing+1] - pos_avg_timings[idx_timing]) <= hop_length:
                idx_timing += 1
            e = pos_avg_timings[idx_timing]
            pos_events.append((s, e))
            idx_timing += 1
        return pos_events


    def predict_queries(self, soundscape_basename, n_queries, noise_factor=0):
        """
        Return the query timings.
        """
        if self.fixed_queries:
            soundscape_length = qs.get_soundscape_length(self.base_dir, soundscape_basename)

            fix_queries = np.linspace(0, soundscape_length, n_queries+1)
            fix_queries = list(zip(fix_queries[:-1], fix_queries[1:]))
            
            return fix_queries
        else:
            timings, embeddings = datasets.load_timings_and_embeddings(self.base_dir, soundscape_basename, embedding_dim=1024)
            probas = self.predict_probas(embeddings, noise_factor=noise_factor)

            al_queries = queries_from_probas(probas, timings, n_queries)
            al_queries = sorted(al_queries, key=lambda x: x[0])

            return al_queries

    def update_positive_prototype(self, p_embeddings):
        xm_mean = np.mean(p_embeddings, axis=0)
        m       = len(p_embeddings)

        xn_mean = self.p_prototype
        n       = self.p_count

        self.p_prototype = weighted_average(xn_mean, n, xm_mean, m)
        self.p_count += m

    def update_negative_prototype(self, n_embeddings):
        xm_mean = np.mean(n_embeddings, axis=0)
        m       = len(n_embeddings)

        xn_mean = self.n_prototype
        n       = self.n_count

        self.n_prototype = weighted_average(xn_mean, n, xm_mean, m)
        self.n_count += m

    def update(self, p_embeddings, n_embeddings):
        """
        Update the model given the last query batch to the Oracle.
        """
        # TODO: is this correct?
        if len(p_embeddings) > 0:
            self.update_positive_prototype(p_embeddings)
        if len(n_embeddings) > 0:
            self.update_negative_prototype(n_embeddings)

    def rank_soundscapes(self, soundscape_basenames):
        ranks = []
        for soundscape_basename in soundscape_basenames:
            rank = self.entropy_score(soundscape_basename)
            ranks.append(rank)

        return utils.sort_by_rank(ranks=ranks, values=soundscape_basenames)

    def entropy_score(self, soundscape_basename):
        # load the embeddings for the soundscape
        _, embeddings = datasets.load_timings_and_embeddings(self.base_dir, soundscape_basename, embedding_dim=1024)

        probas = self.predict_probas(embeddings)
        entropies = [binary_entropy(p) for p in probas]
        return np.mean(entropies)

    def next_soundscape_basename(self, remaining_soundscape_basenames):
        if self.random_soundscape:
            # TODO: maybe remove this seed again?
            #np.random.seed(56)
            return np.random.choice(remaining_soundscape_basenames)
        else:
            ranked_soundscapes = self.rank_soundscapes(remaining_soundscape_basenames)
            return ranked_soundscapes[0]


class BaseActiveLearner():

    """
    The base class for an active learning model.
    """
    def __init__(self):
        pass

    def predict(self, x):
        """
        Hard classification of the input.
        """
        return torch.argmax(self.predict_proba(x))

    def predict_proba(self, x):
        """
        Predict pseudo-probabilities of the input belonging to each class.
        """
        pass

    def predict_query_timings(self, active_dataset):
        """
        Return the query timings.
        """
        print("Predict query timings ...")

    def update(self, active_dataset):
        """
        Update the model given the last query batch to the Oracle.
        """
        print("Update active learner ...")
        pass

    def evaluate(self, dataset):
        """
        Evaluate the model on the dataset.
        """
        print("Evaluate active learner ...")
        pass
