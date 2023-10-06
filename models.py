import numpy as np

import datasets
import query_strategies as qs

import change_point_detection as cpd

from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.power(x1-x2, 2)))

def weighted_average(xn_mean, n, xm_mean, m):
    return xn_mean * (n / (n + m)) + xm_mean * (m / (n + m))

def queries_from_probas_2(probas, timings, n_queries):
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

    #print(peak_timings)

    # create AL queries
    query_starts  = [0] + peak_timings
    query_ends = peak_timings + [np.mean(timings[-1])]
    AL_queries = list(zip(query_starts, query_ends))

    return AL_queries


def queries_from_probas(probas, ts, n_queries, rel_height):
    # TODO: the relative height may be important
    peaks, info = find_peaks(probas, prominence=0)
    (widths, width_heights, left_ips, right_ips) = peak_widths(probas, peaks, rel_height=rel_height)

    # remap
    # TODO: it is a problem that ts[0] is 1.5
    m = interp1d([0, len(probas)-1], [ts[0], ts[-1]])
    starts = m(left_ips)
    ends   = m(right_ips)

    queries = list(zip(starts, ends))

    peak_prominences = info['prominences']

    # rank by prominence
    _, ranked_queries = list(zip(*sorted(list(zip(peak_prominences, queries)), key=lambda x: x[0], reverse=True)))

    n_pos_queries = int(np.floor((n_queries - 1) / 2))
    pos_queries   = list(ranked_queries[:n_pos_queries])
    pos_queries   = sorted(pos_queries, key=lambda x: x[0])
    neg_queries   = datasets.compute_neg_from_pos(pos_queries, ts[-1])

    #print("pos: ", pos_queries)
    #print("neg: ", neg_queries)

    queries = pos_queries + neg_queries

    assert(len(queries) <= n_queries)

    queries = sorted(queries, key=lambda x: x[0])

    return queries

class ProtoActiveLearner():
    """
    The base class for an active learning model.
    """
    def __init__(self, base_dir):
        self.base_dir    = base_dir

        # initial state of prototypes
        self.n_prototype = np.zeros(1024)
        self.n_count     = 0
        self.p_prototype = np.zeros(1024)
        self.p_count     = 0

    def predict(self, query_embeddings, threshold, temp=1):
        """
        Hard classification of the input.
        """
        probas = self.predict_probas(query_embeddings, temp=temp)
        pred = probas >= threshold
        return pred.astype(np.int32)

    def predict_probas(self, query_embeddings, temp=1):
        """
        Predict pseudo-probabilities of the input belonging to each class.
        """
        probas = np.zeros(len(query_embeddings))
        for idx_query, query_embedding in enumerate(query_embeddings):
            d_n = euclidean_distance(query_embedding, self.n_prototype)
            d_p = euclidean_distance(query_embedding, self.p_prototype)

            proba = np.exp(-d_p / temp) / (np.exp(-d_p/temp) + np.exp(-d_n/temp))
            probas[idx_query] = proba

        return probas

    def predict_queries(self, soundscape_basename, n_queries, rel_height=0.5):
        """
        Return the query timings.
        """

        timings, embeddings = datasets.load_timings_and_embeddings(self.base_dir, soundscape_basename, embedding_dim=1024)
        probas = self.predict_probas(embeddings)
        mean_timings = np.mean(timings, axis=1)



        # TODO: I need a more reliable method to detect the start of a peak...

        #diffs = np.abs(probas[1:] - probas[:-1])[:-1]
        #mean_timings = mean_timings[1:-1]

        # sort by differences
        #sorted_diffs, ranked_mean_timings = list(zip(*(sorted(list(zip(diffs, mean_timings)), key=lambda x: x[0], reverse=True))))

        #n_mean_timings = ranked_mean_timings[:n_queries-1]

        #mean_timings = sorted(n_mean_timings)
        
        # construct queries
        #soundscape_length = qs.get_soundscape_length(self.base_dir, soundscape_basename)
        #query_starts  = [0] + mean_timings
        #query_ends = mean_timings + [soundscape_length]
        #al_queries = list(zip(query_starts, query_ends))

        al_queries = queries_from_probas_2(probas, timings, n_queries)
        #al_queries = queries_from_probas(probas, mean_timings, n_queries, rel_height)
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
        self.update_positive_prototype(p_embeddings)
        self.update_negative_prototype(n_embeddings)

    #def evaluate(self, dataset):
    #    """
    #    Evaluate the model on the dataset.
    #    """
    #    print("Evaluate active learner ...")
    #    pass

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
