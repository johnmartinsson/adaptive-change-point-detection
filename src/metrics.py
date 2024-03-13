import warnings
import numpy as np
import mir_eval
import scipy

from metrics_utils import match_events

def test_metrics():
    """
        Basic unit test to show that things are working as expected. Prints the
        expected output and the actual output for a couple of easy examples.
    """
    print("Test 1.")
    print("----------------------------------------")
    events_ref  = [(0,1), (2,3), (4,5)]
    events_pred = [(0,1),        (4,5)]
    print("expected : tp = 2, fp = 0, fn = 1")
    tp, fp, fn, total_events = compute_tp_fp_fn(events_ref, events_pred)
    print("output   : tp = {}, fp = {}, fn = {}".format(tp, fp, fn))

    precision = precision_score(tp, fp)
    recall = recall_score(tp, fn)
    f1 = f1_score(precision, recall)
    print("expected precision: 1.0")
    print("output   precision: {}".format(precision))
    print("expected recall: 0.67")
    print("output   precision: {:.2f}".format(recall))
    print("expected f1 score: 0.80")
    print("output   f1 score: {:.2f}".format(f1))
    print("----------------------------------------")

    print("Test 2.")
    print("----------------------------------------")
    events_ref  = [(0,1), (2,3), (4,5)]
    events_pred = [(0,1), (1,2), (4,5)]
    print("expected : tp = 2, fp = 1, fn = 1")
    tp, fp, fn, total_events = compute_tp_fp_fn(events_ref, events_pred)
    print("output   : tp = {}, fp = {}, fn = {}".format(tp, fp, fn))
    print("----------------------------------------")

    print("Test 3.")
    print("----------------------------------------")
    events_ref  = [(0,1),   (2,3),            (4,5)]
    events_pred = [(0,0.5), (2,2.5),(2.5, 3), (4,4.5)]
    print("expected : tp = 3, fp = 1, fn = 0")
    tp, fp, fn, total_events = compute_tp_fp_fn(events_ref, events_pred, min_iou=0.3)
    print("output   : tp = {}, fp = {}, fn = {}".format(tp, fp, fn))
    print("----------------------------------------")

    print("Test 4.")
    print("----------------------------------------")
    events_ref  = [(0,1),   (2,3),   (4,5)]
    events_pred = [(0,0.6), (2,2.5), (4,4.5)]
    print("expected : tp = 1, fp = 2, fn = 2")
    tp, fp, fn, total_events = compute_tp_fp_fn(events_ref, events_pred, min_iou=0.5)
    print("output   : tp = {}, fp = {}, fn = {}".format(tp, fp, fn))
    print("----------------------------------------")


def class_average_matched_miou(events_ref, events_pred, event_labels, min_iou=0.000001):
    """
        Compute the class average matched mean intersection-over-union (IOU) from a list of
        reference and predicted events.

        Parameters
        ----------
        events_ref : list of tuples
            A list of reference events. Each event is a tuple of the form
            (start_time, end_time, label).
        events_pred : list of tuples
            A list of predicted events. Each event is a tuple of the form
            (start_time, end_time, label).
        event_labels : list of strings
            A list of the event labels.

        Returns
        -------
        miou : float
            The class average matched mean intersection-over-union (IOU).
    """

    miou = 0.0
    for event_label in range(event_labels):

        events_ref_class  = [(e['onset'], e['offset']) for e in events_ref  if e['event_label'] == event_label]
        events_pred_class = [(e['onset'], e['offset']) for e in events_pred if e['event_label'] == event_label]

        if len(events_ref_class) == 0:
            print("no references")
            warnings.warn("No reference events found.")
        if len(events_pred_class) == 0:
            print("no events")
            warnings.warn("No predicted events found.")

        if len(events_ref_class) == 0 and len(events_pred_class) == 0:
            print("no events and no references")
            warnings.warn("No events found, IOU set to 1.0")
            return 1.0

        miou_class = average_matched_iou(events_ref_class, events_pred_class, min_iou=min_iou)

        miou += miou_class

    return miou / len(event_labels)

def coverage(q1, q2):
    """
        Compute the coverage of q1 by q2. Coverage is defined as the ratio of
        the intersection of q1 and q2 to the length of q1.
    """
    _intersection = intersection(q1, q2)
    return _intersection / (q1[1]-q1[0])

def intersection(q1, q2):
    (a, b) = q1
    (c, d) = q2
    if b < c or d < a:
        return 0
    else:
        return np.min([b, d]) - np.max([a, c])

def union(q1, q2):
    (a, b) = q1
    (c, d) = q2

    s = np.min([a, c])
    e = np.max([b ,d])

    return e-s

def iou(q1, q2):
    _intersection = intersection(q1, q2)
    _union = union(q1, q2)
    #print(q1, q2)
    #print("intersection: ", _intersection)
    #print("union: ", _union)
    return intersection(q1, q2) / union(q1, q2)

def average_matched_iou(events_ref, events_pred, min_iou=0.3):
    """
        Compute the average IOU of matched events.
    """

    events_ref_ndarray = np.array(events_ref).transpose() # shape (n, 2) -> (2, n)
    events_pred_ndarray = np.array(events_pred).transpose() # shape (m, 2) -> (2, m)

    matches = match_events(events_ref_ndarray, events_pred_ndarray, min_iou=min_iou)

    # TODO: is this correct?
    if len(matches) == 0:
        warnings.warn("No matches found, IOU set to 0.0")
        # TODO: if there are no matches, IOU is 0
        return 0

    ref_match_indices = np.array([idx for (idx, _) in matches])
    pred_match_indices = np.array([idx for (_, idx) in matches])
    
    pos_pred_matches = events_pred_ndarray.transpose()[pred_match_indices]
    pos_ref_matches  = events_ref_ndarray.transpose()[ref_match_indices]

    ious = []
    for q_ref, q_pred in zip(pos_ref_matches, pos_pred_matches):
        ious.append(iou(q_ref, q_pred))

    return np.mean(ious)


def compute_tp_fp_fn(events_ref, events_pred, min_iou=0.3):

    if len(events_pred) == 0:
        return 0, 0, len(events_ref), len(events_ref)
    
    events_ref_ndarray = np.array(events_ref).transpose() # shape (n, 2) -> (2, n)
    events_pred_ndarray = np.array(events_pred).transpose() # shape (m, 2) -> (2, m)

    matches = match_events(events_ref_ndarray, events_pred_ndarray, min_iou=min_iou)
    tp = len(matches)
    fp = len(events_pred) - tp
    fn = len(events_ref) - tp
    total_events = len(events_ref)

    return tp, fp, fn, total_events

def precision_score(tp, fp):
    return tp / (tp + fp)

def recall_score(tp, fn):
    return tp / (tp + fn)

def f1_score(tp, fp, fn): #precision, recall):
    return (2*tp) / (2*tp + fp + fn) #(2 * precision * recall) / (precision + recall)

def f1_score_from_events(events_ref, events_pred, min_iou):
    tp, fp, fn, total_events = compute_tp_fp_fn(events_ref, events_pred, min_iou)

    # TODO: this should never happen
    if tp + fp + fn == 0:
        warnings.warn("F1-score is not defined when tp + fp + fn = 0, F1-score set to 1.0")
        return 1.0

    # TODO: what should this be?
    if tp + fp == 0:
        warnings.warn("Precision is not defined when tp + fp = 0, F1-score set to 0.0")
        return 0
    
    #precision = precision_score(tp, fp)
    #recall    = recall_score(tp, fn)

    f1 = f1_score(tp, fp, fn) #precision, recall)
    return f1
