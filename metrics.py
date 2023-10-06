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

    events_ref_ndarray = np.array(events_ref).transpose() # shape (n, 2) -> (2, n)
    events_pred_ndarray = np.array(events_pred).transpose() # shape (m, 2) -> (2, m)

    matches = match_events(events_ref_ndarray, events_pred_ndarray, min_iou=min_iou)

    ref_match_indices = np.array([idx for (idx, _) in matches])
    pred_match_indices = np.array([idx for (_, idx) in matches])

    pos_pred_matches = events_pred_ndarray.transpose()[pred_match_indices]
    pos_ref_matches  = events_ref_ndarray.transpose()[ref_match_indices]

    ious = []
    for q_ref, q_pred in zip(pos_ref_matches, pos_pred_matches):
        #print("intervals: ", q_ref, q_pred)
        #print("iou: ", iou(q_ref, q_pred))
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

def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)

def f1_score_from_events(events_ref, events_pred, min_iou):
    tp, fp, fn, total_events = compute_tp_fp_fn(events_ref, events_pred, min_iou)

    # TODO: what should this be?
    if tp + fp == 0:
        return 0
    
    precision = precision_score(tp, fp)
    recall    = recall_score(tp, fn)

    # TODO: what should this be?
    if precision + recall == 0:
        return 0
    
    f1 = f1_score(precision, recall)
    return f1
