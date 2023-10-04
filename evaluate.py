import os

import metrics
import oracles
import datasets_utils
import query_strategies as qs

def valid_queries(queries, base_dir, soundscape_basename):
    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)

    sorted_queries = sorted(queries, key = lambda x: x[0])

    # check no overlap
    for idx_query in range(len(sorted_queries)-1):
        q1 = sorted_queries[idx_query]
        q2 = sorted_queries[idx_query + 1]

        assert(q1[1] <= q2[0])

    # check sums correctly
    tot = 0
    for (s, e) in sorted_queries:
        L = e-s
        tot += L

    assert(tot == soundscape_length)

def pos_events_from_queries(queries, oracle, soundscape_basename):
    pos_events = []
    for (q_st, q_et) in queries:
        c = oracle.query(q_st, q_et, soundscape_basename)
        if c == 1:
            pos_events.append((q_st, q_et))
    return pos_events

def evaluate_all(base_dir, soundscape_basename, min_iou=0.001, n_queries=0):

    # create oracle
    oracle = oracles.WeakLabelOracle(base_dir)

    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)

    # create queries
    opt_queries = qs.optimal_query_strategy(base_dir, soundscape_basename, soundscape_length)
    if n_queries == 0:
        n_queries   = len(opt_queries) # this is the annotation budget
    cp_queries  = qs.change_point_query_strategy(n_queries, base_dir, soundscape_basename, soundscape_length)
    fix_queries = qs.fixed_query_strategy(n_queries, base_dir, soundscape_basename, soundscape_length)

    #assert(len(cp_queries) == len(fix_queries))

    # assert valied queries
    valid_queries(opt_queries, base_dir, soundscape_basename)
    valid_queries(cp_queries, base_dir, soundscape_basename)
    valid_queries(fix_queries, base_dir, soundscape_basename)

    queries_and_names = [
        (opt_queries, 'opt'),
        (cp_queries,  'cp'),
        (fix_queries, 'fix')
    ]

    name_to_label_quality = {}
    ref_path = os.path.join(base_dir, '{}.txt'.format(soundscape_basename))
    for (queries, name) in queries_and_names:
        pos_ref  = datasets_utils.load_pos_ref(ref_path)
        pos_pred = pos_events_from_queries(queries, oracle, soundscape_basename)
        
        label_quality = metrics.f1_score_from_events(pos_ref, pos_pred, min_iou=min_iou)
        name_to_label_quality[name] = label_quality

    return name_to_label_quality, len(opt_queries)
