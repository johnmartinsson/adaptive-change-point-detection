import numpy as np
import os
import metrics
import glob

def class_name2class_id(class_name):
    # TODO: rewrite this to multi-class formulation,
    # for now only handles binary problems
    if 'me' in class_name:
        return 1
    elif 'dog' in class_name:
        return 1
    elif 'baby' in class_name:
        return 1
    else:
        return 0
    #if 'bg' in class_name:
    #    return 0
    #elif 'fg' in class_name:
    #    return 1

def load_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            d = l.split('\t')
            start_time = float(d[0])
            end_time = float(d[1])
            c = class_name2class_id(d[2])
            annotations.append((start_time, end_time, c))
    return annotations

class WeakLabelOracle:
    def __init__(self, base_dir, fp_noise, fn_noise, coverage_threshold):
        super(WeakLabelOracle, self).__init__()
        annotation_file_paths = glob.glob(os.path.join(base_dir, '*.txt'))
        annotation_file_paths = list(filter(lambda x: not "embedding" in x, annotation_file_paths))

        # noisy oracle
        self.fp_noise = fp_noise
        self.fn_noise = fn_noise
        self.coverage_threshold = coverage_threshold

        if len(annotation_file_paths) == 0:
            raise ValueError("no annotation files in: {}".format(base_dir))

        self.annotations = {}

        for annotation_file_path in annotation_file_paths:
            key = os.path.basename(annotation_file_path).split('.')[0]
            self.annotations[key] = load_annotations(annotation_file_path)

    def pos_events_from_queries(self, queries, soundscape_basename):
        # TODO: does this introduce label noise? I think not.
        pos_events = []
        for (q_st, q_et) in queries:
            c = self.query(q_st, q_et, soundscape_basename)
            if c == 1:
                pos_events.append((q_st, q_et))
        return pos_events

    def query(self, q_start_time, q_end_time, soundscape_basename):
        ann = self.annotations[soundscape_basename]
        #print("ann: ", ann)
        for (a_start_time, a_end_time, c) in ann:
            # ground truth timings
            q_gt = (a_start_time, a_end_time)
            # query timings
            q_qu = (q_start_time, q_end_time)
            # TODO: what is the right threshold here?
            if metrics.coverage(q_gt, q_qu) > self.coverage_threshold:
                return np.random.choice([0, 1], p=[self.fn_noise, 1-self.fn_noise])

        return np.random.choice([0, 1], p=[1-self.fp_noise, self.fp_noise])
