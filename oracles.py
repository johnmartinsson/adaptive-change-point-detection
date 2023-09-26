import os
import glob

def class_name2class_id(class_name):
    if 'bg' in class_name:
        return 0
    elif 'fg' in class_name:
        return 1

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
    def __init__(self, base_dir):
        super(WeakLabelOracle, self).__init__()
        annotation_file_paths = glob.glob(os.path.join(base_dir, '*.txt'))
        annotation_file_paths = list(filter(lambda x: not "embedding" in x, annotation_file_paths))

        if len(annotation_file_paths) == 0:
            raise ValueError("no annotation files in: {}".format(base_dir))

        self.annotations = {}

        for annotation_file_path in annotation_file_paths:
            key = os.path.basename(annotation_file_path).split('.')[0]
            self.annotations[key] = load_annotations(annotation_file_path)
            
    def query(self, start_time, end_time, soundscape_basename):
        ann = self.annotations[soundscape_basename]
        for (a_start_time, a_end_time, c) in ann:
            if a_start_time >= start_time and a_end_time <= end_time:
                return c
        return 0
