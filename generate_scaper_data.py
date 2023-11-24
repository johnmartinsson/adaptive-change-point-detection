import os
import scaper
import numpy as np
import soundfile as sf
import glob
import argparse
import metrics

parser = argparse.ArgumentParser(description='A data generation script.')
parser.add_argument('--data_dir', help='The data dir containing the train and test source directories', required=True, type=str)
parser.add_argument('--dataset_name', help='The name of the generated dataset', required=True, type=str)
parser.add_argument('--snr', help='The signal-to-noise ratio', required=True, type=float)
parser.add_argument('--n_soundscapes', help='The number of soundscapes to generate', required=True, type=int)
parser.add_argument('--bg_label', help='The backgrounds to use.', required=False, type=str)
parser.add_argument('--fg_label', help='The foregrounds to use.', required=False, type=str)
args = parser.parse_args()

def has_overlapping_events(annotation_list):
    for (s1, e1, c) in annotation_list:
        for (s2, e2, c) in annotation_list:
            q1 = (s1, e1)
            q2 = (s2, e2)
            if metrics.coverage(q1, q2) > 0 and metrics.coverage(q1, q2) < 1.0:
                return True
    return False

for idx_split, split in enumerate(['train', 'test']):
    print("Generating {} soundscapes ...".format(split))

    # OUTPUT FOLDER
    outfolder = os.path.join(args.data_dir, 'generated_datasets', args.dataset_name, '{}_soundscapes_snr_{}'.format(split, args.snr))
    #outfolder = '{}/generated_datasets/{}/{}_soundscapes_snr_{}/'.format(args.data_dir, split, args.snr)

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # SCAPER SETTINGS
    fg_folder = '{}/sources/{}_sources/foreground/'.format(args.data_dir, split)
    bg_folder = '{}/sources/{}_sources/background/'.format(args.data_dir, split)

    # get and set the average background loudness
    #lufss = []
    #for bg_path in glob.glob(os.path.join(bg_folder, 'bg', '*.wav')):
    #    wave, sample_rate = sf.read(bg_path)
    #    lufs = scaper.audio.get_integrated_lufs(wave, samplerate=sample_rate)
    #    lufss.append(lufs)

    #average_lufs = np.mean(lufss)
    #print("LUFS average = {}".format(average_lufs))
    ref_db = -20 # LUFS #average_lufs

    n_soundscapes = args.n_soundscapes #10
    duration = 30.0 

    min_events = 3
    max_events = 3

    source_time_dist = 'const'
    source_time = 0.0

    event_duration_dist = 'uniform'
    event_duration_min = 4 #0.1
    event_duration_max = 4 #0.3

    event_time_dist = 'uniform'
    event_time_min = 4 # TODO: this is here because of a CPD problem where we can not detect events before 3 seconds.. need to solve
    event_time_max = 26 # duration-event_duration_max

    snr_dist = 'uniform'
    snr_min = args.snr
    snr_max = args.snr

    pitch_dist = 'uniform'
    pitch_min = 0 #-3.0
    pitch_max = 0 #3.0

    time_stretch_dist = 'uniform'
    time_stretch_min = 1.0 #0.8
    time_stretch_max = 1.0 #1.2

    basename = 'soundscape'
        
    for n in range(n_soundscapes):
        
        print('Generating soundscape: {:d}/{:d}'.format(n+1, n_soundscapes))
        
        # create a scaper
        sc = scaper.Scaper(duration, fg_folder, bg_folder, random_state=n_soundscapes*idx_split + n)
        sc.protected_labels = []
        sc.ref_db = ref_db
        
        # add background
        if args.bg_label == 'all':
            bg_label = ('choose', [])
        else:
            bg_label = ('choose', [args.bg_label])

        sc.add_background(label=bg_label, 
                          source_file=('choose', []), 
                          source_time=('const', 0))

        # add random number of foreground events
        n_events = np.random.randint(min_events, max_events+1)

        if args.fg_label == 'all':
            fg_label = ('choose', [])
        else:
            fg_label = ('choose', [args.fg_label])
        for _ in range(n_events):
            sc.add_event(label=fg_label,
                         source_file=('choose', []), 
                         source_time=(source_time_dist, source_time), 
                         event_time=(event_time_dist, event_time_min, event_time_max), 
                         event_duration=(event_duration_dist, event_duration_min, event_duration_max), 
                         snr=(snr_dist, snr_min, snr_max),
                         pitch_shift=(pitch_dist, pitch_min, pitch_max),
                         time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
        
        # generate
        audiofile = os.path.join(outfolder, "{}_{:d}.wav".format(basename, n))
        jamsfile = os.path.join(outfolder, "{}_{:d}.jams".format(basename, n))
        txtfile = os.path.join(outfolder, "{}_{:d}.txt".format(basename, n))
        
        # TODO: maybe loop until no overlapping events?
        overlapping_events = True
        while overlapping_events:
            sounscape_audio, soundscape_jam, annotation_list, event_audio_list = sc.generate(
                audio_path            = audiofile,
                jams_path             = jamsfile,
                allow_repeated_label  = True,
                allow_repeated_source = False,
                reverb                = None, 
                fix_clipping          = True, # TODO: is this reasonable?
                peak_normalization    = False,
                quick_pitch_time      = False,
                save_isolated_events  = False,
                isolated_events_path  = None,
                disable_sox_warnings  = True,
                no_audio              = True, #False,
                txt_path              = txtfile,
            )

            overlapping_events = has_overlapping_events(annotation_list)
            if overlapping_events:
                print("OVERLAPPING: ", annotation_list)

        # TODO: generate audio from jams file
        scaper.generate_from_jams(jams_infile = jamsfile, audio_outfile = audiofile)
   
    #txt_paths = glob.glob(os.path.join(outfolder, "*.txt"))
    #print("#########################################################")
    #header = "Audiofilename,\t\tStarttime,\t\tEndtime,\t\tClass\n"
    #print(header)
    #with open(os.path.join(outfolder, "annotations.csv"), 'w') as fw:
    #    fw.write(header)
    #    for txt_path in txt_paths:
    #        wave_path = os.path.basename(txt_path.replace('.txt', '.wav'))
    #        with open(txt_path, 'r') as fr:
    #            lines = fr.readlines()
    #            for line in lines:
    #                l = "{},\t".format(wave_path) + ",\t".join(line.split('\t'))
    #                fw.write(l)
                    #print(l)
