import os
import scaper
import numpy as np
import soundfile as sf
import glob
import argparse

parser = argparse.ArgumentParser(description='A data generation script.')
parser.add_argument('--data_dir', help='The data dir containing the train and test source directories', required=True, type=str)
parser.add_argument('--snr', help='The signal-to-noise ratio', required=True, type=float)
args = parser.parse_args()

for split in ['train', 'test']:
    print("Generating {} soundscapes ...".format(split))

    # OUTPUT FOLDER
    outfolder = '{}/{}_soundscapes_snr_{}/'.format(args.data_dir, split, args.snr)

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # SCAPER SETTINGS
    fg_folder = '{}/{}_source/foreground/'.format(args.data_dir, split)
    bg_folder = '{}/{}_source/background/'.format(args.data_dir, split)

    # get and set the average background loudness
    lufss = []
    for bg_path in glob.glob(os.path.join(bg_folder, 'bg', '*.wav')):
        wave, sample_rate = sf.read(bg_path)
        lufs = scaper.audio.get_integrated_lufs(wave, samplerate=sample_rate)
        lufss.append(lufs)

    average_lufs = np.mean(lufss)
    print("LUFS average = {}".format(average_lufs))
    ref_db = average_lufs

    n_soundscapes = 20
    duration = 10.0 

    min_events = 2
    max_events = 3

    source_time_dist = 'const'
    source_time = 0.0

    event_duration_dist = 'uniform'
    event_duration_min = 10 #0.1
    event_duration_max = 10 #0.3

    event_time_dist = 'uniform'
    event_time_min = 0
    event_time_max = duration-event_duration_max



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
        
    # Generate 1000 soundscapes using a truncated normal distribution of start times

    for n in range(n_soundscapes):
        
        print('Generating soundscape: {:d}/{:d}'.format(n+1, n_soundscapes))
        
        # create a scaper
        sc = scaper.Scaper(duration, fg_folder, bg_folder, random_state=42)
        sc.protected_labels = []
        sc.ref_db = ref_db
        
        # add background
        sc.add_background(label=('const', 'bg'), 
                          source_file=('choose', []), 
                          source_time=('const', 0))

        # add random number of foreground events
        n_events = np.random.randint(min_events, max_events+1)
        for _ in range(n_events):
            sc.add_event(label=('choose', []), 
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
        
        sc.generate(audiofile, jamsfile,
                    allow_repeated_label=True,
                    allow_repeated_source=False,
                    reverb=0.1, # TODO: what does this do?
                    disable_sox_warnings=True,
                    no_audio=False,
                    txt_path=txtfile)
   
    txt_paths = glob.glob(os.path.join(outfolder, "*.txt"))
    #print("#########################################################")
    header = "Audiofilename,\t\tStarttime,\t\tEndtime,\t\tClass\n"
    #print(header)
    with open(os.path.join(outfolder, "annotations.csv"), 'w') as fw:
        fw.write(header)
        for txt_path in txt_paths:
            wave_path = os.path.basename(txt_path.replace('.txt', '.wav'))
            with open(txt_path, 'r') as fr:
                lines = fr.readlines()
                for line in lines:
                    l = "{},\t".format(wave_path) + ",\t".join(line.split('\t'))
                    fw.write(l)
                    #print(l)
