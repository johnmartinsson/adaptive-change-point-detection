# Active learning for bioacoustics

A repository containing an active learning framework for bioacoustics.

# Generate the dataset

## Extract the scaper source data files

    python dcase_to_scaper_source_material.py --data_dir=<dcase_dir>/Development_Set/Training_Set/BV/ --scaper_dir=./scaper_source_files/BV/ --class_name=AMRE

This will load the DCASE annotations and treat all positive annotations of AMRE as the foreground event class, and everything else as the background event class. Training and test data are split on a recording/file level.

## Generate the soundscapes

    python generate_scaper_data.py --data_dir=./scaper_source_files/BV/AMRE

This will use the scaper source data files, and generate train_soundscapes and test_soundscapes. The soundscapes directories will contain (*.wav, *.txt) file pairs, where the timings of the foreground event is annotated in the *.txt files. Please see the source code for further details on the soundscape generation process.

## Extract the embeddings

Setup the BirdNET-Analyzer v2.4 (https://github.com/kahst/BirdNET-Analyzer).

    python embeddings.py --i ./scaper_source_files/BV/AMRE/train_soundscapes/ --o ./scaper_source_files/BV/AMRE/train_soundscapes/ --threads 8 --batchsize 16 --overlap 0

This will generate the embeddings and store them in the same directory. Embeddings are for 3 second segments with the specified overlap.
    

## Background

0. Define 10 interesting scenes.
1. Collect background sounds from Aporee.

    wind forest -bird -birds
   
   
   
## Foreground

1. Collect foreground sounds.
2. BirdCLEF?
3. 
