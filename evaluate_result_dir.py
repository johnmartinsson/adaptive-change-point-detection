import sys
import glob
import config
import sound_event_eval as see
import os
import concurrent.futures

def main():
    result_dir =  sys.argv[1] #'./results/2024-01-30/'

    # find all config.yaml files in the result_dir
    config_files = glob.glob(os.path.join(result_dir, '**', 'config.yaml'), recursive=True)
    #print(config_files)
    confs = []
    for config_file in config_files:
        conf = config.Config()
        conf.load_config_yaml(config_file)
        confs.append(conf)

    # confs = [config.Config().load_config_yaml(config_file) for config_file in config_files]
    # for conf in confs:
    #     see.evaluate_test_and_train(conf)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        try:
            executor.map(see.evaluate_test_and_train, confs)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

if __name__ == '__main__':
    main()