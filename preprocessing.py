import subprocess
import glob
from tqdm.contrib.concurrent import process_map
import sys
import os

def convert_wav(file_path):
    file_uri = file_path.split('/')
    out_path = '/'.join(file_uri[:-1] + [file_uri[-1].replace('.wav', '_normed_16k.wav')])
    result = subprocess.run(['ffmpeg', '-loglevel', 'panic', '-y', '-i', file_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', out_path], stdout=subprocess.PIPE)
    return result

def main(data_path):
    all_wav_file = glob.glob(os.path.join(data_path, '**/*.wav'), recursive=True)
    if len(all_wav_file) == 0:
        print("No wav file in ", os.path.join(data_path, '**/*.wav'))
    else:
        process_map(convert_wav, all_wav_file, max_workers=10, chunksize=1)

if __name__ == "__main__":
    # python preprocessing.py data_path
    main(sys.argv[1])