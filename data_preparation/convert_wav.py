import glob
from tqdm.contrib.concurrent import process_map
import subprocess

def convert_wav(file_path):
    result = subprocess.run(['ffmpeg', '-loglevel', 'panic', '-y', '-i', file_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', file_path.replace('.mp3', '.wav')], stdout=subprocess.PIPE)

if __name__ == "__main__":
    all_mp3_file = glob.glob('*.mp3')
    r = process_map(convert_wav, all_mp3_file, max_workers=40, chunksize=1)
