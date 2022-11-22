import json
import torchaudio
from tqdm.contrib.concurrent import process_map
import os

base_path = './formated_data'
if not os.path.exists(base_path):
    os.makedirs(base_path)

def handle_file(sample_data):
    audio_file = sample_data['audio']
    sample_id = sample_data['sid']
    full_data = True
    if not os.path.exists(os.path.join(base_path, sample_id)):
        os.makedirs(os.path.join(base_path, sample_id))
        full_data = False
    else:
        for idx, seg in enumerate(sample_data['lyric']):
            if not os.path.exists(os.path.join(base_path, sample_id, '{}_{}.wav'.format(sample_id, idx))):
                full_data = False
            if not os.path.exists(os.path.join(base_path, sample_id, '{}_{}.txt'.format(sample_id, idx))):
                full_data = False
    if full_data:
        return
    try:
        wavform, rate = torchaudio.load(audio_file)
        for idx, seg in enumerate(sample_data['lyric']):
            start = seg[0][0]
            end = seg[-1][1]
            audio = wavform[:, int(start*16):int(end*16)]
            torchaudio.save(os.path.join(base_path, sample_id, '{}_{}.wav'.format(sample_id, idx)), 
                            audio, rate)
            with open(os.path.join(base_path, sample_id, '{}_{}.txt'.format(sample_id, idx)), 'w',encoding='utf-8') as file:
                for sen in seg:
                    file.write('{}\t{}\t{}\n'.format(sen[0] - start, sen[1] - start, sen[2]))
    except:
        print("remove", sample_id)
        os.rmdir(os.path.join(base_path, sample_id))

            
if __name__ == "__main__":
    with open('data_norm.json', 'r', encoding='utf-8') as file:
        list_audio = list(json.load(file).values())
    print(len(list_audio))
    r = process_map(handle_file, list_audio, max_workers=60, chunksize=1)