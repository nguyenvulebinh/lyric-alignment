from tqdm import tqdm
import glob
import json

all_file_lrc_json = glob.glob("*.json")
all_file_audio = glob.glob("*.wav")

# Load all json lyric
map_lyric = []
for file_path in tqdm(all_file_lrc_json):
    with open(file_path, 'r', encoding='utf-8') as file:
        map_lyric.append(json.load(file))

# Mapping lyric and it's wav path
map_lyric_normed_by_length = dict({})
segment_text = []
sample_length = []
for sample in tqdm(map_lyric):
    for seg in sample['lyric']:
        if seg[-1][1] - seg[0][0] > 3000:
            if map_lyric_normed_by_length.get(sample['sid'], None) is None:
                map_lyric_normed_by_length[sample['sid']] = {
                    'lyric': []
                }
            map_lyric_normed_by_length[sample['sid']]['lyric'].append(seg)
            sample_length.append((seg[-1][1] - seg[0][0]) / 1000)
            for text in seg:
                segment_text.append(text[2])
for item in all_file_audio:
    sid = item.split('/')[-1].split('.')[0]
    if map_lyric_normed_by_length.get(sid, None) is not None:
        map_lyric_normed_by_length[sid]['audio'] = item
        map_lyric_normed_by_length[sid]['sid'] = sid

# Load cleaned word dictionary
dict_map = dict({})
with open('map_dict.txt', 'r', encoding='utf-8') as file:
    lines = file.read().strip().split('\n')
    for line in tqdm(lines):
        info = line.strip().split('\t')
        if len(info) == 3 and int(info[1]) >= 10:
            dict_map[info[0]] = info[2]

# Define clean and norm text function
valid_dict = set(dict_map.keys())
def is_valid_word(seg):
    texts = ' '.join([item[2] for item in seg])
    words = [w.strip(' -.?!,)("\'…“”*_[]') for w in texts.lower().split()]
    return len(set(words) - valid_dict) == 0
def norm_seg(seg):
    norm_sens = []
    for sen in seg:
        s, e, words = sen
        words = [dict_map[w.strip(' -.?!,)("\'…“”*_[]')] for w in words.lower().split()]
        norm_sens.append([s, e, ' '.join(words)])
    return norm_sens


# Clean the text segment
map_lyric_normed_by_word_dict = dict({})
for sid in tqdm(map_lyric_normed_by_length.keys()):
    list_seg = map_lyric_normed_by_length[sid]['lyric']
    list_seg = [item for item in list_seg if is_valid_word(item)]
    if len(list_seg) > 0 and map_lyric_normed_by_length[sid].get('audio', None) is not None:
        list_seg = [norm_seg(item) for item in list_seg]
        map_lyric_normed_by_word_dict[sid] = {
            'sid' : sid,
            'lyric': list_seg,
            'audio': map_lyric_normed_by_length[sid]['audio']
        }

with open('data_norm.json', 'w', encoding='utf-8') as file:
    json.dump(map_lyric_normed_by_word_dict, file, indent=4, ensure_ascii=False)
