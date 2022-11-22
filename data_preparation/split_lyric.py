import glob
import re
import random
import json
from tqdm.contrib.concurrent import process_map  # or thread_map

time_lyric_regex = r"[\[|<](\d\d):(\d\d.\d\d)[\]|>](.*)"
time_regex = r"[\[|<](\d\d):(\d\d.\d\d)[\]|>]"

def ignore_line(text):
    text_lower = text.lower().replace(' ', '').strip(' -.?!,)("\'…“”*_[]')
    if text_lower.startswith('bàihát:') or text_lower.startswith('casĩ:') or text_lower.startswith('cakhúc:') or text_lower.startswith('sángtác:') or text_lower.startswith('trìnhbày:') or text_lower.startswith('lyricby:') or text_lower.startswith('đk') :
        return True
    if len(text.strip(' -.?!,)("\'…“”*_[]')) == 0:
        return True
    return False

def ignore_file(text):
    matches = list(re.finditer(time_regex, text, re.MULTILINE))
    if len(matches) > 0:
        return True
    return False

def get_max_sample():
    return random.choice([10000, 15000, 20000])

def handle_lrc(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split('\n')
    
    timestamp = []
    text = []
    
    label = []
    try:
        for line in lines:
            matches = list(re.finditer(time_lyric_regex, line, re.MULTILINE))
            if len(matches) > 0:
                if ignore_file(matches[0].group(3)):
                    return
                if not ignore_line(matches[0].group(3)):
                    start_time = int(float(matches[0].group(1)) * 60 * 1000) + int(float(matches[0].group(2)) * 1000)
                    timestamp.append(start_time)
                    text.append(matches[0].group(3).strip())
                    
    except:
        print(file_path)
        return
    
    timestamp = timestamp[3:]
    text = text[3:]
    
    tmp_length = 0
    tmp_segs = []
    current_time_stamp = 0
    for idx in range(len(timestamp) - 1):
        start_time = timestamp[idx]
        end_time = timestamp[idx + 1]
        current_length = end_time - start_time
        if current_time_stamp > end_time:
            return
        
        if len(text[idx]) == 0 and current_length < 2000:
            tmp_length += current_length
        elif len(text[idx]) == 0 or current_length > 15000:
            if len(tmp_segs) > 0:
                label.append(tmp_segs)
                tmp_length = 0
                tmp_segs = []
        elif tmp_length > get_max_sample() and len(tmp_segs) > 0:
            label.append(tmp_segs)
            
            tmp_length = end_time - start_time
            tmp_segs = [(start_time, end_time, text[idx])]
        else:
            tmp_length += end_time - start_time
            tmp_segs.append((start_time, end_time, text[idx]))
        current_time_stamp = end_time
        
    if len(tmp_segs) > 0:
        label.append(tmp_segs)
    
    sid = file_path.split('/')[-1].split('.')[0]
    
    with open('{}.json'.format(sid), 'w', encoding='utf-8') as file:
        json.dump({
            "sid": sid,
            "lyric": label
        }, file, ensure_ascii=False, indent=4)
    
    return label

if __name__ == "__main__":
    all_file_lrc = glob.glob("*.lrc")
    r = process_map(handle_lrc, all_file_lrc, max_workers=40, chunksize=1)