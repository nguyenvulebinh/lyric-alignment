from datasets import Dataset, Audio
import glob

def map_bytes_lyric(sample):
    sample['audio'] = dict({})
    with open(sample['path_wav'],'rb') as file:
        sample['audio']['bytes'] = bytes(file.read())
    with open(sample['path_lyric'], 'r', encoding='utf-8') as file:
        label = file.read().strip().split('\n')
        segment_text = []
        segment_align = []
        for segment in label:
            info = segment.split('\t')
            ss = int(info[0])
            se = int(info[1])
            segment_text.append(info[2])
            segment_align.append((ss, se))
            
        sample['segment_text'] = segment_text
        sample['segment_align'] = segment_align
        sample['sid'] = sample['id']
    return sample

def make_pretrain_data():
    file_wavs = glob.glob('./formated_data/**/*.wav')
    print(len(file_wavs))
    file_label = glob.glob('./formated_data/**/*.txt')
    print(len(file_label))
    
    map_labels = dict({})
    for item in file_wavs:
        id = item.split('/')[-1].split('.')[0]
        if map_labels.get(id, None) is None:
            map_labels[id] = {
                'path_wav': item
            }
            
    for item in file_label:
        id = item.split('/')[-1].split('.')[0]
        if map_labels.get(id, None) is not None:
            map_labels[id]['path_lyric'] = item

    dict_ds = {
        'id': [],
        'path_wav': [],
        'path_lyric': []
    }
    
    for item in map_labels.keys():
        if map_labels[item].get('path_wav', None) is not None and map_labels[item].get('path_lyric', None) is not None:
            dict_ds['id'].append(item)
            dict_ds['path_wav'].append(map_labels[item]['path_wav'])
            dict_ds['path_lyric'].append(map_labels[item]['path_lyric'])
    
    pretrain_dataset = Dataset.from_dict(dict_ds)
    print(pretrain_dataset)
    
    pretrain_dataset = pretrain_dataset.map(map_bytes_lyric, 
                                    num_proc=50, 
                                    writer_batch_size=10, 
                                    cache_file_name='./cache/pretrain.arrow', 
                                    remove_columns=pretrain_dataset.column_names)
    pretrain_dataset = pretrain_dataset.cast_column('audio', Audio(decode=True))
    pretrain_dataset.save_to_disk('./pretrain_data')
    
    

if __name__ == "__main__":
    make_pretrain_data()