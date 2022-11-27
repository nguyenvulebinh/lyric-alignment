import torch
from transformers import Wav2Vec2Processor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
# from utils import norm_text_segment
import random

@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    
    @staticmethod
    def normalize_input_label(audio, segment_text, segment_timestamp):
        idx = 0
        max_size = 15000
        batch_segment = []
        tmp_segment = []
        tmp_size = 0
        batch_audio = []
        batch_label = []
        try_time = 30
        while idx < len(segment_text):
            current_length = segment_timestamp[idx][1] - segment_timestamp[idx][0]
            if tmp_size + current_length < max_size:
                tmp_segment.append((segment_text[idx], segment_timestamp[idx]))
                tmp_size += current_length
            else:
                if len(tmp_segment) > 0:
                    batch_segment.append(tmp_segment)
                    
                prev_length = segment_timestamp[idx - 1][1] - segment_timestamp[idx - 1][0]
                if current_length + prev_length < max_size:
                    idx -= 1
                    tmp_segment = [(segment_text[idx], segment_timestamp[idx])]
                    tmp_size = prev_length
                elif current_length < max_size:
                    tmp_segment = [(segment_text[idx], segment_timestamp[idx])]
                    tmp_size = current_length
            idx += 1
            try_time -= 1
            if try_time < 0:
                break
        if len(tmp_segment) > 0:
            batch_segment.append(tmp_segment)
        
        for segment in batch_segment:
            text_raw = ' '.join([item[0] for item in segment])
            start_time = segment[0][1][0]
            end_time = segment[-1][1][1]
            
            segment_length = end_time - start_time
            if segment_length > max_size:
                continue
            
            batch_audio.append(audio[start_time*16:end_time*16])
            batch_label.append(text_raw)
        
        return batch_audio, batch_label

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        
        batch_audio = []
        batch_label = []
        
        
        for sample_index in range(len(features)):
            try:
                audio_segments, text_segments = DataCollatorCTCWithPadding.normalize_input_label(features[sample_index]['audio']['array'], 
                                                                                                features[sample_index]['segment_text'], 
                                                                                                features[sample_index]['segment_align'])
                batch_audio.extend(audio_segments)
                batch_label.extend(text_segments)
            except:
                continue
            
        data = list(zip(batch_audio, batch_label))
        random.shuffle(data)    
        max_sample = int(len(features) * 1.5)
        data = data[:max_sample]
        
        batch_audio = [item[0] for item in data]
        batch_label = [item[1] for item in data]
        
        input_features = [{"input_values": item} for item in batch_audio]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        
        # encode target text to label ids 
        batch_label = [self.processor.tokenizer(item).input_ids for item in batch_label]        
        # get the tokenized label sequences
        label_features = [{"input_ids": item} for item in batch_label]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch