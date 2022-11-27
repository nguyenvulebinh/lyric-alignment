import os
os.environ['WANDB_PROJECT'] = 'lyric_alignment'

from transformers.trainer_utils import IntervalStrategy
from transformers import Wav2Vec2Processor
from model_handling import Wav2Vec2ForCTC
from data_handling import DataCollatorCTCWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_metric, load_from_disk, load_dataset
import numpy as np

wer_metric = load_metric("wer")
pre_trained_model_path = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
pretrain_data_path = 'nguyenvulebinh/song_dataset'
repo_name = './tune_acoustic_model'
checkpoint_name = "pretrain_large"

def get_compute_metrics_fnc(processor, wer_metric):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    return compute_metrics

if __name__ == "__main__":
    processor = Wav2Vec2Processor.from_pretrained(pre_trained_model_path)
    model = Wav2Vec2ForCTC.from_pretrained(pre_trained_model_path)
    model.freeze_feature_encoder()
    model_total_params = sum(p.numel() for p in model.parameters())
    model_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model_total_params: {}\nmodel_total_params_trainable: {}".format(model_total_params, model_total_params_trainable))
    
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    all_dataset = load_dataset(pretrain_data_path)
    splits = all_dataset.train_test_split(test_size=5000, seed=101, shuffle=True)
    train_dataset = splits['train']
    eval_dataset = splits['test']
    
    print(train_dataset)
    print(eval_dataset)
    print(eval_dataset[0])    
    
    # Done experiment with batch_size = 8 | accumulation_steps=4 
    batch_size = 2
    num_epochs=50
    accumulation_steps=1
    
    training_args = TrainingArguments(
        output_dir=f'{repo_name}/{checkpoint_name}',
        logging_dir=f'{repo_name}/{checkpoint_name}/log',
        group_by_length=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        num_train_epochs=num_epochs,
        gradient_accumulation_steps=accumulation_steps,
        metric_for_best_model='loss',
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=False, 
        remove_unused_columns=False,
        dataloader_num_workers=8,
        save_steps=2000,
        eval_steps=20000,
        logging_steps=10,
        learning_rate=0.0001,
        weight_decay=0.005,
        warmup_steps=4000,
        save_total_limit=20,
        ignore_data_skip=True,
        report_to="wandb",  # enable logging to W&B,
        run_name=checkpoint_name  # name of the W&B run (optional)
    ) 
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        #compute_metrics=get_compute_metrics_fnc(processor, wer_metric),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    # trainer.evaluate()