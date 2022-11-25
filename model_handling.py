from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from torch import nn
import warnings
import torch
from transformers.modeling_outputs import CausalLMOutput
from collections import OrderedDict

_HIDDEN_STATES_START_POSITION = 2


class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.feature_transform = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(config.hidden_size, config.hidden_size)),
            ('bn1', nn.BatchNorm1d(config.hidden_size)),
            ('activation1', nn.LeakyReLU()),
            ('drop1', nn.Dropout(config.final_dropout)),
            ('linear2', nn.Linear(config.hidden_size, config.hidden_size)),
            ('bn2', nn.BatchNorm1d(config.hidden_size)),
            ('activation2', nn.LeakyReLU()),
            ('drop2', nn.Dropout(config.final_dropout)),
            ('linear3', nn.Linear(config.hidden_size, config.hidden_size)),
            ('bn3', nn.BatchNorm1d(config.hidden_size)),
            ('activation3', nn.LeakyReLU()),
            ('drop3', nn.Dropout(config.final_dropout))
        ]))

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.is_wav2vec_freeze = False

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_wav2vec(self, is_freeze=True):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        if is_freeze:
            self.is_wav2vec_freeze = True
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        else:
            self.is_wav2vec_freeze = False
            for param in self.wav2vec2.parameters():
                param.requires_grad = True
        self.freeze_feature_encoder()

        model_total_params = sum(p.numel() for p in self.parameters())
        model_total_params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("model_total_params: {}\nmodel_total_params_trainable: {}".format(model_total_params,
                                                                                model_total_params_trainable))

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        B, T, F = hidden_states.size()
        hidden_states = hidden_states.view(B * T, F)

        hidden_states = self.feature_transform(hidden_states)

        hidden_states = hidden_states.view(B, T, F)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
