import torch
import torch.nn as nn
import numpy as np

from transformers import DistilBertPreTrainedModel, DistilBertModel
from transformers import DistilBertConfig

from transformers.activations import gelu
from transformers.modeling_outputs import QuestionAnsweringModelOutput

MASK_PROB = 0.15

class DistilBertForMLMQA(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config) 
        self.mask_prob = MASK_PROB
        self.distilbert = DistilBertModel(config)
        self.dummy_param = nn.Parameter(torch.empty(0)) # dummy param to get device later

        # MLM layers
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # QA layers
        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)

        self.init_weights()

        self.mlm_loss_fct = nn.CrossEntropyLoss()

    def get_output_embeddings(self):
        return self.vocab_projector

    def set_output_embeddings(self, new_embeddings):
        self.vocab_projector = new_embeddings

    def mask_input_ids(self, input_ids):
        # select random indices so that MASK_PROB number of indices are masked
        nrow = input_ids.size()[0]
        ncol = input_ids.size()[1]
        r = torch.randint(low=0, high=nrow, size=(int(MASK_PROB*nrow*ncol),)).long() 
        c = torch.randint(low=0,high=ncol,size=(int(MASK_PROB*nrow*ncol),)).long() 

        input_ids_cpy = input_ids.detach().clone().to(self.dummy_param.device)
        input_ids_cpy[r][:,c] = 103 # [MASK] token id
        
        # labels are same as original inputs prior to masking, except 
        #       indices with padding have (0) now have value -100
        labels = torch.ones(input_ids.size()).long().to(self.dummy_param.device)*(-100)
        labels[r][:,c] = input_ids[r][:,c]

        return input_ids_cpy, labels

    def forward(
        self,
        input_ids=None, 
        attention_mask=None, 
        head_mask=None, 
        inputs_embeds=None,
        apply_input_mask=False, # decide on this
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        apply_input_mask=False, # this should be true if using mask_input_ids fn for original mlm_qa model
        qa=True,
        labels=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # for masking
        input_ids_cpy = input_ids
        if labels is None and apply_input_mask:
            input_ids_cpy, labels = self.mask_input_ids(input_ids)

        dlbrt_output = self.distilbert(
                input_ids=input_ids_cpy,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)

        # MLM
        if labels is not None:
            prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
            prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        # QA
        if qa:
            logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
            end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

        mlm_loss = None
        qa_loss = None
        total_loss = None

        if qa and start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2

        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1).long())

        if mlm_loss is not None and qa_loss is not None:
            total_loss = mlm_loss + qa_loss
        else if mlm_loss is not None:
            total_loss = qa_loss
        else:
            total_loss = mlm_loss

        if not return_dict:
            output = (start_logits, end_logits) + dlbrt_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        if qa:
            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=dlbrt_output.hidden_states,
                attentions=dlbrt_output.attentions,
                )
        else:
            return MaskedLMOutput(
                loss=total_loss,
                logits=prediction_logits,
                hidden_states=dlbrt_output.hidden_states,
                attentions=dlbrt_output.attentions,
                )




