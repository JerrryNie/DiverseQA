from math import sqrt
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertForQuestionAnswering
import torch.nn.functional as F
import numpy as np
EPS = 1e-10
K = 5.0

answer_type_freq_span80 = {
    0: 659156,
    1: 164527,
    2: 23179,
    3: 2173,
    4: 4545
}
_answer_type_freq_span80 = dict(sorted(answer_type_freq_span80.items()))
_answer_type_freq_span80 = np.array(list(_answer_type_freq_span80.values()))
_answer_type_freq_span80 = _answer_type_freq_span80 / _answer_type_freq_span80.sum()
adjustments_span80 = np.log(_answer_type_freq_span80 + 1e-12)


class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        # print('11')
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        # print('12')
        fraction = torch.div(numerator, (logvar2.exp()))
        # print('13')
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=1)
        # print('14')
        return kl.mean(dim=0)


class DiverseQA(nn.Module):
    # AblationSWEP_Plus_Discriminator
    def __init__(self, args):
        super(DiverseQA, self).__init__()
        self.model_name = args.bert_model
        self.type_num = 5
        self.bert_model = BertForQuestionAnswering.from_pretrained(
            args.bert_model)
        self.noise_net = nn.Sequential(nn.Linear(args.hidden_size,
                                                 args.hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(args.hidden_size,
                                                 args.hidden_size * 2))
        self.type_classifier = nn.Sequential(nn.Linear(args.hidden_size,
                                                       self.type_num))  # 输入是一个token的noise向量，输出是各个类别的logits
        config = self.bert_model.config
        self.dropout = config.hidden_dropout_prob  # 0.1
        self.Dropout = nn.Dropout(self.dropout)

    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                start_positions=None,
                end_positions=None,
                answer_types=None):

        if start_positions is not None and end_positions is not None:
            assert answer_types is not None
            global adjustments_span80
            _adjustments = torch.from_numpy(adjustments_span80)
            _adjustments = _adjustments.cuda()
            embeddings = self.bert_model.get_input_embeddings()
            encoder = self.bert_model.bert
            with torch.no_grad():
                encoder_inputs = {"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "token_type_ids": token_type_ids}

                outputs = encoder(**encoder_inputs)
                hiddens = outputs[0]
            mask = attention_mask.view(-1)
            indices = (mask == 1)
            mu_logvar = self.noise_net(hiddens)
            mu, log_var = torch.chunk(mu_logvar, 2, dim=-1)
            zs = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
            noise = zs

            prior_mu = torch.ones_like(mu)
            # If p < 0.5, sqrt makes variance the larger
            prior_var = torch.ones_like(mu) * sqrt(self.dropout / (1-self.dropout))
            prior_logvar = torch.log(prior_var)

            kl_criterion = GaussianKLLoss()
            h = hiddens.size(-1)
            _mu = mu.view(-1, h)[indices]
            _log_var = log_var.view(-1, h)[indices]
            _prior_mu = prior_mu.view(-1, h)[indices]
            _prior_logvar = prior_logvar.view(-1, h)[indices]

            kl = kl_criterion(_mu, _log_var, _prior_mu, _prior_logvar)

            inputs_embeds = embeddings(input_ids)

            total_inputs_embeds = torch.cat([inputs_embeds * noise, inputs_embeds], 0)
            # total_inputs_embeds = torch.cat([inputs_embeds, inputs_embeds], 0)
            total_attention_mask = torch.cat([attention_mask, attention_mask], 0)
            total_token_type_ids = torch.cat([token_type_ids, token_type_ids], 0)
            total_start_positions = torch.cat([start_positions, start_positions], 0)
            total_end_positions = torch.cat([end_positions, end_positions], 0)
            total_inputs = {"inputs_embeds": total_inputs_embeds,
                            "attention_mask": total_attention_mask,
                            "token_type_ids": total_token_type_ids,
                            "start_positions": total_start_positions,
                            "end_positions": total_end_positions}
            outputs = self.bert_model(**total_inputs)
            # nll = outputs[0]
            qa_loss = outputs[0]

            type_logits = self.type_classifier(noise)
            loss_fct = CrossEntropyLoss(ignore_index=self.type_num)
            _answer_types = answer_types.contiguous().view(-1)[indices]
            _type_logits = type_logits.contiguous().view(-1, type_logits.size(-1))[indices] + _adjustments
            type_loss = loss_fct(_type_logits, _answer_types)
            loss = qa_loss + type_loss
            # loss = qa_loss
            return (loss, kl)

        else:
            inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids,
                      "start_positions": start_positions,
                      "end_positions": end_positions}

            outputs = self.bert_model(**inputs)
            return outputs
