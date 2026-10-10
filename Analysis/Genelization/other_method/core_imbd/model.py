import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
import copy
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def from_pretrained(cls, model_name, kwargs):
    """Simple wrapper: always load directly from model_name path or HF name.

    不再使用 cache_dir，也不做本地路径拼接，完全由调用方通过
    scoring_model_name/reference_model_name 控制加载路径。
    """
    return cls.from_pretrained(model_name, **kwargs)


class DiscrepancyEstimator(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 reference_model_name: str=None,
                 scoring_model: AutoModelForCausalLM=None,
                 reference_model: AutoModelForCausalLM=None,
                 scoring_tokenizer: AutoTokenizer=None,
                 reference_tokenizer: AutoTokenizer=None,
                 train_method: str='DDL',
                 pretrained_ckpt: str=None,
                 ):
        super().__init__()
        assert train_method in ['DDL', 'SPO'], 'train_method should be DDL or SPO.'
        self.train_method = train_method
        if pretrained_ckpt is not None:
            self.load_pretrained(pretrained_ckpt)
        else:
            if scoring_model_name is not None:
                self.scoring_model_name = scoring_model_name
                # falcon 系列模型使用 bfloat16，更稳定
                scoring_dtype = torch.bfloat16 if 'falcon' in scoring_model_name.lower() else torch.bfloat16
                self.scoring_model = from_pretrained(AutoModelForCausalLM,
                                                     scoring_model_name,
                                                     kwargs=dict(torch_dtype=scoring_dtype))
                self.scoring_tokenizer = from_pretrained(AutoTokenizer,
                                                         scoring_model_name,
                                                         kwargs={'padding_side': 'right',
                                                                 'use_fast': True if 'facebook/opt-' not in scoring_model_name else False},
                                                         )
            else:
                if scoring_model is None or scoring_tokenizer is None:
                    raise ValueError('You should provide scoring_model_name or scoring_model and scoring_tokenizer.')
                self.scoring_model = scoring_model
                self.scoring_tokenizer = scoring_tokenizer
                self.scoring_model_name = scoring_model.config._name_or_path
            if self.scoring_tokenizer.pad_token is None:
                self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
                self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

            if reference_model_name is not None:
                reference_dtype = torch.bfloat16 if 'falcon' in reference_model_name.lower() else torch.bfloat16
                self.reference_model = from_pretrained(AutoModelForCausalLM,
                                                       reference_model_name,
                                                       kwargs=dict(torch_dtype=reference_dtype))
                self.reference_tokenizer = from_pretrained(AutoTokenizer,
                                                           reference_model_name,
                                                           kwargs={'padding_side': 'right',
                                                                   'use_fast': True if 'facebook/opt-' not in reference_model_name else False},
                                       )
                self.reference_model_name = reference_model_name
            else:
                if reference_model is None and reference_tokenizer is None:
                    # 不显式传入 reference 模型时：
                    # - DDL: 完全不使用 reference 模型
                    # - SPO: 使用当前 scoring 模型的输出作为 "reference"，避免再加载/拷贝一份 7B 模型，节省显存
                    self.reference_model = None
                    self.reference_tokenizer = None
                    self.reference_model_name = None
                elif reference_model is not None and reference_tokenizer is not None:
                    self.reference_model = reference_model
                    self.reference_tokenizer = reference_tokenizer
                    self.reference_model_name = reference_model.config._name_or_path
                else:
                    raise ValueError('You should provide reference_model and reference_tokenizer at the same time.')

            if self.reference_tokenizer is not None:
                if self.reference_tokenizer.pad_token is None:
                    self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token
                    self.reference_tokenizer.pad_token_id = self.reference_tokenizer.eos_token_id
            
    def add_lora_config(self, lora_config: LoraConfig):
        self.lora_config = lora_config
        self.scoring_model = get_peft_model(self.scoring_model, self.lora_config)

    def save_pretrained(self, save_directory):
        """
        Save the model's state_dict to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        # torch.save(self.state_dict(), os.path.join(save_directory, "model.bin"))
        self.scoring_model.save_pretrained(os.path.join(save_directory, "scoring_model"))
        self.scoring_tokenizer.save_pretrained(os.path.join(save_directory, "scoring_model"))
        if self.reference_model is not None:
            self.reference_model.save_pretrained(os.path.join(save_directory, "reference_model"))
            self.reference_tokenizer.save_pretrained(os.path.join(save_directory, "reference_model"))

    def load_pretrained(self, load_directory):
        """
        Load the model's state_dict from the specified directory.
        """
        if not os.path.exists(load_directory):
            raise ValueError(f"Directory {load_directory} does not exist.")

        self.scoring_model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(load_directory, "scoring_model"), torch_dtype=torch.bfloat16)
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "scoring_model"))
        self.scoring_model_name = self.scoring_model.config._name_or_path

        if os.path.exists(os.path.join(load_directory, "reference_model")):
            self.reference_model = AutoModelForCausalLM.from_pretrained(os.path.join(load_directory, "reference_model"), torch_dtype=torch.bfloat16)
            self.reference_tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "reference_model"))
            self.reference_model_name = self.reference_model.config._name_or_path
        else:
            self.reference_model = None
            self.reference_tokenizer = None
            self.reference_model_name = None

        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id
        if self.reference_tokenizer is not None:
            if self.reference_tokenizer.pad_token is None:
                self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token
                self.reference_tokenizer.pad_token_id = self.reference_tokenizer.eos_token_id
        

    def get_sampling_discrepancy_analytic(self, reference_logits, scoring_logits, labels, attention_mask):

        if reference_logits.size(-1) != scoring_logits.size(-1):
            vocab_size = min(reference_logits.size(-1), scoring_logits.size(-1))
            reference_logits = reference_logits[:, :, :vocab_size]
            scoring_logits = scoring_logits[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == scoring_logits.ndim - 1 else labels
        lprobs_score = torch.log_softmax(scoring_logits, dim=-1)
        probs_ref = torch.softmax(reference_logits, dim=-1)
        
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

        mask = attention_mask[:, 1:].float()  # [bsz, seq_len-1], 1 for non-pad, 0 for pad
        log_likelihood_sum = (log_likelihood * mask).sum(dim=-1)  # [bsz], sum over non-pad tokens
        mean_ref_sum = (mean_ref * mask).sum(dim=-1)  # [bsz], sum over non-pad tokens
        var_ref_sum = (var_ref * mask).sum(dim=-1)  # [bsz], sum over non-pad tokens
        discrepancy = (log_likelihood_sum - mean_ref_sum) / (var_ref_sum.sqrt() + 1e-8)  # [bsz], avoid division by zero
        
        return discrepancy, log_likelihood_sum

    def get_discrepancy_of_scoring_and_reference_models(self,
                                                        input_ids_for_scoring_model,
                                                        attention_mask_for_scoring_model,
                                                        input_ids_for_reference_model=None,
                                                        attention_mask_for_reference_model=None,
                                                        ) -> dict:
        labels = input_ids_for_scoring_model[:, 1:] # shape: [bsz, sentence_len - 1]
        scoring_logits = self.scoring_model(input_ids_for_scoring_model,
                                            attention_mask=attention_mask_for_scoring_model).logits[:, :-1, :]

        # 计算当前 scoring 模型的 discrepancy / logprob
        discrepancy_score, logprob_score = self.get_sampling_discrepancy_analytic(
            scoring_logits,
            scoring_logits,
            labels,
            attention_mask=attention_mask_for_scoring_model,
        )

        # 处理 reference：
        # 1) 显式提供了 reference_input_ids + reference_model：正常跑一遍 reference 模型
        # 2) 未提供 reference_input_ids，但 train_method=='SPO'：
        #    使用 scoring_logits 的 detach 版本作为 "reference"，即自引导的 baseline，避免额外显存占用
        # 3) 其它情况（如 DDL）：不返回 reference 相关内容
        if input_ids_for_reference_model is not None and self.reference_model is not None:
            with torch.no_grad():
                reference_labels = input_ids_for_reference_model[:, 1:]
                assert torch.all(reference_labels == labels), "Tokenizer is mismatch."
                reference_logits = self.reference_model(
                    input_ids_for_reference_model,
                    attention_mask=attention_mask_for_reference_model,
                ).logits[:, :-1, :]
            discrepancy_ref, logprob_ref = self.get_sampling_discrepancy_analytic(
                reference_logits,
                reference_logits,
                labels,
                attention_mask=attention_mask_for_reference_model,
            )
        elif self.train_method == 'SPO':
            # Self-Preference: 直接用当前 scoring 的输出作为 reference baseline（梯度不回传）
            discrepancy_ref, logprob_ref = (
                discrepancy_score.detach(),
                logprob_score.detach(),
            )
        else:
            discrepancy_ref, logprob_ref = None, None

        return {
            'scoring_discrepancy': discrepancy_score,
            'scoring_logprob': logprob_score,
            'reference_discrepancy': discrepancy_ref,
            'reference_logprob': logprob_ref,
        }
    
    def forward(self,
                scoring_human_input_ids,
                scoring_human_attention_mask,
                scoring_llm_input_ids,
                scoring_llm_attention_mask,
                reference_human_input_ids=None,
                reference_human_attention_mask=None,
                reference_llm_input_ids=None,
                reference_llm_attention_mask=None,
                ) -> dict:
        human_output = self.get_discrepancy_of_scoring_and_reference_models(
            input_ids_for_scoring_model=scoring_human_input_ids,
            attention_mask_for_scoring_model=scoring_human_attention_mask,
            input_ids_for_reference_model=reference_human_input_ids,
            attention_mask_for_reference_model=reference_human_attention_mask,
        )
        llm_output = self.get_discrepancy_of_scoring_and_reference_models(
            input_ids_for_scoring_model=scoring_llm_input_ids,
            attention_mask_for_scoring_model=scoring_llm_attention_mask,
            input_ids_for_reference_model=reference_llm_input_ids,
            attention_mask_for_reference_model=reference_llm_attention_mask,
        )
        
        return {
            'scoring_human_discrepancy': human_output['scoring_discrepancy'],
            'scoring_human_logprob': human_output['scoring_logprob'],
            'scoring_llm_discrepancy': llm_output['scoring_discrepancy'],
            'scoring_llm_logprob': llm_output['scoring_logprob'],
            'reference_human_discrepancy': human_output['reference_discrepancy'],
            'reference_human_logprob': human_output['reference_logprob'],
            'reference_llm_discrepancy': llm_output['reference_discrepancy'],
            'reference_llm_logprob': llm_output['reference_logprob'],
        }