from typing import List, Optional, Union, Tuple

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from tqdm import tqdm


from .gen_util import ContrastGenerationMixin

#! Combine assist LLM with the target LLM
class ContrastLLM(torch.nn.Module, ContrastGenerationMixin):
    def __init__(self, basellm : AutoModelForCausalLM, assist_llm : AutoModelForCausalLM, weight : float, top_logit_filter=0.0) -> None:
        super().__init__()
        self.basellm = basellm
        self.assist_llm = assist_llm
        self.weight = weight
        self.device = self.basellm.device
        self.config = self.basellm.config
        self.generation_config = basellm.generation_config
        self.top_logit_filter = top_logit_filter
        self.tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
    
    def get_loss(self, logits, labels=None, attention_mask=None, reduciton='mean'):
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if reduciton == 'batchmean':
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss.sum(dim=-1) / (attention_mask.sum(dim=-1))
            else:
                loss_fct = CrossEntropyLoss(reduction=reduciton)
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
        return loss


    # Greedy decoding with progress tracking
    def greedy_generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 20):
        self.eval()
        generated_ids = input_ids.clone()

        with tqdm(total=max_new_tokens, desc="Generating (Greedy)", unit="token") as pbar:
            for _ in range(max_new_tokens):
                logits = self(input_ids=generated_ids).logits
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                pbar.update(1)  # Update progress bar

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        return generated_ids

    # Sampling-based decoding with progress tracking
    def sampling_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50
    ):
        self.eval()
        generated_ids = input_ids.clone()

        with tqdm(total=max_new_tokens, desc="Generating (Sampling)", unit="token") as pbar:
            for _ in range(max_new_tokens):
                logits = self(input_ids=generated_ids).logits
                logits = logits[:, -1, :] / temperature

                probs = F.softmax(logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                sampled_token_id = top_k_indices[0, torch.multinomial(top_k_probs[0], 1)]  # Sample from top-k

                sampled_token_id = sampled_token_id.unsqueeze(1)  # Adjust shape for concatenation
                generated_ids = torch.cat([generated_ids, sampled_token_id], dim=-1)
                pbar.update(1)  # Update progress bar

                if sampled_token_id.squeeze().item() == self.tokenizer.eos_token_id:
                    break

        return generated_ids





    # adapted from LLamaForCausalLM
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        #! This forward only returns the logits, never use this for training

        output_attentions = False
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.basellm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        assit_outputs = self.assist_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # outlogits = outputs.logits        
        baselogits = outputs.logits
        assist_logits = assit_outputs.logits

        if self.top_logit_filter > 0.0:
            baselogits, mask, probs_thresh = self.relative_top_filter(baselogits, self.top_logit_filter)
            #! lowprob-0 filter
            assist_logits[mask] = 0
            logits = baselogits + self.weight * assist_logits
        else:
            logits = baselogits + self.weight * assist_logits

        loss = None
        loss = self.get_loss(logits, labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )