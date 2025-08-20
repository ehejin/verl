# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", reward_config=None) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.reward_config=reward_config

    def tokenizer_fingerprint(self):
        """
        Return a small dict describing the worker's tokenizer (and, if present,
        the rollout engine tokenizer). Safe to send over Ray.
        """
        import os, hashlib

        def _one(tok):
            if tok is None:
                return None
            added = getattr(tok, "get_added_vocab", lambda: {})()
            init_kwargs = getattr(tok, "init_kwargs", {})
            tok_file = init_kwargs.get("tokenizer_file", None)
            md5 = None
            if tok_file and os.path.exists(tok_file):
                try:
                    with open(tok_file, "rb") as f:
                        md5 = hashlib.md5(f.read()).hexdigest()
                except Exception:
                    md5 = None
            return {
                "name_or_path": getattr(tok, "name_or_path", None),
                "vocab_size": getattr(tok, "vocab_size", None),
                "added_vocab_len": len(added),
                "total_size": (getattr(tok, "vocab_size", 0) + len(added)),
                "pad_id": getattr(tok, "pad_token_id", None),
                "eos_id": getattr(tok, "eos_token_id", None),
                "bos_id": getattr(tok, "bos_token_id", None),
                "unk_id": getattr(tok, "unk_token_id", None),
                "tokenizer_json_md5": md5,
            }

        # Primary tokenizer held by the worker (what compute_log_prob/update_actor use)
        primary = _one(getattr(self, "tokenizer", None))

        # If the rollout engine (e.g., vLLM) owns its own tokenizer, include it too
        engine_tok = None
        for attr in ["rollout_engine", "vllm_engine", "engine", "inference_engine"]:
            eng = getattr(self, attr, None)
            tok = getattr(eng, "tokenizer", None) if eng is not None else None
            if tok is not None:
                engine_tok = _one(tok)
                break

        return {"worker_primary": primary, "worker_engine": engine_tok}


    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        print("\n\n\n\n\n\n")
        print("[REWARD MANAGER TOKENIZER CHECKKK]", self.tokenizer_fingerprint())

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            tok_len = len(self.tokenizer)

            # After you compute valid_response_ids
            mx = int(valid_response_ids.max().item()) if valid_response_ids.numel() else -1
            mn = int(valid_response_ids.min().item()) if valid_response_ids.numel() else  0
            if mx >= tok_len or mn < 0:
                print(f"[TOKEN RANGE MISMATCH] min={mn} max={mx} vs tokenizer_len={tok_len}")
                # Optional: dump first few offending ids
                print("ids_head:", valid_response_ids[:32].tolist())

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            if self.reward_config is not None:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    reward_config=self.reward_config
                )
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info
                )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
