# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from __future__ import annotations
from abc import ABC, abstractmethod
import os 
import json
import hashlib
import logging
from difflib import SequenceMatcher  # 用於計算相似度
from openai import OpenAI

from typing import Collection, Sequence, Type
import numpy as np
import time

import http.client

from implementation import evaluator
from implementation import programs_database

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = 'https://api.bltcy.ai/v1'

class LLM(ABC):
    """Language model that predicts continuation of provided source code.

    RZ: The sampled function code must be trimmed! Especially using instruct-based LLM.
    -For example, the sampled function code (with description) is:
    ------------------------------------------------------------------------------------------------------------------
    Here is the function.
    def priority_v2(..., ...) -> Any:
        a = np.array([1, 2, 3])
        if len(a) > 2:
            return a / a.sum()
        else:
            return a / a.mean()
    This function is going to ..., and returns ...[Descriptions by LLM]
    ------------------------------------------------------------------------------------------------------------------
    -The descriptions above the function's signature, and the function's signature must be removed.
    -The above code must be trimmed as follows:
    ------------------------------------------------------------------------------------------------------------------
        a = np.array([1, 2, 3])
            if len(a) > 2:
                return a / a.sum()
            else:
                return a / a.mean()
        Here is the function. This function is going to ..., and returns ...[Descriptions by LLM]
    ------------------------------------------------------------------------------------------------------------------
    Please note that the indent must be preserved. And the additional descriptions can also be preserved,
    which will be trimmed by Evaluator.
    """

    def __init__(self, samples_per_prompt: int, multi_strategy_config=None) -> None:
        self._samples_per_prompt = samples_per_prompt
        self._multi_strategy_config = multi_strategy_config
        self._additional_prompt = ""
        self._current_strategy_prompt = ""
        
    def _get_strategy_prompt(self) -> str:
        """Get prompt based on selected optimization strategies."""
        if not self._multi_strategy_config or not self._multi_strategy_config.enable_multi_strategy:
            return "", []
        
        strategies = self._multi_strategy_config.OPTIMIZATION_STRATEGIES
        selected_strategies = []
        selected_strategy_names = []  # 记录选择的策略名称
        
        multi_num = min(self._multi_strategy_config.multi_num, 
                    len(self._multi_strategy_config.multi_strategies))
        
   
        if multi_num > 0 and self._multi_strategy_config.multi_strategies:
            # Always include primary strategy
            primary = self._multi_strategy_config.multi_strategies[0]
            if primary in strategies:
                selected_strategies.append(strategies[primary])
                selected_strategy_names.append(primary)  # 记录策略名称
            
            # Add random secondary strategies
            secondary_options = self._multi_strategy_config.multi_strategies[1:]
            if secondary_options and multi_num > 1:
                num_secondary = min(multi_num - 1, len(secondary_options))
                selected_secondary = np.random.choice(
                    secondary_options, num_secondary, replace=False).tolist()
                for s in selected_secondary:
                    if s in strategies:
                        selected_strategies.append(strategies[s])
                        selected_strategy_names.append(s)  # 记录策略名称
                    
        # 构建提示词
        if not selected_strategies:
            return "", []
        
        # Format: FOCUS ON X, Y AND Z: Create a solution that balances...
        strategy_names = [s["short_name"] for s in selected_strategies]
        strategy_descriptions = [s["description"] for s in selected_strategies]
        strategy_guidances = [s["guidance"] for s in selected_strategies]
        if len(strategy_names) == 1:
            focus_list = strategy_names[0]
            desc_list = strategy_descriptions[0]
        elif len(strategy_names) == 2:
            focus_list = " AND ".join(strategy_names)
            desc_list = " and ".join(strategy_descriptions)
        else:
            focus_list = ", ".join(strategy_names[:-1]) + f" AND {strategy_names[-1]}"
            desc_list = ", ".join(strategy_descriptions[:-1]) + f" and {strategy_descriptions[-1]}"
        
        guidance_combined = " ".join(strategy_guidances)
        if len(strategy_names) == 1:
            return f"FOCUS ON {focus_list}: {guidance_combined}", selected_strategy_names
            
        else:
            return f"FOCUS ON {focus_list}: Create a solution that balances {desc_list}. {guidance_combined}", selected_strategy_names

    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
    """Node that samples program continuations and sends them for analysis.
    """
    _global_samples_nums: int = 1  # RZ: this variable records the global sample nums

    def __init__(
            self,
            database: programs_database.ProgramsDatabase,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
            multi_strategy_config=None,
            log_dir: str | None = None,
            api_key: str | None = None,  
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt, multi_strategy_config)
        self._max_sample_nums = max_sample_nums
        self._evaluated_hashes = set()  # 存儲已評估代碼的哈希值
        self._evaluated_functions = []  # 存儲已評估代碼的完整內容
        self._json_dir = os.path.join(log_dir, 'samples')
        self._api_key = api_key 
    
    def _load_evaluated_hashes(self):
        """從日志文件夾加載已評估代碼的哈希值和完整內容。"""
        if not self._json_dir:
            return
        for file_name in os.listdir(self._json_dir):
            if file_name.endswith('.json'):
                with open(os.path.join(self._json_dir, file_name), 'r') as json_file:
                    content = json.load(json_file)
                    function_code = content.get('function', '')
                    code_hash = hashlib.sha256(function_code.encode()).hexdigest()
                    self._evaluated_hashes.add(code_hash)
                    self._evaluated_functions.append(function_code)

    def is_duplicate_by_hash(self, function_code: str) -> bool:
        """檢查代碼是否已評估過（基於哈希值）。"""
        code_hash = hashlib.sha256(function_code.encode()).hexdigest()
        return code_hash in self._evaluated_hashes

    def is_duplicate_by_similarity(self, function_code: str, threshold: float = 0.9) -> bool:
        """檢查代碼是否已評估過（基於相似度）。"""
        for evaluated_code in self._evaluated_functions:
            similarity = SequenceMatcher(None, function_code, evaluated_code).ratio()
            if similarity >= threshold:
                return True
        return False
    
    def is_duplicate_by_ai_agent(self, function_code: str, threshold: float = 0.9) -> bool:
        """
        檢查代碼是否已評估過（基於 AI Agent 評估）。
        Args:
            function_code: 要檢查的代碼。
            threshold: 相似性分數的閾值，默認為 0.9。
        Returns:
            如果代碼被認為是重複的，返回 True；否則返回 False。
        """
        for evaluated_code in self._evaluated_functions:
            try:
                # 使用 AI Agent 比較代碼相似性

                client = OpenAI(api_key=self._api_key, base_url=BASE_URL)


                message = [
                            {"role": "system", "content": "You are a code similarity analyzer. Compare the two code snippets and return only a similarity score from [0, 1] based on their aim and logic. 1 means identical, 0 means completely different. Output should be a single number."},
                            {"role": "user", "content": f"Code 1:\n{function_code}\n\nCode 2:\n{evaluated_code}"}]

                response = client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=message,
                    stream=False,
                )

                words_to_remove = ['similarity','score', 'Score', 'Similarity',':',  ' ']
                score_text = response.choices[0].message.content.strip()
                print(f"AI Agent 評估結果: {score_text}")

                for word in words_to_remove:
                    score_text = score_text.replace(word, '')
                score = float(score_text)
                if score >= threshold:
                    return True
            except Exception as e:
                logging.error(f"AI Agent 評估失敗: {e}")
                continue
        return False
    

    def sample(self, **kwargs):
        """Continuously gets prompts, samples programs, sends them for analysis.
        """
        method = kwargs["method"]  # 必須從 kwargs 中獲取
        threshold = kwargs["threshold"]  # 必須從 kwargs 中獲取

        
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break

            

            prompt = self._database.get_prompt()
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt
            
            for sample in samples:
                function_code = '''def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    """Improved version of `priority_v0`."""'''+ sample  # RZ: add function signature
                

                
                self._global_sample_nums_plus_one()  # RZ: add _global_sample_nums
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)

                if method == "hash" and self.is_duplicate_by_hash(function_code):  # 基於哈希值檢查
                    print("#########################################")
                    print("#  Skipping duplicate function (hash):  #")
                    print("#########################################")
                    print(function_code)
                    continue
                elif method == "similarity" and self.is_duplicate_by_similarity(function_code, threshold):  # 基於相似度檢查
                    print("###############################################")
                    print("#  Skipping duplicate function (similarity):  #")
                    print("###############################################")
                    print(function_code)
                    continue
                elif method == "ai_agent" and self.is_duplicate_by_ai_agent(function_code, threshold):  # 基於 AI Agent 檢查
                    print("###############################################")
                    print("#  Skipping duplicate function (AI Agent):    #")
                    print("###############################################")
                    print(function_code)
                    continue

                # 如果不是重複代碼，記錄哈希值和完整內容
                code_hash = hashlib.sha256(function_code.encode()).hexdigest()
                self._evaluated_hashes.add(code_hash)
                self._evaluated_functions.append(function_code)


                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1
