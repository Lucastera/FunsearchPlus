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

from typing import Collection, Sequence, Type
import numpy as np
import time

from implementation import evaluator
from implementation import programs_database


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
        
        # if multi_num > 0 and self._multi_strategy_config.multi_strategies:
        #     selected_keys = np.random.choice(
        #         self._multi_strategy_config.multi_strategies, 
        #         multi_num, replace=False).tolist()
        #     for key in selected_keys:
        #         if key in strategies:
        #             selected_strategies.append(strategies[key])
        #             selected_strategy_names.append(key)  # 记录策略名称
                    
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
            multi_strategy_config=None
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt, multi_strategy_config)
        self._max_sample_nums = max_sample_nums

    def sample(self, **kwargs):
        """Continuously gets prompts, samples programs, sends them for analysis.
        """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break

            prompt = self._database.get_prompt()
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt
            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                self._global_sample_nums_plus_one()  # RZ: add _global_sample_nums
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
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
