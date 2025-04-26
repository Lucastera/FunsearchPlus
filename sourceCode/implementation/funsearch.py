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

"""A single-threaded implementation of the FunSearch pipeline."""
from __future__ import annotations

# from collections.abc import Sequence

# RZ: there are multiple errors in the original code
# we should use typing.xxx rather than collections.abc.xxx
from typing import Any, Tuple, Sequence

from implementation import code_manipulation
from implementation import config as config_lib
from implementation import evaluator
from implementation import programs_database
from implementation import sampler
from implementation import profile


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run.

    RZ: The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@funsearch.run', '@funsearch.evolve' respectively.
    The function labeled with '@funsearch.run' is going to evaluate the generated code (like fitness evaluation).
    The function labeled with '@funsearch.evolve' is the function to be searched (like 'greedy' in cap-set).
    This function (_extract_function_names) makes sure that these decorators appears in the specification.
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
    return evolve_functions[0], run_functions[0]


def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        enable_duplicate_check: bool = True,  # 是否啟用重複代碼檢查
        duplicate_check_method: str = "similarity",  # 檢查方法 ("hash" 或 "similarity")
        similarity_threshold: float = 0.9,  # 相似度閾值（僅適用於 "similarity" 方法）
        **kwargs
):
    """Launches a FunSearch experiment.
    Args:
        specification: 問題的模板代碼。
        inputs       : 問題的數據集（見 'bin_packing_utils.py'）。
        config       : 配置文件。
        max_sample_nums: LLM 的最大採樣數量。'None' 表示無限制。
        enable_duplicate_check: 是否啟用重複代碼檢查。
        duplicate_check_method: 檢查方法 ("hash" 或 "similarity")。
        similarity_threshold: 相似度閾值（僅適用於 "similarity" 方法）。
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = programs_database.ProgramsDatabase(config.programs_database, template, function_to_evolve)

    # 初始化完成後，根據參數啟用或禁用重複代碼檢查
    if enable_duplicate_check:
        profiler = profile.Profiler(log_dir=kwargs.get('log_dir', None))
        profiler._evaluated_hashes.clear()
        profiler._evaluated_functions.clear()
    else:
        profiler = profile.Profiler(log_dir=kwargs.get('log_dir', None))
        profiler._evaluated_hashes.clear()
        profiler._evaluated_functions.clear()

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            sandbox_class=class_config.sandbox_class
        ))

    # We send the initial implementation to be analysed by one of the evaluators.
    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(
        initial,
        island_id=None,
        version_generated=None,
        profiler=profiler,
        method=duplicate_check_method,  # 傳遞檢查方法
        threshold=similarity_threshold  # 傳遞相似度閾值
    )

    # Set global max sample nums.
    samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt, max_sample_nums=max_sample_nums, llm_class=class_config.llm_class, log_dir=kwargs.get('log_dir', None))
                for _ in range(config.num_samplers)]

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    for s in samplers:
        s.sample(profiler=profiler, method=duplicate_check_method, threshold=similarity_threshold)
