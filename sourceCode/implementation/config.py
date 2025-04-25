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

"""Configuration of a FunSearch experiment."""
from __future__ import annotations

import dataclasses
from typing import Type

from implementation import sampler
from implementation import evaluator


@dataclasses.dataclass(frozen=True)
class MultiStrategyConfig:
    """Configuration for multi-strategy optimization.
    
    Attributes:
      enable_multi_strategy: Whether to enable multi-strategy optimization.
      diversity_mode: Strategy selection mode (0: single, 1: primary+secondary, 2: multi-objective).
      multi_num: Number of strategies to combine in each prompt.
      multi_strategies: List of strategies to choose from.
    """
    enable_multi_strategy: bool = False
    diversity_mode: int = 0
    multi_num: int = 2
    multi_strategies: list[str] = dataclasses.field(default_factory=lambda: ["performance"])
    
    # Define available optimization strategies
    OPTIMIZATION_STRATEGIES: dict = dataclasses.field(default_factory=lambda: {
        "performance": {
            "name": "PERFORMANCE",
            "description": "efficiency and speed optimization",
            "guidance": "Implement O(n) algorithms. Use vectorized operations. Avoid nested loops. Minimize temporary arrays. Prefer in-place operations.",
            "short_name": "PERFORMANCE"
        },
        "algorithm": {
            "name": "ALGORITHM", 
            "description": "novel algorithmic approaches",
            "guidance": "Try greedy, dynamic programming, or approximation methods. Use non-linear weighting. Consider preprocessing and edge cases.",
            "short_name": "ALGORITHM"
        },
        "code_structure": {
            "name": "CODE_STRUCTURE",
            "description": "clean and maintainable code",
            "guidance": "Use descriptive names. Add comments for complex logic. Keep functions concise. Avoid nested conditionals.",
            "short_name": "CODE_STRUCTURE"
        },
        "python_features": {
            "name": "PYTHON_FEATURES",
            "description": "modern Python language features",
            "guidance": "Use broadcasting, built-ins (max, sum). Apply conditional expressions. Use list comprehensions and efficient array operations.",
            "short_name": "PYTHON_FEATURES"
        },
        "memory_usage": {
            "name": "MEMORY_USAGE",
            "description": "memory efficiency",
            "guidance": "Use in-place operations. Prefer views over copies. Use boolean indexing. Reuse memory when possible.",
            "short_name": "MEMORY_USAGE"
        }
    })

@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
    """Configuration of a ProgramsDatabase.

    Attributes:
      functions_per_prompt: Number of previous programs to include in prompts.
      num_islands: Number of islands to maintain as a diversity mechanism.
      reset_period: How often (in seconds) the weakest islands should be reset.
      cluster_sampling_temperature_init: Initial temperature for softmax sampling
          of clusters within an island.
      cluster_sampling_temperature_period: Period of linear decay of the cluster
          sampling temperature.
    """
    functions_per_prompt: int = 2
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration of a FunSearch experiment.

    Attributes:
      programs_database: Configuration of the evolutionary algorithm.
      num_samplers: Number of independent Samplers in the experiment. A value
          larger than 1 only has an effect when the samplers are able to execute
          in parallel, e.g. on different machines of a distributed system.
      num_evaluators: Number of independent program Evaluators in the experiment.
          A value larger than 1 is only expected to be useful when the Evaluators
          can execute in parallel as part of a distributed system.
      samples_per_prompt: How many independently sampled program continuations to
          obtain for each prompt.
    """
    programs_database: ProgramsDatabaseConfig = dataclasses.field(default_factory=ProgramsDatabaseConfig)
    num_samplers: int = 1  # RZ: I just use one samplers
    # num_evaluators: int = 140
    num_evaluators: int = 1  # RZ: I just use one evaluators
    samples_per_prompt: int = 4
    multi_strategy: MultiStrategyConfig = dataclasses.field(default_factory=MultiStrategyConfig)


@dataclasses.dataclass()
class ClassConfig:
    """Implemented by RZ. Configuration of 'class LLM' and 'class SandBox' used in this implementation.
    """
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]

