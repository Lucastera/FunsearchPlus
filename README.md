# FunsearchPlus

An enhanced version of DeepMind's FunSearch framework that improves LLM-driven heuristic algorithm search efficiency for combinatorial optimization problems.

## Overview
FunsearchPlus introduces two key improvements:

1. <b>Code Pruning System</b>: Eliminates redundant samples using:
    + Hash-based detection
    + Similarity detection 
    + AI agent detection

2. <b>Multi-Objective Optimization</b>:Guides LLM generation with multiple objectives:
    + Code structure
    + Code quality
    + Algorithmic innovation
    + Python features utilization
    + Mathematical optimization

## Setup
### Requirements

- Python 3.9+
- Dependencies: 
    + transformers
    + bitsandbytes
    + torch
    + absl-py
    + scipy
    + flask 
    + flask-cors
    + numbu
    + tensorboard
    + tensorboardX

### Installation
```
git clone https://github.com/Lucastera/FunsearchPlus.git
cd FunsearchPlus
pip install -r requirements.txt
```

## Quick Start
```
# Multi-Objective Optimization wilth similarity pruning
python sourceCode/bin_packing_funsearch.py --dataset or3 --strategies quality code_structure algorithm --enable_multi --multi_num 2 --enable_dup_check --dup_method similarity --max_samples 50

# Similarity pruning
python sourceCode/bin_packing_funsearch.py --dataset or3 --enable_dup_check --dup_method similarity --max_samples 50

# FunSearch
python sourceCode/bin_packing_funsearch.py --dataset or3 --max_samples 50
```

### Key Parameters

- `--dataset`: Choose dataset (weibull, or3)
- `--strategies`: Select optimization strategies
- `--enable_multi`: Enable multi-strategy optimization
- `--enable_dup_check`: Enable duplicate code detection
- `--dup_method`: Duplicate checking method (hash, similarity, ai_agent)

## Project Structure

- sourceCode: Main implementation
- result: Experiment results
- multi_objective_evaluation.py: Evaluation Multi-Objective Optimization for generating comparison tables with baselines
- draw_multi_picture.ipynb: Jupyter notebook for generating line charts
- code_pruning_evaluation.ipynb: Evaluation Code Pruning System for generating comparison tables with baselines
