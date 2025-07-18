# AdvWave - Audio Adversarial Attack Framework

## Overview
AdvWave is a research framework for conducting adversarial attacks against audio-based AI models. It supports multiple attack methods and evaluation metrics for assessing attack effectiveness.

## Features
- Supports multiple target models: GPT4-o, Qwen2, SpeechGPT, AnyGPT, Omni-Speech
- Multiple attack methods:
  - Text-based: GCG, Beast, AutoDan
  - Audio-based: Custom audio attacks
  - Transfer attacks
- Comprehensive evaluation:
  - Text-based evaluation using LLMs (Llama3, SorryBench)
  - Audio-based evaluation
  - Stealthiness assessment

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run basic attack:
```bash
python main.py --model gpt4o --dataset advbench --jailbreak audio_ours
```

## Core Modules
- `main.py`: Main entry point for attacks and evaluations
- `jailbreak.py`: Implements various attack methods
- `audio_stealthiness_judge.py`: Evaluates attack stealthiness
- `qwen_judge.py`: Qwen-specific evaluation
- `compute_score.py`: Scoring utilities
- `models_audio/`: Target model implementations
- `evaluation_model/`: Evaluation model implementations
- `dataload/`: Data loading utilities

## Parameters
Key command-line arguments:
- `--model`: Target model (gpt4o, qwen2, speechgpt, etc.)
- `--dataset`: Dataset (advbench)
- `--jailbreak`: Attack method (text_gcg, audio_ours, etc.)
- `--temperature`: Generation temperature
- `--text_evaluation_model`: Text evaluator (llama3, sorrybench)
- `--audio_evaluation_model`: Audio evaluator (gpt4o, qwen)
- `--start/--end`: Sample range to process
- `--idstr`: Run identifier for output organization

## Examples
1. Universal audio attack:
```bash
python main.py --model speechgpt --dataset advbench --jailbreak audio_ours_universal --num_universal 5
```

2. Text-based attack with custom evaluation:
```bash
python main.py --model qwen2 --dataset advbench --jailbreak text_gcg --text_evaluation_model llama3
```

## Output
Results are saved in `output/[dataset]/[jailbreak]/[model]/[idstr]` with:
- JSONL logs containing prompts, responses, and scores
- Audio files (for audio attacks)
- Evaluation metrics