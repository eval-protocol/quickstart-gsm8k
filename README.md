# Evaluate GSM8K answers with Eval Protocol

This example runs a single model completion per question and scores the response against the expected answer. Eval Protocol runs the evaluator locally, and Fireworks serves the model.

## Quick Start

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Environment Setup:

Set your Fireworks API key:

```bash
export FIREWORKS_API_KEY=your-fireworks-key-here
```

## Running Locally

The dataset `gsm8k_sample.jsonl` is included in this repository and referenced by the evaluation.

**Terminal 1** – Start the local UI server to view results:

```bash
ep logs
```

**Terminal 2** – Run the evaluation:

```bash
python evaluation.py
```

### Expected Test Output

With `ep logs` running, open `http://localhost:8000` to inspect the results. A typical run looks like:

```
INFO:evaluation:I am beginning to execute GSM8k rollout: early-health-03
INFO:evaluation:I am done executing GSM8k rollout: early-health-03
Runs (Parallel): 100%|██████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.16s/run]
PASSED
================================================================================
📊 LOCAL UI EVALUATION RESULTS
================================================================================
📊 Invocation liquid-others-39:
  📊 Aggregate scores: http://localhost:8000/pivot?...
  📋 Trajectories: http://localhost:8000/table?...
================================================================================
```

## How It Works

- **Rollout**: For each dataset row, `SingleTurnRolloutProcessor` calls the model once and appends the assistant message to `row.messages`. To learn more, see the [Eval Protocol docs](https://evalprotocol.io/reference/rollout-processors#singleturnrolloutprocessor).
- **Evaluation**: We parse the last assistant message, extract the first digit sequence inside `<answer>...</answer>`, and compare against the ground truth’s `<answer>...</answer>` digits.
- **Score**: Exact match yields `1.0`, otherwise `0.0`.

## Debugging Tips

While the evaluation runs, use the local UI to explore:

- **Rollout Overview**: Click the pivot or table views to see overall scores and per-row status.
- **Individual Row Details**: Open a row to inspect prompts, responses, and metadata.
- **Live Log Streaming**: Use “View Logs” to stream logs and troubleshoot any errors.

## Training on Fireworks

Fireworks has retired managed reinforcement fine-tuning. To train with this scoring rule, adapt the evaluator to a reward function in a [Fireworks Training API](https://docs.fireworks.ai/fine-tuning/training-api/introduction) training loop.

## Contact Us / Learn More
- **Discord**: `https://discord.gg/mMqQxvFD9A` (join the `#eval-protocol` channel)
- **Eval Protocol Docs**: `https://evalprotocol.io/introduction`
- **Fireworks AI Platform**: `https://fireworks.ai`
