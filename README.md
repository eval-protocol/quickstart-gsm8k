# Fine Tuning a GSM8K Evaluator with Eval Protocol

Evaluate and train math reasoning models against GSM8K-style prompts using Eval Protocol. This repository provides a minimal, working setup you can run locally and then scale to Reinforcement Fine Tuning (RFT). It evaluates GSM8K-style math answers and can optionally kick off RFT. The main components are:

- **Eval Protocol** - Orchestrates rollout execution, local UI, evaluator packaging, and RFT launcher
- **SingleTurnRolloutProcessor** - Performs a single LiteLLM completion per row
- **Evaluation** - Parses the first digits inside `<answer>...</answer>` and compares to ground truth

Each dataset row contains a conversation ending with a model answer that should include `<answer>...</answer>`. We extract the first digit sequence and compare it against the ground truth’s `<answer>...</answer>` contents to compute a 0/1 score.

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

The RFT create process below automatically reads and uploads these secrets to Fireworks.

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

You should see a run that completes and opens a local dashboard on `http://localhost:8000`. A typical run looks like:

```
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

## Single Command to Train

To kick off training on Fireworks using your local evaluator:

```bash
eval-protocol create rft \
  --base-model accounts/fireworks/models/qwen3-0p6b
```

This command:
1. **🔐 Uploads Secrets** – Reads your env (e.g., `FIREWORKS_API_KEY`) and creates/updates Fireworks secrets
2. **📦 Uploads Evaluator** – Packages and uploads your evaluation code (e.g., `evaluation.py::gsm8k_example`)
3. **⏳ Waits for Build** – Polls evaluator status until ACTIVE (timeout: 10 minutes)
4. **📊 Creates Dataset** – Uploads your `gsm8k_sample.jsonl`
5. **🚀 Launches RFT Job** – Starts reinforcement fine-tuning with your evaluator

### Configuration & Troubleshooting

- **Training Parameters**: Defaults from Eval Protocol are used (batch size, epochs, learning rate, etc.). For full RFT flag options, see the Fireworks docs.
- **Changing Evaluators**: If you modify `evaluation.py` and want to re-upload:

```bash
eval-protocol create rft \
  --base-model accounts/fireworks/models/qwen3-0p6b \
  --force
```

- **Evaluator Upload Timing Out**: If your evaluator takes longer than 10 minutes to build, you’ll see:
```
⏰ Timeout after 10.0m - evaluator is not yet ACTIVE

❌ Evaluator is not ready within the timeout period.
📊 Please check the evaluator status at: https://app.fireworks.ai/dashboard/evaluators/<your-evaluator-id>
   Wait for it to become ACTIVE, then run 'eval-protocol create rft' again.
```

### Monitor Training Progress

After successful job creation, you’ll see links like:

```
✅ Created Reinforcement Fine-tuning Job
   name: accounts/<your-account>/reinforcementFineTuningJobs/<job-id>

📊 Dashboard Links:
   Evaluator: https://app.fireworks.ai/dashboard/evaluators/<your-evaluator-id>
   Dataset:   https://app.fireworks.ai/dashboard/datasets/<your-dataset-id>
   RFT Job:   https://app.fireworks.ai/dashboard/fine-tuning/reinforcement/<job-id>
```

Click the **RFT Job** link to view real-time training progress and rollouts.

### Example Successful RFT Output

```
(.venv) (base) derekxu@Mac-3616 quickstart-gsm8k % eval-protocol create rft \
  --base-model accounts/fireworks/models/qwen3-0p6b
INFO:eval_protocol.platform_api:eval_protocol.platform_api: No .env.dev or .env file found. Relying on shell/existing environment variables.
Scanning for evaluation tests...

Found 1 test: gsm8k_example - evaluation.py:32
? Upload this test? Yes
Found 1 API keys to upload as Fireworks secrets...
Ensuring FIREWORKS_API_KEY is registered as a secret on Fireworks for rollout...
INFO:eval_protocol.platform_api:Secret 'FIREWORKS_API_KEY' already exists. Will attempt to update.
INFO:eval_protocol.platform_api:Successfully updated secret 'FIREWORKS_API_KEY' on Fireworks platform.
✓ FIREWORKS_API_KEY secret created/updated on Fireworks.

Uploading evaluator 'evaluation-gsm8k-example' for gsm8k_example...
INFO:eval_protocol.evaluation:Loaded 2 files for metric 'quickstart-gsm8k' from /Users/derekxu/Documents/code/quickstart-gsm8k
INFO:eval_protocol.evaluation:Including entryPoint in payload: evaluation.py::gsm8k_example
INFO:eval_protocol.evaluation:Create API Request Payload: {
  "parent": "accounts/derek-7518aa",
  "evaluator": {
    "displayName": "evaluation-gsm8k-example",
    "description": "Evaluator for evaluation.gsm8k_example",
    "multiMetrics": true,
    "commitHash": "0.0.0.dev1+g789516d.dirty",
    "criteria": [
      {
        "name": "quickstart-gsm8k",
        "type": "CODE_SNIPPETS",
        "description": "Evaluator for evaluation.gsm8k_example"
      }
    ],
    "requirements": "",
    "rollupSettings": {
      "skipRollup": true
    },
    "entryPoint": "evaluation.py::gsm8k_example"
  },
  "evaluatorId": "evaluation-gsm8k-example"
}
INFO:eval_protocol.evaluation:Creating evaluator 'evaluation-gsm8k-example' for account 'derek-7518aa'...
INFO:eval_protocol.evaluation:Creating evaluator at: https://api.fireworks.ai/v1/accounts/derek-7518aa/evaluatorsV2
INFO:eval_protocol.evaluation:Successfully created evaluator 'evaluation-gsm8k-example'
INFO:eval_protocol.evaluation:Creating tar.gz with 0 ignore patterns
INFO:eval_protocol.evaluation:Created /Users/derekxu/Documents/code/quickstart-gsm8k/quickstart-gsm8k.tar.gz (228,806 bytes)
INFO:eval_protocol.evaluation:Requesting upload endpoint for quickstart-gsm8k.tar.gz
INFO:eval_protocol.evaluation:Uploading quickstart-gsm8k.tar.gz to GCS...
INFO:eval_protocol.evaluation:Successfully uploaded quickstart-gsm8k.tar.gz
INFO:eval_protocol.evaluation:Upload validated successfully

✅ Successfully uploaded evaluator: evaluation-gsm8k-example
📊 View in Fireworks Dashboard:
   https://app.fireworks.ai/dashboard/evaluators/evaluation-gsm8k-example

✓ Uploaded/ensured evaluator: evaluation-gsm8k-example
Waiting for evaluator 'evaluation-gsm8k-example' to become ACTIVE...
Polling evaluator status (timeout: 10m, interval: 10s)...
⏳ Evaluator is still building... (0.0m elapsed)
⏳ Evaluator is still building... (0.2m elapsed)
⏳ Evaluator is still building... (0.3m elapsed)
⏳ Evaluator is still building... (0.5m elapsed)
⏳ Evaluator is still building... (0.7m elapsed)
⏳ Evaluator is still building... (0.9m elapsed)
⏳ Evaluator is still building... (1.0m elapsed)
⏳ Evaluator is still building... (1.2m elapsed)
⏳ Evaluator is still building... (1.4m elapsed)
⏳ Evaluator is still building... (1.5m elapsed)
⏳ Evaluator is still building... (1.7m elapsed)
⏳ Evaluator is still building... (1.9m elapsed)
⏳ Evaluator is still building... (2.1m elapsed)
⏳ Evaluator is still building... (2.2m elapsed)
⏳ Evaluator is still building... (2.4m elapsed)
⏳ Evaluator is still building... (2.6m elapsed)
⏳ Evaluator is still building... (2.7m elapsed)
⏳ Evaluator is still building... (2.9m elapsed)
⏳ Evaluator is still building... (3.1m elapsed)
⏳ Evaluator is still building... (3.2m elapsed)
⏳ Evaluator is still building... (3.4m elapsed)
⏳ Evaluator is still building... (3.6m elapsed)
⏳ Evaluator is still building... (3.8m elapsed)
✅ Evaluator is ACTIVE and ready!
✓ Using JSONL from input_dataset: gsm8k_sample.jsonl
✓ Created and uploaded dataset: evaluation-gsm8k-example-dataset-20251110011054
Prepared RFT job for evaluator 'evaluation-gsm8k-example' using dataset 'evaluation-gsm8k-example-dataset-20251110011054'

✅ Created Reinforcement Fine-tuning Job
   name: accounts/derek-7518aa/reinforcementFineTuningJobs/dsfwy9hc

📊 Dashboard Links:
   Evaluator: https://app.fireworks.ai/dashboard/evaluators/evaluation-gsm8k-example
   Dataset:   https://app.fireworks.ai/dashboard/datasets/evaluation-gsm8k-example-dataset-20251110011054
   RFT Job:   https://app.fireworks.ai/dashboard/fine-tuning/reinforcement/dsfwy9hc
```

## How It Works

- **Rollout**: For each dataset row, `SingleTurnRolloutProcessor` calls the model once and appends the assistant message to `row.messages`.
- **Evaluation**: We parse the last assistant message, extract the first digit sequence inside `<answer>...</answer>`, and compare against the ground truth’s `<answer>...</answer>` digits.
- **Score**: Exact match yields `1.0`, otherwise `0.0`.

## Debugging Tips

When your evaluation or training is running, use the local UI to explore:

- **Rollout Overview**: Click the pivot or table views to see overall scores and per-row status.
- **Individual Row Details**: Open a row to inspect prompts, responses, and metadata.
- **Live Log Streaming**: Use “View Logs” to stream logs and troubleshoot any errors.

## Contact Us / Learn More
- **Discord**: `https://discord.gg/mMqQxvFD9A` (join the `#eval-protocol` channel)
- **Eval Protocol Docs**: `https://evalprotocol.io/introduction`
- **Remote Rollout Processor Tutorial**: `https://evalprotocol.io/tutorial/remote-rollout-processor`
- **Fireworks AI Platform**: `https://fireworks.ai`

## Appendix

### Dataset

This repo includes a trimmed `gsm8k_sample.jsonl`. The evaluation references it directly:
```
input_dataset=[<repo>/gsm8k_sample.jsonl]
```

### Notes on Environment and Tools

- The evaluation relies on the Eval Protocol pytest plugin; running `pytest` after installation will auto-load it.
