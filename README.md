# Samaa (سَماع) — Reliable Saudi Voice Entity Capture

**Problem.** A voice agent can finish a call "successfully" while capturing a critical
entity wrong: one wrong digit in a mobile number, ID, or amount. If that value is
silently written to the CRM, the business outcome is wrong even though the call looked fine.

**Goal.** Given one caller turn in which the agent asked for a specific entity type,
return a normalized value and an action (**accept / confirm / re-ask**) that minimizes
**silent errors** at the lowest confirmation cost, on Saudi Arabic telephone audio.

> Status: 🚧 in progress. Results below are placeholders until the final test run.

## Headline result

> At a fixed **15% confirmation rate**, silent entity errors on real phone audio:
> base whisper-small **X%** → Saudi + telephony fine-tune **Y%** (95% CI [a, b]).

## Pipeline

```
Telephone audio → ASR (fine-tuned Whisper) → entity extraction + number normalization
→ validation (format / Luhn checksum / range) → calibrated confidence
→ ACCEPT | CONFIRM | RE-ASK → mock CRM → logged outcome
```

| Entity | Validation | What carries the risk |
|---|---|---|
| National ID / Iqama | 10 digits + prefix + **Luhn checksum** | mostly the checksum |
| Mobile | `05` + 8 digits | ASR confidence |
| Amount | plausibility range | ASR confidence + risk tiers |

## Experiments

**ASR models (same clips, same normalization):**
1. `openai/whisper-small` (base)
2. `openai/whisper-large-v3-turbo` (strong zero-shot)
3. `VohoAI/voho-saudi-stt-small` (public Saudi fine-tune)
4. whisper-small fine-tuned on Saudi speech (clean)
5. whisper-small fine-tuned on Saudi speech + telephone augmentation

Runs 4 and 5 use identical clips, steps, seed and hyperparameters; the only difference
is the augmentation (`configs/train_base.yaml`). One codec is held out of training.

## Results

_Tables and plots are added from `results/` after the final test run._

- ASR: WER / CER on SADA clean, SADA telephone (seen codecs), held-out codec, real phone recordings
- Entities: exact match, digit error rate, extraction failures per entity type
- Risk–coverage: silent error rate vs confirmation rate, per ASR model
- Confidence ablation and reliability diagram
- Failure analysis
- Latency (p50 / p95 per stage)

## Design decisions

- **No LLM.** The entity type is known in advance and the output must be exact. An LLM
  could "repair" an invalid ID into a checksum-valid wrong one, creating a silent error.
- **Rules where correctness is checkable** (parsing, validators, policy); **learned models
  only where needed** (ASR, confidence calibration).
- **Thresholds are tuned on dev speakers only**, frozen, then the test set is run once.

## Data and privacy

- **SADA** (SDAIA, CC BY-NC-SA 4.0) for ASR fine-tuning and evaluation. Fine-tuned weights
  inherit the non-commercial license.
- **In-domain recordings** from consenting speakers reading **synthetic** values only
  (`data/make_speaker_scripts.py`). No real numbers or IDs are ever collected.
  Audio and consent forms are never committed (`.gitignore`).
- Speaker-level split: S01–S03 dev, the rest test.

## Quickstart

```bash
pip install -r requirements.txt          # also needs ffmpeg installed
pytest -q                                # validator + normalization tests

python data/telephony.py --list-codecs   # which phone codecs your ffmpeg supports
python data/make_speaker_scripts.py      # speaker sheets + manifest (synthetic values)
python data/inspect_sada.py --split test --n 3000
python eval/asr_eval.py --model openai/whisper-small --n 200
python eval/asr_eval.py --model openai/whisper-small --n 200 --codec mulaw --snr 15
```

## Repo structure

```
configs/     training config shared by both fine-tuning runs
data/        SADA inspection, telephone augmentation, speaker scripts, recording protocol
train/       Whisper fine-tuning (Day 2)
samaa/       library: normalize, validate, audio, sada; later numbers, extract, confidence, policy
app/         FastAPI + Gradio demo (Day 5)
eval/        ASR eval; later entity eval, risk-coverage, bootstrap
results/     committed JSON/CSV/PNG results
tests/
```

## Limitations (to be completed)

- Assumes callers reject wrong read-backs; some people confirm anything.
- Small in-domain test set; all results reported with speaker-level bootstrap intervals.
- SADA is TV speech with few digit sequences; entity results rely on in-domain recordings.
