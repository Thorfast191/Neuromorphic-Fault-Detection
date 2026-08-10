# Neuromorphic Fault Detection

Event-driven machine fault detection using spiking neural networks (SNNs).
Instead of continuously processing sensor data, the system encodes vibration
signals into sparse spike trains and reacts only when the signal changes,
enabling low-power, low-latency fault detection suited to resource-constrained
deployment (IoT, embedded, edge devices).

## Novelty: event-driven efficiency without specialized neuromorphic hardware

Most SNN fault-detection research either stays in pure simulation (energy
savings estimated from spike counts, never measured) or assumes access to
dedicated neuromorphic silicon (Loihi, Akida, SpiNNaker). This project targets
a different, more accessible claim:

1. **Event-driven encoding as the actual mechanism** — a delta (change-based)
   spike encoder feeds a small LIF-based SNN trained with surrogate-gradient
   backpropagation, rather than converting to spikes only after the fact via
   rate coding.
2. **Incipient-fault detection** — the CWRU bearing dataset labels encode
   ordinal fault severity (0.007" → 0.014" → 0.021" = mild → severe) per fault
   type. Evaluation is broken down by severity tier to show the model catches
   faults at their earliest, hardest-to-detect stage, not just the obvious
   severe case.
3. **Real sparsity/energy comparison groundwork** — event-driven sparse
   compute (spike/ESOP counting) is compared against a dense CNN1D baseline,
   with a clearly-labeled literature-based energy estimate now, and an
   explicit path to replace it with real measured power on commodity
   hardware (Arduino via TFLite Micro) in a later phase.

## Project status

**Phase A (current): complete.** A full pipeline — CWRU dataset → delta spike
encoding → LIF-based SNN classifier → training → severity-tier evaluation +
energy/sparsity comparison — is implemented and verified end-to-end.

**Phase B (planned):** real on-device deployment and measured energy on an
Arduino-class microcontroller via TFLite Micro, and online adaptation to
sensor drift via STDP.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

Download CWRU bearing `.mat` files into `data/raw/CWRU/`, either from the
[official CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
or a Kaggle mirror. `find_mat_files` scans `data/raw/CWRU/` recursively, so
any subfolder layout works. Fault files must be named with a
`B007`/`B014`/`B021`/`IR007`/`IR014`/`IR021`/`OR007`/`OR014`/`OR021` prefix
(case-insensitive, e.g. `B007_0.mat` or `IR014_1_175.mat` both work) and the
healthy baseline file must contain "normal" anywhere in its filename (e.g.
`Normal_0.mat` or `Time_Normal_1_098.mat`) — this is the standard convention
most CWRU redistributions (including most Kaggle mirrors) already use.

## Usage

```bash
python train.py --config configs/default.yaml
python train.py --resume                      # continue an interrupted run
python evaluate.py --config configs/default.yaml
```

`train.py` trains the model specified by `model.architecture` in the config
(default: `lif_classifier`) and checkpoints it under `results/<architecture>/`.
`evaluate.py` reproduces the held-out test split, evaluates the trained SNN,
optionally trains the dense CNN1D baseline for comparison, and writes a full
report (per-class/severity-tier breakdowns, confusion matrix, energy/sparsity
comparison) to `results/report/`.

## Training on Google Colab (GPU, disconnect-safe)

Colab wipes the runtime on disconnect, so checkpoints are written to mounted
Google Drive and training resumes from the last completed epoch.

1. Open `main_trial_1.ipynb` in Colab (`Runtime > Change runtime type > GPU`).
2. Run **§0 Environment setup**. It clones this repo if needed, installs
   `snntorch`, mounts Drive, and points results at
   `MyDrive/Neuromorphic-Fault-Detection/results`.
3. Run the notebook top to bottom.

If the session drops, reconnect and re-run §0 → §5. Training continues from the
last checkpoint, restoring model weights, optimizer state, early-stopping
patience, history and RNG state — a resumed run is bit-identical to an
uninterrupted one.

The CWRU `.mat` files ship in the repo under `cwru/`, so no dataset upload is
needed. Checkpoints are written atomically (temp file + rename), so a
disconnect mid-save cannot corrupt the file you resume from.

### Path resolution

Both entry points resolve paths at runtime instead of trusting the config
verbatim, so the same code runs locally and in a fresh clone:

| | resolution order |
|---|---|
| Results | `NFD_RESULTS_DIR` env var → `results.output_dir` |
| Dataset | `NFD_DATA_ROOT` env var → `dataset.root` → first of `cwru/`, `data/raw/CWRU/` containing `.mat` files |

So a Colab shell cell can also run the scripts directly:

```bash
NFD_RESULTS_DIR=/content/drive/MyDrive/Neuromorphic-Fault-Detection/results \
  python train.py --resume
```

## Repository layout

- `datasets/` — CWRU bearing dataset loader and fault-label definitions.
- `preprocessing/`, `features/` — signal filtering, normalization, windowing,
  and classical vibration-analysis feature extraction.
- `encoding/` — spike encoders (rate, latency, temporal, phase, delta,
  population, adaptive, hybrid) with a registry/factory.
- `models/` — LIF neuron, surrogate gradients, spiking linear layer, and the
  `LIFClassifier` network.
- `baselines/` — dense `CNN1DBaseline` used as the energy/accuracy reference
  point.
- `training/` — model-agnostic `Trainer` (resumable), loss/optimizer
  factories, early stopping, checkpointing.
- `utils/paths.py`, `utils/colab.py` — runtime path resolution and the Colab
  Drive-mount bootstrap.
- `evaluation/` — classification metrics, severity-tier/incipient-detection
  breakdown, sparsity/energy estimation, report generation.
- `explainability/` — spike-activity telemetry and interpretability tools.
- `deployment/` — hardware export interfaces (TFLite Micro export is stubbed
  pending Phase B).
- `configs/default.yaml` — the single experiment configuration file.

The repo was trimmed to only what's implemented for Phase A — empty
placeholder modules for later milestones (additional baselines, other
neuron/network architectures, STDP/online adaptation, hardware exports,
augmentation, optimization/quantization, tests, etc.) were removed rather
than left as unimplemented stubs. They'll be added back as Phase B work
(real Arduino deployment, online adaptation) actually needs them.
