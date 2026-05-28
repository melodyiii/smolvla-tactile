<img width="3728" height="2964" alt="setup_photo_inboxpicking" src="https://github.com/user-attachments/assets/b7871b1b-49a3-4534-a463-f31026ca5ccb" />
<img width="1019" height="643" alt="5151d31fdab0c6a001ccef6d5bbbbe8b" src="https://github.com/user-attachments/assets/44a9c05e-5efa-44ae-9598-804ca5ffa0ec" />

# smolvla-tactile

SO-101 tactile data collection and real-robot deployment overlay for LeRobot.

This repository does not re-host the full upstream LeRobot codebase. It only publishes the project-specific extension layer:

- tactile robot integration
- tactile sensor driver
- deployment scripts for `SmolVLA` and `SmolVLA + Tactile`
- data collection entry script
- project documentation

## Upstream dependency

This project is designed to be applied on top of the official LeRobot repository:

- upstream repo: `https://github.com/huggingface/lerobot.git`
- pinned commit: `00b662de02734a6972ec674b8792696ecd1cb28e`

The goal of this repository is to publish only the custom overlay, not a full mirror of upstream.

## Repository layout

```text
smolvla-tactile/
├── docs/
│   ├── DEPLOY_GUIDE.md
│   └── TACTILE_PROJECT_PLAYBOOK.md
├── overlay/
│   ├── deploy/
│   ├── scripts/
│   ├── src/
│   └── test.sh
├── apply_overlay.sh
├── LICENSE
└── README.md
```

## What is included

- `overlay/deploy/`
	tactile deployment and evaluation scripts
- `overlay/scripts/`
	shell entrypoints for deployment and evaluation
- `overlay/src/lerobot/common/robot_devices/robots/tactile_so101.py`
	SO-101 tactile robot wrapper
- `overlay/src/lerobot/common/robot_devices/sensors/tactile.py`
	tactile serial driver
- `overlay/src/lerobot/robots/so_follower/config_so_follower.py`
	`so101_tactile` config extension
- `overlay/src/lerobot/robots/utils.py`
	robot factory extension
- `overlay/src/lerobot/scripts/lerobot_record.py`
	tactile sidecar saving in record loop
- `overlay/test.sh`
	sample tactile collection entry script with environment variables instead of machine-specific paths
- `docs/`
	project and deployment documentation

## What is intentionally excluded

This repository does not publish:

- datasets
- outputs
- checkpoints
- calibration files
- tokens or secrets

## Quick start

### 1. Clone upstream LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 00b662de02734a6972ec674b8792696ecd1cb28e
```

### 2. Clone this overlay repo

```bash
git clone https://github.com/melodyiii/smolvla-tactile.git
```

### 3. Apply the overlay

```bash
bash /path/to/smolvla-tactile/apply_overlay.sh /path/to/lerobot
```

This copies the custom files into the upstream checkout.

### 4. Install LeRobot

Follow the official LeRobot installation guide. A minimal source install is usually:

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e .
pip install -e ".[feetech]"
```

### 5. Data collection

After applying the overlay inside the LeRobot checkout:

```bash
bash test.sh
```

Before running, set or edit:

- robot port
- tactile ports
- teleop port
- calibration directories
- dataset repo id
- task name

### 6. Real-robot deployment

Examples inside the LeRobot checkout:

```bash
bash scripts/run_tactile_vla.sh
bash scripts/run_vla_only.sh
bash scripts/run_eval.sh
```

## Documentation

- `docs/TACTILE_PROJECT_PLAYBOOK.md`
	full project explanation, internship packaging, interview preparation
- `docs/DEPLOY_GUIDE.md`
	deployment and evaluation technical guide

## Notes

- This repository is an overlay publication repo, not a standalone replacement for LeRobot.
- Some files originate from or modify Apache-2.0 licensed upstream LeRobot code. See `LICENSE`.


