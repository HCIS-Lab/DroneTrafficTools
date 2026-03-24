# HetroD Tools

Code for the ICRA 2026 paper  
**HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic**

[Paper](https://arxiv.org/abs/2602.03447) | [Dataset](https://hetroddata.github.io/HetroD/)

Yu-Hsiang Chen, Wei-Jer Chang, Christian Kotulla, Thomas Keutgens, Steffen Runde, Tobias Moers, Christoph Klas, Wei Zhan, Masayoshi Tomizuka, Yi-Ting Chen

National Yang Ming Chiao Tung University, UC Berkeley, fka GmbH

## Overview

This repository provides compact tools for converting drone-view traffic datasets into formats used by autonomous driving simulation, motion prediction, and planning pipelines.

Current support:
- Drone datasets: `HetroD`, `inD`, `INTERACTION`, `SinD`
- Output formats: [ScenarioNet](https://github.com/metadriverse/scenarionet), [VBD](https://github.com/SafeRoboticsLab/VBD), [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer)

Core capabilities:
- Ego-centric scenario alignment
- Scenario segmentation from long recordings
- Map and trajectory conversion into ScenarioNet-compatible format

## Structure

```text
drone-tool/
├── scenarionet-converter/
│   ├── hetrod_scene.py
│   ├── inD_scene.py
│   ├── interaction_scene.py
│   └── sind_scene.py
├── scenarionet-VBD-converter/
│   └── convert_scenarionet_to_vbd.py
└── scenarionet-scenariodreamer-converter/
    └── scenarionet_to_scenariodreamer_waymo.py
```

## Installation

Please install the required ScenarioNet and MetaDrive environments first:
- [ScenarioNet](https://github.com/metadriverse/scenarionet)
- [MetaDrive](https://github.com/metadriverse/metadrive)

Then install the common Python dependencies:

```bash
pip install numpy pandas scipy shapely lxml utm tqdm matplotlib omegaconf
```

## Quick Start

Choose `segment_size` from the dataset frame rate and your desired clip length:

```text
segment_size = frame_rate × desired_seconds
```

Frame rates used by the supported datasets:
- `HetroD`: `30 Hz`
- `inD`: `25 Hz`
- `INTERACTION`: `10 Hz`
- `SinD`: `29.97 Hz`

Example:
- `HetroD` at `30 Hz` with `segment_size=273` gives `273 / 30 = 9.1` seconds per clip.

Convert `HetroD` to `ScenarioNet`:

```bash
python scenarionet-converter/hetrod_scene.py \
  --root_dir /path/to/HetroD-dataset-v1.1 \
  --segment_size 273 \
  --output_dir /path/to/output
```

Convert `inD` to `ScenarioNet`:

```bash
python scenarionet-converter/inD_scene.py \
  --root_dir /path/to/inD-dataset-v1.1 \
  --segment_size 228 \
  --output_dir /path/to/output
```

Convert `INTERACTION` to `ScenarioNet`:

```bash
python scenarionet-converter/interaction_scene.py \
  --root_dir /path/to/INTERACTION \
  --segment_size 91 \
  --output_dir /path/to/output
```

Convert `SinD` to `ScenarioNet`:

```bash
python scenarionet-converter/sind_scene.py \
  --root_dir /path/to/SinD \
  --segment_size 273 \
  --output_dir /path/to/output
```

Convert `ScenarioNet` to `VBD`:

```bash
python scenarionet-VBD-converter/convert_scenarionet_to_vbd.py \
  --input_dir /path/to/scenarionet \
  --output_dir /path/to/vbd \
  --frame_rate 30.0 \
  --include_raw
```

Convert `ScenarioNet` to `Scenario Dreamer`:

```bash
python scenarionet-scenariodreamer-converter/scenarionet_to_scenariodreamer_waymo.py \
  --input_dir /path/to/scenarionet \
  --output_dir /path/to/scenariodreamer \
  --cfg_path cfgs/dataset/waymo_autoencoder_temporal.yaml \
  --train_ratio 0.7 \
  --val_ratio 0.2 \
  --test_ratio 0.1 \
  --seed 0
```

Place this script inside Scenario Dreamer's `scripts/` directory before running.

## Output

ScenarioNet conversion produces:
- Scenario `.pkl` files
- `dataset_summary.pkl`
- `dataset_mapping.pkl`

## Citation

```bibtex
@inproceedings{hetrod,
  title={HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic},
  author={Yu-Hsiang Chen and Wei-Jer Chang and Christian Kotulla and Thomas Keutgens and Steffen Runde and Tobias Moers and Christoph Klas and Wei Zhan and Masayoshi Tomizuka and Yi-Ting Chen},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```
