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

The ScenarioNet converters now normalize all supported datasets to a Waymo-like scenario layout:
- `91` timesteps
- `10 Hz`
- `ts = 0.0 ... 9.0`
- `current_time_index = 10`

Source frame rates used internally:
- `HetroD`: `30 Hz`
- `inD`: `25 Hz`
- `INTERACTION`: `10 Hz`
- `SinD`: `29.97 Hz`

The converters compute the correct raw window from the dataset frame rate automatically. You usually do not need to pass `--segment_size`; if you do, it is ignored when Waymo alignment is enabled.

Convert `HetroD` to `ScenarioNet`:

```bash
python scenarionet-converter/hetrod_scene.py \
  --root_dir /path/to/HetroD-dataset-v1.1 \
  --output_dir /path/to/output
```

Convert `inD` to `ScenarioNet`:

```bash
python scenarionet-converter/inD_scene.py \
  --root_dir /path/to/inD-dataset-v1.1 \
  --output_dir /path/to/output
```

Convert `INTERACTION` to `ScenarioNet`:

```bash
python scenarionet-converter/interaction_scene.py \
  --root_dir /path/to/INTERACTION \
  --output_dir /path/to/output
```

Convert `SinD` to `ScenarioNet`:

```bash
python scenarionet-converter/sind_scene.py \
  --root_dir /path/to/SinD \
  --output_dir /path/to/output
```

Convert `ScenarioNet` to `VBD`:

```bash
python scenarionet-VBD-converter/convert_scenarionet_to_vbd.py \
  --input_dir /path/to/scenarionet \
  --output_dir /path/to/vbd \
  --include_raw
```

Notes:
- `convert_scenarionet_to_vbd.py` now infers frame rate from `metadata["ts"]` by default.
- Only pass `--frame_rate` if you explicitly want to override the inferred value.

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

Notes:
- The Scenario Dreamer converter now expects ScenarioNet inputs that are already normalized to `10 Hz / 91` steps.
- It uses `metadata["sdc_id"]` instead of assuming the SDC track is renamed to `"ego"`.

## Output

ScenarioNet conversion produces:
- Scenario `.pkl` files
- `dataset_summary.pkl`
- `dataset_mapping.pkl`

The generated ScenarioNet files are aligned to the official Waymo ScenarioNet schema at the format level:
- top-level keys match Waymo samples
- metadata keys match Waymo samples
- track state array shapes and dtypes match Waymo samples
- lane features use Waymo-style `left_boundaries`, `right_boundaries`, `left_neighbor`, `right_neighbor`, `width`, `speed_limit_kmh`, and `speed_limit_mph`

Known remaining semantic differences from raw Waymo data:
- `dynamic_map_states` is empty unless the source dataset provides traffic light state
- lane boundary / neighbor intervals are simplified compared with Waymo's finer-grained chunked lane semantics

## License

### Code
The source code in this repository is licensed under the [Apache License 2.0](LICENSE).

### HetroD Dataset
The **HetroD dataset** is provided under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license for academic and non-commercial research purposes.

### Commercial Use
The Non-Commercial (NC) restriction applies to the general public. For commercial licensing inquiries or to discuss internal usage rights for project partners (e.g., fka GmbH), please contact: **[levelXData](https://levelxdata.com/)** 

## Citation

```bibtex
@inproceedings{hetrod,
  title={HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic},
  author={Yu-Hsiang Chen and Wei-Jer Chang and Christian Kotulla and Thomas Keutgens and Steffen Runde and Tobias Moers and Christoph Klas and Wei Zhan and Masayoshi Tomizuka and Yi-Ting Chen},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```
