# LadderSym

**Music Error Detection with Iterative Inter-Stream Alignment and Symbolic Score Prompting**

LadderSym is a transformer-based system for end-to-end music performance error detection.

## Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/ben2002chou/LadderSYM.git
cd LadderSYM
conda create -n laddersym python=3.11 -y
conda activate laddersym
pip install -r requirements.txt
wandb login
```

Download datasets and official checkpoints (recommended):

```bash
python setup_laddersym_assets.py --datasets all --checkpoints all
source .env.laddersym
```

Manual setup is also supported (see `Data and Checkpoints` below).

## Training

```bash
python train_laddersym.py \
  --config-name=config_maestro \
  model=laddersym_MT3Net \
  dataset=MAESTRO \
  split_frame_length=2000
```

```bash
python train_laddersym.py \
  --config-name=config_coco \
  model=laddersym_MT3Net \
  dataset=CocoChorales \
  split_frame_length=2000
```

## Evaluation

MAESTRO-E:

```bash
python test_laddersym.py \
  --config-dir=config \
  --config-name=config_maestro \
  model=laddersym_MT3Net \
  path=/path/to/model.ckpt \
  eval.eval_dataset=MAESTRO \
  eval.exp_tag_name=laddersym_maestro \
  hydra/job_logging=disabled \
  eval.contiguous_inference=True \
  split_frame_length=2000
```

CocoChorales-E:

```bash
python test_laddersym_coco.py \
  --config-dir=config \
  --config-name=config_coco \
  model=laddersym_MT3Net \
  path=/path/to/model.ckpt \
  eval.eval_dataset=CocoChorales \
  eval.exp_tag_name=laddersym_coco \
  hydra/job_logging=disabled \
  eval.contiguous_inference=True \
  split_frame_length=2000
```

## Inference

Batch inference over piece folders (recommended):

```bash
python laddersym_test_inference.py \
  --config-dir=config \
  --config-name=config_maestro_prompted \
  model=laddersym_MT3Net \
  path=pretrained/laddersym/checkpoints/maestro/prompted/model.ckpt \
  dataset_dir=$LADDERSYM_MAESTRO_ROOT \
  output_mid_name=laddersym_output.mid \
  overwrite=true \
  hydra/job_logging=disabled
```

Single-piece inference with explicit file paths:

```bash
python laddersym_test_inference.py \
  --config-dir=config \
  --config-name=config_maestro_prompted \
  model=laddersym_MT3Net \
  path=pretrained/laddersym/checkpoints/maestro/prompted/model.ckpt \
  mistake_file=$LADDERSYM_MAESTRO_ROOT/<piece_id>/mistake.wav \
  score_file=$LADDERSYM_MAESTRO_ROOT/<piece_id>/score.wav \
  prompt_file=$LADDERSYM_MAESTRO_ROOT/<piece_id>/score.mid \
  output_dir=$LADDERSYM_MAESTRO_ROOT/<piece_id> \
  output_mid_name=laddersym_output.mid \
  hydra/job_logging=disabled
```

## Interpreting Outputs

The output MIDI contains three semantic tracks:

- Track 1: Extra notes
- Track 2: Missing notes
- Track 3: Correct notes
- Example output path: `<piece_dir>/laddersym_output.mid`

## Data and Checkpoints

- Datasets: [CocoChorales-E](https://huggingface.co/datasets/ben2002chou/CocoChorales-E), [MAESTRO-E](https://huggingface.co/datasets/ben2002chou/MAESTRO-E)
- Checkpoints repo: [ben2002chou/laddersym-checkpoints](https://huggingface.co/ben2002chou/laddersym-checkpoints)
- Viewer-friendly previews: [CocoChorales-E-preview](https://huggingface.co/datasets/ben2002chou/CocoChorales-E-preview), [MAESTRO-E-preview](https://huggingface.co/datasets/ben2002chou/MAESTRO-E-preview)

Recommended setup:

```bash
python setup_laddersym_assets.py --datasets all --checkpoints all
source .env.laddersym
```

Checkpoint variants:

| Dataset | Prompted | Unprompted |
| --- | --- | --- |
| CocoChorales-E | [model.ckpt](https://huggingface.co/ben2002chou/laddersym-checkpoints/blob/main/checkpoints/cocochorales/prompted/model.ckpt) | [model.ckpt](https://huggingface.co/ben2002chou/laddersym-checkpoints/blob/main/checkpoints/cocochorales/unprompted/model.ckpt) |
| MAESTRO-E | [model.ckpt](https://huggingface.co/ben2002chou/laddersym-checkpoints/blob/main/checkpoints/maestro/prompted/model.ckpt) | [model.ckpt](https://huggingface.co/ben2002chou/laddersym-checkpoints/blob/main/checkpoints/maestro/unprompted/model.ckpt) |

Advanced setup options:

```bash
# only datasets
python setup_laddersym_assets.py --datasets all --checkpoints none

# only checkpoints
python setup_laddersym_assets.py --datasets none --checkpoints all
```

## Citation

```bibtex
@inproceedings{
chou2026laddersym,
title={LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection},
author={Benjamin Shiue-Hal Chou and Purvish Jajal and Nicholas John Eliopoulos and James C. Davis and George K Thiruvathukal and Kristen Yeon-Ji Yun and Yung-Hsiang Lu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=cizuvfyQXs}
}
```
