# Results and Figures

This file now summarizes the refreshed experiment set rather than the older notebook-era tables.

## Current Best Results

### Gender on UBIPr

| Model | Best Image Accuracy | Notes |
|---|---:|---|
| `ResNet18` | `0.9810` | Best overall gender result |
| `ResNet34` | `0.9774` | Strong reference baseline |
| `ResNet50` | `0.9693` | Larger backbone did not improve |
| `PeriGenderV2` | `0.9711` | Best custom gender architecture |

### Age on Strict Periocular UTKFace

| Model | Best Test Accuracy | Notes |
|---|---:|---|
| `PeriAgeResNet34` | `0.8668` | Best overall age result |
| `ResNet34` | `0.8508` | Best plain baseline |
| `ResNet18` | `0.8407` | Strong compact baseline |
| `ResNet50` | `0.8158` | Bigger was not better |
| `PeriAgeV2` | `0.7983` | Best scratch-style custom age model |

## Why These Numbers Matter

The refreshed repo is stricter and more faithful to the project goal than the original workflow:

- age now uses periocular-only crops rather than full-face images
- gender now uses subject-level splitting on UBIPr
- runs are stored with timestamped metrics and checkpoints

That makes the current numbers much more reproducible and easier to trust.

## Generated Figures

The report figure generator writes plots under:

- `results/figures/gender/`
- `results/figures/age/`

Use:

```bash
python scripts/generate_report_artifacts.py
```

### Age

![Age best accuracy](./figures/age/age_best_accuracy.png)

![Age accuracy curves](./figures/age/age_accuracy_curves.png)

### Gender

![Gender best accuracy](./figures/gender/gender_best_accuracy.png)

![Gender accuracy curves](./figures/gender/gender_accuracy_curves.png)

![Gender confusion: ResNet18](./figures/gender/resnet18_confusion.png)

![Gender confusion: ResNet34](./figures/gender/resnet34_confusion.png)

![Gender confusion: PeriGenderV2](./figures/gender/perigender_v2_confusion.png)

## Canonical Runs Used For Repo Reporting

### Gender

- baseline: `runs/gender/ubipr/baselines/resnet18_e30/20260330_212202`
- baseline reference: `runs/gender/ubipr/baselines/resnet34_e30/20260330_214519`
- custom: `runs/gender/ubipr/custom/perigender_v2_adamw/20260331_004845`

### Age

- baseline: `runs/age/periocular/baselines/resnet34/20260330_151158`
- custom v2: `runs/age/periocular/custom/periage_v2_bs32/20260330_174946`
- hybrid best: `runs/age/periocular/hybrid/periage_resnet34_ft/20260330_202332`
