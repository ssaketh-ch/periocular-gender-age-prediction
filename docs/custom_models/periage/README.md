# PeriAge Family

This folder documents the custom age models used in the refreshed project:

- `PeriAge` (v1)
- `PeriAgeV2` (v2)
- `PeriAgeResNet34` (hybrid)

The source code lives in:

- `models/periage.py`

## Context

These models were built to predict age bucket from periocular imagery rather than the full face.
That makes the task much harder because the eye region contains less age information than the entire face.

The design lineage comes from the same original periocular multiscale feature paper used for the gender model:

- Hussain, M., Alrabiah, R., & AboAlSamh, H. A. (2023). *Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features*. Intelligent Automation & Soft Computing, 35(3), 2941-2962.

The periocular biometric context also relates to:

- Chandrashekhar Padole, Hugo Proenca. *Periocular Recognition: Analysis of Performance Degradation Factors*. Proceedings of the Fifth IAPR/IEEE International Conference on Biometrics (ICB 2012), New Delhi, India, March 30-April 1, 2012.

## Why Age Is Harder Than Gender

Gender prediction here is binary.
Age prediction is a 10-class decade-bucket problem.

For age, the model must distinguish subtle differences such as:

- skin texture
- wrinkle density
- eyelid looseness
- eyebrow aging patterns

Those signals are weaker and noisier in periocular crops than in whole-face images.

## PeriAge v1

### Design

`PeriAge` starts from the `PeriGender` backbone and adapts it for square `224x224` inputs.

Key additions:

1. `SkipConnection4` upsamples to match the required spatial shape.
2. The main feature path also uses an upsample block before concatenation.
3. The final head predicts 10 age buckets.

### What It Is Trying To Do

The model reuses the same multiscale concept:

- early layers capture small periocular texture changes
- deeper layers capture broader structural cues
- all scales are fused to classify age bucket

### Limitation

The final head is still shallow, and the multiscale concatenation is mostly "raw."
That makes it harder to compete with pretrained ResNet baselines.

## PeriAge v2

### Design Upgrade

`PeriAgeV2` strengthens the part after multiscale concatenation.

Changes:

1. The 524-channel concatenated tensor is passed through a learnable `1x1` fusion block.
2. Global pooling is followed by a deeper MLP head.
3. Dropout is used more deliberately in the classifier.

### Why It Helps

Age prediction needs a more expressive classifier than gender prediction because:

- there are more classes
- class boundaries are fuzzier
- neighboring age buckets are visually similar

The added fusion block helps the model learn which multiscale features deserve emphasis before classification.

## PeriAgeResNet34

### Why This Hybrid Exists

The custom models were competitive, but they lagged behind pretrained ResNet baselines.
To compete more directly, the repo now includes a hybrid model:

- pretrained ResNet34 backbone
- multiscale feature taps from intermediate residual stages
- custom fusion head for age prediction

### Architecture Logic

1. Use the pretrained ResNet stem and layers as a strong feature extractor.
2. Tap features from `layer2`, `layer3`, and `layer4`.
3. Project each tap to a common lower-dimensional space with `1x1` convolutions.
4. Pool them and concatenate them.
5. Feed the fused vector into a dropout + MLP classifier head.

### Why It Works Better

This approach combines:

- pretrained representation power from ResNet
- multiscale periocular-specific fusion from the custom family

In practice, this gave the strongest age result in the refreshed repo.

## Relationship to the Original Paper

The paper focused on multiscale periocular feature reuse.
The `PeriAge` family keeps that idea, but adapts it to a harder 10-class age problem.

The hybrid `PeriAgeResNet34` is not a pure reproduction of the paper.
It is a research-driven extension:

- keep multiscale periocular fusion
- add pretrained deep visual priors
- fine-tune carefully so the pretrained features are not destroyed early

## Practical Takeaway

In the refreshed experiments:

- scratch-style custom age models improved, but stayed below pretrained ResNets
- `PeriAgeV2` improved the custom family substantially
- `PeriAgeResNet34` was the first model to beat the plain ResNet34 age baseline in periocular-only evaluation

