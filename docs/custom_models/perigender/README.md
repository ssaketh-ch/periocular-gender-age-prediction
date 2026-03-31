# PeriGender Family

This folder documents the two custom gender models used in the refreshed project:

- `PeriGender` (v1)
- `PeriGenderV2` (v2)

The source code lives in:

- `models/perigender.py`

## Context

The original custom architecture was inspired by:

- Hussain, M., Alrabiah, R., & AboAlSamh, H. A. (2023). *Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features*. Intelligent Automation & Soft Computing, 35(3), 2941-2962.

The UBIPr dataset used for gender evaluation originates from:

- Chandrashekhar Padole, Hugo Proenca. *Periocular Recognition: Analysis of Performance Degradation Factors*. Proceedings of the Fifth IAPR/IEEE International Conference on Biometrics (ICB 2012), New Delhi, India, March 30-April 1, 2012.

## What Problem The Model Solves

The input is a periocular image: the region around the eye rather than the whole face.
The output is a binary prediction:

- `female`
- `male`

This is a simpler task than age prediction because:

- it is binary rather than 10-way classification
- the UBIPr data is comparatively controlled
- the periocular region retains strong textural cues for gender

## PeriGender v1

### High-Level Idea

`PeriGender` tries to imitate the paper's multiscale feature reuse idea.
Instead of relying only on the deepest feature map, it collects intermediate features from several depths and concatenates them before classification.

### Backbone Layout

1. Initial convolution and max-pooling reduce spatial size and extract early edge/texture features.
2. A series of custom residual blocks increases channel depth.
3. Four skip branches capture features at different stages.
4. All skip outputs are concatenated with the final backbone tensor.
5. Global average pooling and a dropout + linear head produce the final gender logits.

### Why That Matters

In simple terms:

- shallow layers capture local details like eyelid edges and fine texture
- deeper layers capture larger patterns like eyebrow shape or overall periocular structure
- concatenating them gives the classifier more than one "view" of the image

### Technical Weaknesses of v1

- the classifier head is very shallow
- skip outputs are concatenated directly without a learnable fusion bottleneck
- the architecture is trained from scratch, while the ResNet baselines are pretrained
- the final head may not be expressive enough to exploit the multiscale tensor fully

## PeriGender v2

### What Changed

`PeriGenderV2` keeps the same multiscale backbone idea, but improves the fusion and classification stage.

Main changes:

1. The concatenated 524-channel multiscale tensor is passed through a learnable `1x1` fusion layer.
2. Batch normalization and ReLU are applied after fusion.
3. The classifier head is expanded to a 2-layer MLP with dropout.

### Why This Is Better

The `1x1` fusion layer acts like a learnable mixing stage:

- it can decide which skip features matter most
- it reduces noise from naive concatenation
- it creates a cleaner shared representation before the final classifier

The deeper head also gives the model more capacity to separate male/female classes after feature extraction.

### Simple Intuition

`PeriGender` v1 says:
"Here are all my features at once. Make a decision."

`PeriGenderV2` says:
"First let me combine the multiscale features in a smarter way, then I’ll classify."

## Relationship to the Paper

The paper's original idea is multiscale deep feature reuse in the periocular region.
Both v1 and v2 preserve that core concept:

- multiple skip pathways
- feature reuse across scales
- periocular-only input

v2 is not a strict reproduction.
It is an engineering improvement built on the same design principle.

## Practical Takeaway

In this refreshed repo:

- ResNet baselines still perform best on UBIPr gender
- `PeriGenderV2` is the strongest custom gender architecture
- the custom models are still valuable because they are periocular-specific and more faithful to the original research idea than a generic ImageNet backbone

