# Custom Model Families

This folder documents the periocular-specific model families that grew out of the original project.

## Why These Models Exist

The original project was not meant to be just a ResNet fine-tuning exercise. The custom models try to preserve the central research idea from the source paper:

- use the periocular region only
- reuse features from multiple depths
- fuse shallow and deep information rather than relying only on the deepest feature map

The paper that motivated the original architecture direction is:

- Hussain, M., Alrabiah, R., & AboAlSamh, H. A. (2023). *Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features*. Intelligent Automation & Soft Computing, 35(3), 2941-2962.

The gender dataset context in this repo comes from:

- Chandrashekhar Padole, Hugo Proenca. *Periocular Recognition: Analysis of Performance Degradation Factors*. Proceedings of the Fifth IAPR/IEEE International Conference on Biometrics, ICB 2012, New Delhi, India, March 30-April 1, 2012.

## Families

- [PeriGender Family](./perigender/README.md)
- [PeriAge Family](./periage/README.md)

## Versioning Logic

Across both families, the naming follows the same rough pattern:

- `v1`
  The original scratch-style multiscale custom architecture.
- `v2`
  A stronger scratch-style version with better feature fusion and a more expressive classifier head.
- `hybrid`
  A periocular-specific fusion design built on top of a pretrained backbone.

## In Simple Terms

The progression of the repo looks like this:

1. Start from the original research idea.
2. Build a direct custom implementation.
3. Improve the fusion and head while keeping the same multiscale spirit.
4. For the hardest task, age, combine the periocular fusion idea with a pretrained ResNet backbone.

That progression is why the repo now contains both historically faithful custom models and more performance-driven hybrid variants.
