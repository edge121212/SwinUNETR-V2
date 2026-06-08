# Task05 Prostate — quantitative test of the two Attention U-Net premises

n = 32 training cases. Spacing 0.6x0.6x4.0 mm; T2 = channel 0, ADC = channel 1.


## Premise 1 — shape/size variation ('sometimes big, sometimes small')

| class | volume mean±std (mm³) | volume CV | volume range (min–max) | max/min (present cases) | cases absent | mean fg-fraction | intra-volume slice-area CV |
|---|---|---|---|---|---|---|---|
| PZ | 18,771 ± 16,389 | 0.87 | 6,248 – 101,624 | 16.3× | 0 | 0.70% | 0.47 |
| TZ | 55,549 ± 57,385 | 1.03 | 7,950 – 303,782 | 38.2× | 2 | 2.05% | 0.48 |
| Prostate | 74,320 ± 55,057 | 0.74 | 16,030 – 315,404 | 19.7× | 0 | 2.75% | 0.49 |

*2 cases (prostate_18, prostate_32) contain **no TZ at all** — the most extreme form of size variation.*

## Premise 2 — low contrast vs surrounding tissue (5 mm ring)

CNR = |μ_organ−μ_ring| / pooled-σ (higher = easier).  OVL = intensity-histogram overlap in [0,1] (higher = more confusable).

| comparison | modality | CNR mean±std | Michelson mean±std | OVL mean±std |
|---|---|---|---|---|
| prostate vs surrounding | T2 | 0.27 ± 0.22 | 0.063 ± 0.050 | 0.65 ± 0.08 |
| prostate vs surrounding | ADC | 0.71 ± 0.40 | 0.164 ± 0.108 | 0.58 ± 0.11 |
| PZ vs TZ (internal boundary) | T2 | 0.65 ± 0.38 | — | 0.69 ± 0.14 |

## Reference for interpretation

- A coefficient of variation (CV) ≳ 0.3 is conventionally 'high' variability; tumor/organ-size CVs in the literature are typically 0.2–0.5.
- OVL ≳ 0.7 means the organ and its surroundings share most of their intensity range — i.e. they are hard to separate by intensity alone, exactly the situation attention gates are meant to help with.
- The Attention U-Net paper does not publish these exact numbers for pancreas; we report prostate values so the claim rests on measurement, not analogy.

