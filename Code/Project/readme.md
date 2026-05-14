# Shihuahuaco MIL Experiment Data

This README records the current understanding of the data used for the
Shihuahuaco multiple-instance learning (MIL) experiments. It is intended to
make clear which data are used for supervised training, which data are used for
external/manual validation, and which older labels should no longer be treated
as reliable Shihuahuaco labels.

## Core Task

The active experiment is a binary Shihuahuaco detection and coordinate
realignment task over UAV orthomosaic imagery. Each labelled point is converted
into a bag of candidate image patches sampled around the point. The bag label is
binary:

- `1`: Shihuahuaco
- `0`: other tree species

For positive bags, the model is allowed to search a local neighbourhood because
field or census coordinates may be displaced from the visible crown. For
negative bags, the default experiments keep the search radius at zero metres so
that nearby target crowns are not accidentally treated as negative evidence.

## Curated Crown Dataset

The currently trusted supervised source is the curated 2023 crown shapefile:

```text
/mnt/parscratch/users/aca21jo/curated/copas-2023/copas-2023/copas_2023_condatos_vs2.shp
```

This file contains manually or department-curated crown geometries and species
metadata. It has:

- 1,483 total crown rows
- 1,478 unique rounded coordinate keys from the attribute coordinate fields
- 229 rows where `NOMBRE_COM == Shihuahuaco`
- 228 unique rounded Shihuahuaco attribute-coordinate keys

Important fields include:

- `NOMBRE_COM`: common species name used for binary Shihuahuaco relabelling
- `NOMBRE_CIE`: scientific name
- `ZONA_UTM`, `COORDENADA`, `COORDENA_1`: original census-style coordinate
  attributes
- `geometry`: curated crown polygon geometry

For the training experiments, the trusted location is the curated crown
geometry/centroid, not necessarily the old coordinate attributes. Visual checks
showed that the curated geometry centroids often land much better on crowns than
the older census-style coordinate fields.

## Corrected Experiment Splits

The current supervised MIL experiments use the corrected split directory:

```text
./outputs/splits_binary_curated
```

This directory was created from the previous image-backed split files in
`./outputs/splits_binary`, but the species labels were rewritten by nearest
joining each point to the curated crown centroids. The relabelling script was:

```text
relabel_splits_from_curated.py
```

The key relabelling settings were:

- curated file: `copas_2023_condatos_vs2.shp`
- target CRS: `EPSG:32718`
- curated species field: `NOMBRE_COM`
- target label: `Shihuahuaco`
- maximum nearest-centroid distance: `5 m`
- output directory: `./outputs/splits_binary_curated`

The split membership itself was preserved from the previous valid-point split
files. The relabel step changes `Tree` and `BinaryTree`, but does not reshuffle
rows between train, validation, and test.

The corrected image-backed dataset contains 1,126 rows:

| Split | Total rows | Shihuahuaco positives | Other negatives |
| --- | ---: | ---: | ---: |
| train | 784 | 75 | 709 |
| val | 218 | 27 | 191 |
| test | 124 | 23 | 101 |
| total | 1,126 | 125 | 1,001 |

These 125 positives are the valid Shihuahuaco points currently used by the
image-backed supervised experiments. The curated shapefile contains more
Shihuahuaco crowns overall, but only these 125 are present in the current
resolved train/validation/test pipeline subset with the required imagery fields
and split membership.

## Why the Old Binary Splits Are Not Trusted

The older directory:

```text
./outputs/splits_binary
```

should not be used as the final Shihuahuaco supervision source. Before
curated relabelling it contained:

- 141 old positives
- 985 old negatives

Comparison against the curated crown file showed that many old positive labels
were actually other species in the curated dataset. Among the old positives,
only a small subset matched curated `NOMBRE_COM == Shihuahuaco`. Conversely,
many curated Shihuahuaco rows had been labelled as negative. The older split is
therefore useful as a starting point because it contains image paths, coordinate
fields, and split membership, but not as a reliable binary label source.

## Full Census / Large Dataset

The larger forest census table is:

```text
/mnt/parscratch/users/aca21jo/curated/censo_forestal_datos.csv
```

It came from the Google Drive spreadsheet:

```text
Censo Forestal.xlsx - Datos.csv
```

This table is much larger than the curated crown subset:

- 17,972 total rows
- 17,962 unique rounded coordinate keys
- 20 duplicate-coordinate rows
- 2,078 Shihuahuaco rows
- 2,077 unique Shihuahuaco coordinate keys

The main fields used for comparison are:

- `NOMBRE_COMUN`: common species name
- `ZONA_UTM`: UTM zone
- `COORDENADA_ESTE`: easting
- `COORDENADA_NORTE`: northing

Top species counts in this table include:

| Species | Rows |
| --- | ---: |
| Catahua | 2,867 |
| Copaiba | 2,780 |
| Shihuahuaco | 2,078 |
| Capirona | 1,551 |
| Quina quina | 1,342 |
| Quinilla | 996 |
| Lupuna | 918 |
| Manchinga | 901 |
| Yacushapana | 757 |
| Mashonaste | 709 |

The full census is not currently used as supervised imagery. It is better
treated as a large external candidate pool for later manual checking or
generalisation experiments. The reason is that exact coordinate matching between
the full census coordinate fields and the curated crown geometries is weak:

- 197 curated rows matched a full-census coordinate key
- 197 of those agreeing matches had the same species
- 36 curated Shihuahuaco coordinate keys matched full-census Shihuahuaco keys
- most full-census Shihuahuaco coordinates are not close to a curated
  Shihuahuaco crown centroid

Spatially, for all 2,078 full-census Shihuahuaco rows compared to nearest
curated Shihuahuaco crown centroids:

- median distance was about 1,213 m
- mean distance was about 1,723 m
- 13 were within 5 m
- 27 were within 10 m
- 35 were within 20 m
- 38 were within 30 m
- 69 were within 100 m
- 26 fell inside a curated Shihuahuaco polygon

This does not mean the census is useless. It means the census coordinates and
the curated crown geometry layer should not be assumed to be interchangeable.
For the current project, the full census is best used after training: sample a
manageable set of candidate points, run the detector, generate debug patches,
and manually score whether the detector generalises.

## Imagery and Patch Extraction

The model uses high-resolution UAV orthomosaic TIFFs. The pipeline resolves each
row through fields such as:

- `Folder`
- `File`
- `fx`
- `fy`

The MIL scripts build a TIFF index before training. In the current Stanage runs,
the index contained:

- 22 TIFF folders
- 310 TIFF files

The main MIL patch setting is:

- patch size: `160 px`
- bag layout: `grid`
- bag instances: `25`
- grid shape: `5 x 5`
- positive search radius: `20 m`
- negative search radius: `0 m`
- primary image mode: `rgb_white_boost`
- comparison image mode: `rgb_green_mean_white_boost`

The 25-candidate grid means each positive bag contains candidate crops over a
local neighbourhood around the supplied point. In `conv_lse` runs, instance
logits are arranged back into this `5 x 5` grid and a small convolution is
applied over neighbouring logits before MIL pooling. This lets the model combine
the confidence of a candidate patch with the confidence of nearby candidates,
similar in spirit to a CNN kernel, but operating over candidate confidence
scores rather than raw image pixels.

## Encoder Checkpoints

The corrected curated runs should initialise from SSL-only encoders, not from
old binary-adapted checkpoints trained on the noisy/mislabelled split. The
current SSL checkpoints are:

```text
./outputs/phase1_dino_ssl_shared_seasonal/phase1_encoder_best.pth
./outputs/phase1_dino3_ssl_shared_seasonal/phase1_encoder_best.pth
./outputs/phase1_lejepa_ssl_large_gpu/phase1_encoder_best.pth
```

The contaminated older binary checkpoints should be avoided for final curated
experiments because they were trained with incorrect Shihuahuaco labels.

## Current Experiment Root

The corrected MIL factorial runs are stored under:

```text
/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_curated_factorial
```

The active comparison includes:

- DINOv2, DINOv3, and LeJEPA encoders
- `lse` MIL pooling
- `conv_lse` neighbourhood-logit pooling
- `rgb_white_boost` and `rgb_green_mean_white_boost`
- `patch160`
- `bag25`

Early audit results show that the corrected labels produce much cleaner
separation than the old labels. However, validation performance should still be
interpreted carefully because the validation set contains only 27 positives.
The strongest claims should be based on:

- held-out test split performance
- visual inspection of selected candidate patches
- synthetic-shift recovery experiments where the true curated crown centre is
  known
- manual checking on a sampled subset of the larger census dataset

## Weak Census Realignment Subset

For external generalisation, the full census can be sampled as weakly labelled
Shihuahuaco candidates after excluding records that appear to be in the curated
training set. This is not used as clean supervised training data. Instead, the
trained MIL models are applied to these weak points to produce candidate
realigned positions. Those realigned positions can then be manually checked or
used to build model-specific classifier datasets.

The helper script is:

```text
prepare_weak_shihuahuaco_subset.py
```

It filters the full census to `NOMBRE_COMUN == Shihuahuaco`, excludes points
that match or lie close to curated Shihuahuaco crowns, assigns each remaining
point to a covering orthomosaic TIFF, and writes a MIL-compatible point layer
with `Folder`, `File`, `fx`, `fy`, `Tree`, and `BinaryTree`.

The model-application script is:

```text
apply_mil_realign.py
```

It loads a trained MIL run directory, scores each weak census point as a
positive bag, and writes the selected realigned coordinate. It records both raw
instance selection and context-logit selection, but the default realignment
coordinate uses the raw instance selection because this is the less
context-smoothed coordinate estimate.

The Slurm wrapper is:

```text
experiments/mil_shihuaco_factorial/submit_weak_censo_realign.sh
```

By default this queues on `gpu-h100-nvl`, prepares a 500-point weak census
subset, applies every complete model in
`/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_curated_factorial`,
and writes:

```text
/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_curated_factorial/weak_censo_500_realign/weak_shihuahuaco_500_not_curated.gpkg
/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_curated_factorial/weak_censo_500_realign/realigned_by_model/<run_name>/realigned_points.csv
/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_curated_factorial/weak_censo_500_realign/realigned_points_all_models.csv
```

## Recommended Interpretation

Use the curated relabelled split as the supervised experiment dataset. Describe
it as an image-backed curated crown subset with 125 valid Shihuahuaco positives,
not as the full Shihuahuaco population.

Use the full census as a large external source of candidate records and species
metadata, not as clean supervised imagery.

Avoid reporting results from the old `outputs/splits_binary` labels as
Shihuahuaco performance unless they are explicitly framed as pre-correction or
label-noise diagnostics.
