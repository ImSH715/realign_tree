# Paper Revision Notes: MIL Shihuahuaco Method

## Current mismatch

The current paper still describes the main method as a feature-guided search pipeline with U-Net-derived feature polygons, prototype similarity, and a sliding-grid refinement step. That matches the earlier orange-tree / bounded-search work, but it no longer matches the active Shihuahuaco detection method in `train_mil_classifier.py`.

The current codebase instead uses a binary multiple-instance learning (MIL) classifier:

- labels are reduced to `Shihuahuaco` vs `Other` through `BinaryTree`;
- each positive point is treated as a noisy bag of candidate crops within a fixed spatial radius;
- negatives are kept centered by default;
- the model learns instance-level Shihuahuaco scores but is supervised only with bag-level binary labels;
- the selected instance with the highest logit becomes the model's proposed realigned tree position.

## Revised Core Claim

Replace the current claim:

> We align tree points by generating feature polygons and applying a sliding-grid algorithm.

with:

> We address noisy Shihuahuaco tree coordinates as a weakly supervised localization problem. Because a recorded positive point may be displaced from the actual crown, we train a multiple-instance binary classifier over candidate crops sampled around the recorded coordinate. The bag is positive if at least one crop within the search radius contains the target crown. At inference, the highest-scoring instance provides both the tree-level Shihuahuaco score and a candidate realigned coordinate.

## Methodology Replacement Draft

### Overview

The proposed method detects Shihuahuaco trees from high-resolution UAV orthomosaics under noisy point supervision. Field-recorded tree coordinates are not assumed to coincide exactly with crown centres in aerial imagery. Instead of training only on a crop centred at each recorded coordinate, the method treats each labelled point as a bag of spatial candidate crops. A positive bag corresponds to a recorded Shihuahuaco tree and is allowed to search within a bounded radius around the original point. A negative bag corresponds to a non-Shihuahuaco tree and is sampled at the recorded point by default. The model is trained with bag-level labels, while instance-level scores are used to identify the most likely crown location.

### Data Preparation

The ground-truth shapefile is first filtered to labelled tree points and split into training, validation, and test partitions using group-based splitting by orthomosaic file to reduce spatial leakage between nearby points. Species labels are converted into a binary target, where Shihuahuaco is assigned to the positive class and all other species are assigned to the negative class. Each point record stores the orthomosaic folder, source file, and coordinate fields used to resolve the corresponding TIFF tile and convert the coordinate into pixel space.

The dataset should be described as noisy point supervision rather than as clean crown-level ground truth. Visual inspection of the MIL false negatives shows several non-model failure modes, including blank or transparent orthomosaic regions, highly shaded crowns, ambiguous crown boundaries, and cases where the recorded point appears displaced from any visually obvious Shihuahuaco crown. These examples should be reported as a data-quality limitation and used to motivate weak supervision and realignment, not hidden behind a claim of precise labelled crowns.

### Encoder Initialization

The MIL classifier is built on top of a visual encoder initialized from the existing self-supervised and supervised training pipeline. The encoder can use the ResNet50, LeJEPA-style ViT, or DINOv2 backbone wrappers, but the current strongest runs use a DINOv2 encoder trained on seasonal shared orthomosaic imagery and then adapted through binary Shihuahuaco supervision. Image crops are resized to the model input size and normalized with ImageNet statistics. The current MIL experiments also include RGB-channel ablations such as green-channel mean replacement and white-branch boosting.

### Multiple-Instance Bag Construction

For each labelled point, a bag of candidate crops is constructed. Positive bags sample candidate centres within a 20 m radius of the recorded point. During training, candidate offsets are sampled randomly within this radius for ring-based MIL, or sampled from a fixed square grid for convolutional MIL. During validation and analysis, a deterministic set of offsets is used. The current convolutional MIL configuration uses a 5x5 grid, giving 25 candidate instances per bag.

Negative bags use a radius of 0 m by default, so they are represented by the centred crop. This design avoids incorrectly assuming that the full 20 m neighbourhood around a non-target tree is free of Shihuahuaco crowns. It also makes the localization task asymmetric in the intended way: positives are noisy and need search; negatives are used primarily to teach the classifier what non-target centred tree patches look like.

### MIL Objective

Each candidate crop is passed through the encoder and a linear binary classification head, producing an instance logit. Instance logits are pooled into a single bag logit using log-sum-exp pooling:

`bag_logit = tau * logsumexp(instance_logits / tau) - tau * log(n)`

This pooling behaves like a smooth approximation to max pooling and encodes the MIL assumption that a positive bag only needs one strongly positive instance. The model is trained with binary cross-entropy on the bag label, optionally using positive-class weighting and a balanced sampler to handle class imbalance.

### Realignment

At inference, the bag probability gives the Shihuahuaco detection score for the recorded point. The highest-logit instance in the bag is treated as the selected candidate. Its pixel offset from the recorded coordinate is converted back into metres, giving an estimated correction vector. The output CSV records both the original centre and the selected candidate position, including `best_dx_m`, `best_dy_m`, `best_px`, and `best_py` in the direct MIL predictions, or `dx_m`, `dy_m`, `px`, and `py` in the PCA analysis output.

## Experiments Section Draft

### Experiment 1: Centred-Crop Binary Baseline

Train a standard binary classifier on crops centred at the original tree coordinates. This baseline measures how well Shihuahuaco can be detected if the field coordinates are trusted directly. Validation is performed on the held-out split using the same centred-crop extraction. Report accuracy, macro-F1, Shihuahuaco precision, Shihuahuaco recall, Shihuahuaco F1, PR-AUC, and ROC-AUC.

Existing baseline logs show that centred-crop classifiers are highly sensitive to threshold choice and can collapse toward majority or all-positive predictions. For example, one RGB balanced ResNet50 run achieved PR-AUC 0.311 and ROC-AUC 0.547, with Shihuahuaco F1 improving only from 0.403 under the all-positive baseline to 0.414 after threshold tuning. Earlier phase-3 classifier logs report macro-F1 0.522, Shihuahuaco precision 0.300, recall 0.218, and F1 0.253 at the default decision threshold, improving to Shihuahuaco F1 0.422 after threshold tuning.

### Experiment 2: MIL Bag-Level Detection

Train the MIL classifier using positive bags sampled within 20 m and centred negative bags. Evaluate on the validation split using deterministic candidate offsets. Report bag-level classification metrics and ranking metrics from the selected bag probabilities.

The available MIL PCA summaries show a substantial improvement in separability. The green-mean run achieved selected PR-AUC 0.573 and ROC-AUC 0.861, with selected-instance PR-AUC 0.769 and ROC-AUC 0.933. The green-mean plus white-boost run improved this further to selected PR-AUC 0.711 and ROC-AUC 0.889, with selected-instance PR-AUC 0.908 and ROC-AUC 0.961.

At a 0.5 decision threshold, the green-mean plus white-boost best checkpoint produced 48 true positives, 124 true negatives, 39 false positives, and 7 false negatives on 218 validation bags. This corresponds to high Shihuahuaco recall and provides evidence that the MIL formulation is better aligned with the noisy-coordinate problem than centred-crop classification.

### Experiment 3: Original vs Realigned Training Points

This is the key ablation to make the paper convincing. Build two otherwise identical binary classifier training sets:

1. Original-centre training set: crop each point at the recorded coordinate.
2. MIL-realigned training set: for positive Shihuahuaco points, crop at the MIL-selected candidate coordinate; for negatives, keep the original centred crop.

Train the same classifier architecture, optimizer, image size, crop size, class weighting, sampler, and split protocol on both datasets. Evaluate both models on the same held-out validation/test protocol. The expected result is that the classifier trained from MIL-realigned positives should improve Shihuahuaco precision/recall balance, PR-AUC, ROC-AUC, and macro-F1 relative to the original-centre baseline.

To avoid leakage, the realignment model used to generate corrected training coordinates should be trained without using the held-out test labels. If realigned validation/test coordinates are also evaluated, that should be reported as an end-to-end MIL-assisted evaluation rather than a pure training-data-quality ablation.

### Experiment 4: Neighbourhood-Aware MIL

Train a variant of the MIL model where candidate crops are arranged on a square grid and the instance logits are passed through a small learnable 2D convolution before MIL pooling. This tests whether a candidate should be scored not only by its own crop, but also by whether neighbouring crops provide a coherent local Shihuahuaco signal. The direct comparison should keep the encoder, training split, image mode, radius, and optimizer fixed, and vary only the MIL head from independent instance scoring to convolutional neighbourhood scoring.

The intended configuration is a 5x5 candidate grid within the 20 m search radius (`bag_instances=25`, `bag_layout=grid`) with a 3x3 context convolution (`pooling=conv_lse`, `conv_kernel_size=3`). Report the same bag-level metrics as Experiment 2 and compare selected offsets to check whether neighbourhood context reduces isolated false positives without over-smoothing the selected crown position.

Important audit point: the convolutional instance score is a neighbourhood-context score, not necessarily a pure score for the crop centered at that exact grid cell. Visual inspection of large-offset true positives shows cases where the context-selected crop lies on bright dirt, blank imagery, or an edge artefact while the surrounding neighbourhood contains the evidence that drove the bag classification. For coordinate realignment, the selected location should therefore be based on the raw per-crop instance logit, with the context-selected grid cell reported separately as a diagnostic. The paper should distinguish bag-level conv classification from crown-centre realignment.

### Dataset Quality and Failure Analysis

Add a qualitative failure-analysis subsection based on the generated MIL debug contact sheets. The current false-positive sheet shows at least one bright yellow dirt path that is still amplified by the white-boost preprocessing, suggesting that the colour emphasis needs a stricter vegetation-context gate. The false-negative sheet shows that several missed positives are not straightforward missed crowns: one example is effectively black or transparent imagery, and others show heavy shade, ambiguous crown structure, or large coordinate displacement. The large-offset true-positive sheet also reveals shortcut-like selections in which the convolutional context score selects a crop on dirt or blank image edge, even though the bag is correctly classified as positive.

The paper should therefore avoid describing the validation labels as exact crown annotations. A more defensible framing is that the labels are field-recorded tree points with uncertain crown correspondence in the orthomosaic. Report the debug-sheet review as evidence that the task contains label noise, imagery gaps, and uncertain positives. This strengthens the motivation for MIL, but it also means the strongest validation scores should be presented with caution and ideally confirmed with a held-out test split and manual visual audit of selected corrections.

## Recommended Paper Structure

1. Keep the motivation: GPS and stem/crown offsets create noisy point labels.
2. Shorten the orange-tree sliding-grid material into preliminary work or remove it from the main method.
3. Replace Section 4 with the MIL methodology above.
4. Replace the unfinished Section 5 with the three experiments above.
5. Move prototype search and U-Net polygons to prior exploratory work unless those results are still included.
6. Update the abstract so it names weak supervision, multiple-instance learning, binary Shihuahuaco detection, and coordinate realignment from selected candidate crops.
