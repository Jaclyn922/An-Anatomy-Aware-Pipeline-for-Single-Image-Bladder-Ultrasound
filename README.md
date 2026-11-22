# Anatomy-aware denoising for single-image bladder ultrasound
Lumen-only denoising, device protection and image-quality grading

Question: when only **one frame** is available (for example in drug-delivery device research in the bladder), can we
(1) find the anatomy automatically, (2) remove noise where it is safe to do so, (3) keep small bright structures
(a device, a lesion) from being erased or invented, and (4) say whether the frame is good enough to use?

Everything here is computational. No new patient data were collected. Author: Jacklyn Xu (M.S., Biomedical Engineering, Duke University), sole author.

## Pipeline in one picture
```
frame --> U-Net finds the lumen --> three zones (lumen core / wall band / outer tissue)
      --> noise level (robust MAD of the high-pass image inside the imaging fan)
      --> detector marks a possible device (raw image) --> device walled off from the filter
      --> non-local means (or SRAD) inside the lumen core only; wall and outer tissue left as acquired
      --> quality grade (1-5)
```
Why only the lumen: it is fluid (anechoic), so texture inside it carries no anatomy and can be smoothed hard. Outside it
(wall, tissue) texture is real, so it is left untouched. Note on terms: in tissue the grain is speckle; inside the lumen it is
mostly system noise that looks speckle-like. The method relies on the lumen being uniform, not on speckle statistics.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # numpy scipy scikit-image opencv-python pillow matplotlib PyWavelets pandas scikit-learn pytest
pip install torch torchvision       # U-Net and the grading experiments (tested with torch 2.15, torchvision 0.29)
```
Run every command from the project root.

## Data
| Set | What | Where |
|---|---|---|
| Bladder ultrasound set (public) | 233 photographs with hand-drawn lumen polygons (Roboflow Universe, "BladderDetection", user `testyolo`: https://universe.roboflow.com/testyolo-9g95p/bladderdetection). This project uses its 200-image training split. The licence recorded in `src/make_intro_fig.py` is CC BY 4.0; check the dataset page before redistributing. | `data/roboflow/` (images + `_annotations.coco.json`, no extra `train/` folder) |
| Bladder case images (UltrasoundCases.info, https://www.ultrasoundcases.info/search/?q=bladder) | Case images with lesions, scored 1-5 for quality by a senior specialist; I graded 14 of them myself. **Copyrighted: not committed, local only.** | `data/pathology_local/`; the lesion and grading scripts currently read them from `/tmp/u/bladder pic/` (edit the path in `lesion25_masks.py`) |
| My 14 graded images | grade labels in `LABELS` inside `src/grade_cls.py`; prepared arrays in `grade/prep14.npz` | `grade/` |

`data/manifest.csv` has `frame` and `video` columns. **Always split by `video` (or at least `frame`), never by file:**
the 200 files are augmented copies of 67 source frames, and 53 of those frames come from one video. All intervals in this project
are bootstrap resamples over source frames.

## Step 1 - data and masks
```
python src/prepare_data.py --root data/roboflow --out data/masks --raw_out data/masks_raw
python src/compare_masks.py      # outputs/mask_cleaning.png
python src/make_previews.py      # outputs/mask_check.png, outputs/device_preview.png
```
Masks are the dataset authors' hand-drawn polygons, rasterized with PIL and cleaned (fill holes, keep the largest component,
Gaussian blur sigma 1.5 and threshold 0.5). Raw masks are kept in `data/masks_raw`. Masks cover the dark lumen somewhat tightly.

## Step 2 - the three-region baseline filter and SRAD
```
python tests/test_srad.py
```
`src/roi.py`: corrected SRAD (Yu and Acton, with an optional barrier that stops diffusion across region labels), the zones
(`build_rois`: lumen core = mask eroded 3 px, wall band = 3 px either side, outer tissue) and the **baseline three-region filter**
(SRAD in the lumen, bilateral on the wall band, light Gaussian outside). In the code and CSVs it appears as `roi30` / "ROI v1" (no barrier)
and `roi_v2_100` / "ROI v2" (barrier at the lumen core, 100 iterations).

## Step 3 - lumen-only non-local means (the proposed filter)
`src/roi_nlm.py`: non-local means (skimage, patch 5, search distance 6, `fast_mode`) with strength h = 2 x sigma_n, where sigma_n is the
robust MAD of the 3-px high-pass image inside the imaging fan. The result is blended in only inside the lumen core (mask eroded 3 px)
with a 1.5 px Gaussian feather; wall and outer tissue are returned exactly as acquired. h = 2 was chosen on development fold 0
(maximum lumen-spread removal with edge kept >= 0.95); it is not tuned on the test folds.

## Step 4 - device test object and simulator (ground truth for devices)
```
python tests/test_simulate.py
python src/calibrate.py --looks 2     # fits simulator parameters to the real frames (about 3 minutes)
python src/make_sim_figure.py         # outputs/sim_vs_real.png
```
`src/device.py`: a thin 8-shaped tube (Gerono lemniscate centre-line, random pose and scale; length = u x lumen equivalent diameter,
radius = rho x lumen equivalent diameter), inspired by the public TAR-210 schematic. Size, material and echo strength are assumptions, so call it "device-like".
The imaging plane cuts the tube: a pixel is device if its distance to the centre-line is below the radius (slice, not projection).
`src/simulate.py`: reflectivity map from a lumen mask (smooth wall from the signed distance, textured tissue, attenuation with TGC, optional device
with shadow and reverberation), complex-Gaussian noise blurred by the PSF, `looks` averaged patterns, log compression, black level.
`calibrate.py` fits gain, dynamic range, wall and tissue level, PSF size on the Apache video and checks the other 8 sources.

Known limits of the simulator: tissue is speckle plus smooth texture only; the wall is a uniform ring; levels were fitted on one video
(other sources have different gain, so later experiments randomise gain and dynamic range); device shape, size and echo strength are assumptions.

## Step 5 - does denoising hurt device visibility? (simulation)
```
python tests/test_metrics.py          # metrics must catch a deliberate eraser and a deliberate hallucinator
python src/run_experiment.py --n 600  # about 9 minutes on 2 cores -> outputs/results.csv
python src/summarize.py               # outputs/summary.csv, sweep.csv, results_figure.png
python src/make_example_figure.py     # outputs/example_conditions.png
```
`src/degrade.py` samples random acquisition conditions and renders a PRESENT and an ABSENT image per scene. `src/metrics.py`: contrast retention,
2AFC detection with a difference-of-Gaussians observer, extra false structure on no-device images, PSNR/SSIM. Mistakes found and fixed here:
wavelet `estimate_sigma` made NLM a no-op on correlated noise; a pooled ROC and a max-brightness observer were fooled by wall echo.

## Step 6 - diffusion barrier, mask robustness, IQA stability, real-frame check
```
python src/run_experiment.py --set guard --out outputs/results_guard.csv
python src/run_mask_robustness.py --n 300 --conds srad_lumen roi_full roi_g30 roi_d30
python src/run_iqa.py --ntrain 1200 --ntest 240 --k 4
python src/run_real_check.py --per-frame 5
python src/summarize_extra.py         # outputs/extra_summary.txt, guard/mask/iqa/real figures
```

## Step 7 - automatic device measurement and a 1-5 visibility/reliability grade (simulation)
```
python tests/test_measure.py
python src/run_measure_v3.py --ntrain 1600 --nval 200 --ntest 400 --workers 2   # about 20 minutes
python src/summarize_measure.py measure_results_v3.csv _v3                      # outputs/measure_summary_v3.txt
```
Measurand: visible span (largest distance between two device pixels) in the imaging plane. Per-pixel gradient-boosting segmenter, trained per
processing condition on the Apache video and tested on the 8 other videos with fresh seeds; threshold chosen on validation frames; scored once.
Two grades: visibility (binned clean-image device contrast, Spearman about 0.8) and reliability (predicted probability that the measurement is within 20%,
leave-one-video-out; AUC about 0.5 for raw, so it is not a usable gate). Both are simulation-defined, not expert grades.
History kept for honesty: `outputs/measure_results_v1_dev.csv` is a first version; only the final run (new seed) is reported.

## Step 8 - detect, then protect (ROI v3) and repeatability
```
python src/run_dp.py --nA 800 --nB 700 --nval 120 --ntest 300 --workers 2     # about 45 min, spawn-based pool
python src/summarize_dp.py                                                     # outputs/dp_summary.txt, dp_figure.png
python src/run_agreement.py --n 150 --k 4 --workers 2                          # about 40 min
python src/summarize_agreement.py                                              # outputs/agreement_summary.txt
python tests/test_protect.py
```
`roi.roi_filter_protected`: a gradient-boosting pixel classifier on the **raw** image (multi-scale intensity contrast, difference of Gaussians,
Hessian ridge response, local spread, distance to wall; plus large-scale background contrast and position relative to the lumen centroid) gives a
probability map; it is smoothed (sigma 1), thresholded (tau chosen on validation frames), cleaned (small blobs removed, the highest-mass blob and blobs
within 0.5 x equivalent diameter kept, two closings) and dilated 3 px. Labels = lumen + 2 x protect, so SRAD flux is zero across the protected border;
the SRAD noise reference q0 uses unprotected lumen pixels only; protected pixels are set back to their original values after 100 iterations.
With an empty protect mask it equals the barrier-only filter. The detector and measurement models are trained on disjoint Apache frames; scoring uses fresh
scenes from the 8 other videos. The pool start method is `spawn` (forking after gradient boosting deadlocked the workers).

## Step 9 - real frames: noise vs edge, segmentation, added noise, wrong masks
```
python src/prep_filtered.py --workers 2          # filters every real image once per condition -> outputs/cache_filtered.npz
python src/run_real_A.py --workers 2             # Experiment A: noise-vs-edge frontier with strength sweeps -> outputs/realA_rows.csv
python src/summarize_realA.py                    # outputs/realA_summary.txt, realA_figure.png (equal-smoothing comparison)
python src/run_real_seg.py --seeds 0 1 2 --epochs 40   # Experiment B: U-Net lumen segmentation on unseen videos -> outputs/realSeg_rows.csv
python src/summarize_realSeg.py                  # outputs/realSeg_summary.txt, realSeg_figure.png
python src/run_real_noise.py                     # Experiment C: frames with extra noise -> outputs/realNoise_rows.csv
python src/summarize_noise.py                    # outputs/realNoise_summary.txt, realNoise_figure.png
python src/run_real_maskrob.py                   # Experiment M: wrong masks guide the filter -> outputs/maskrob_summary.txt
```
* Conditions are fixed before looking at results: raw, global NLM (h = 1 x sigma), global SRAD (30 and 100 iterations), baseline three-region filter,
  barrier ROI (ROI v2), lumen-only NLM (h = 2 x sigma). Metrics are reference-free and reported as processed/raw: noise removed (`std_removed`, plane-detrended lumen spread),
  edge kept (maximum slope of the boundary profile, signed distance -10 to +10 px), tissue kept (high-pass spread outside).
* Segmenter: tiny U-Net (16 base channels, 3x3 conv + BN + ReLU, 128x128 input, BCE + soft Dice, Adam with OneCycle max 4e-3, batch 8, 40 epochs;
  augmentation: flips, shift +/-8 px, gain 0.9-1.1, bias +/-0.05; output > 0.5, fill holes, largest component).
  4 folds: the 53 Apache frames in 4 contiguous time blocks, the other videos spread by video; 3 frames on each side of a held-out block are dropped from training;
  3 seeds; every source frame is tested once.
* Two modes: `oracle` (hand-drawn mask guides the filter) and `predicted` (mask from the raw-image segmenter, the deployment mode and the headline).
* Statistics: paired bootstrap over source frames (2000 resamples); non-inferiority H0: dDice <= -0.02 vs H1: dDice > -0.02, accepted if the lower 95% limit exceeds -0.02.
  No multiplicity correction; secondary analyses are exploratory.

## Step 10 - image quality grade and lesion check (case images, local only)
```
python src/grade_prep.py        # lumen masks for the graded images + filtered versions -> grade/prep14.npz
python src/grade_cls.py         # frozen ResNet50 + linear 5-grade head, leave-one-image-out -> outputs/gradeCV_summary.txt
python src/grade_perm.py        # shuffled-label control -> outputs/gradePerm_summary.txt
python src/run_grade_real.py    # does a label-free quality score predict segmentation failure on real frames?
python src/lesion25_masks.py    # seeded lumen masks for the case images
python src/external25.py        # noise removed / edge kept / tissue kept / lesion texture -> outputs/external25_summary.txt
python src/lesion_check.py      # qualitative lesion figures;  python src/lesion_pair.py  # raw vs lumen-only NLM pair
```
Grader: approximation of USQNet (frozen ResNet50 features of the original plus 5 augmented views, `nn.Linear(2048, 5)`, Adam 1e-3, batch 2, 20 epochs;
ImageNet weights from timm `resnet50.a1_in1k` converted to torchvision layout and loaded from `/tmp/resnet50_a1.pth`: edit the path in `grade_cls.py` and `grade_perm.py`).
The Roboflow-trained U-Net does not transfer to the case-image scanners, so the lumen masks for these images are heuristic (`src/grade_masks.py`, dark region inside the
filled tissue, checked by eye). Do not report these as segmentation results.

## Main results (all numbers are in outputs/, medians or means with 95% bootstrap intervals)
| Question | Result | File |
|---|---|---|
| Does the U-Net find the lumen on unseen videos? | Dice 0.907 [0.886, 0.928] raw input (0.937 on the 53-frame video, 0.795 on the other videos); median HD95 4.0 px | `realSeg_summary.txt` |
| Equal smoothing (lumen spread removed 0.20): edge kept | lumen-only NLM 1.00, global NLM 0.99, ROI v2 0.88, Gaussian 0.62, global SRAD 0.60 | `realA_summary.txt` |
| Does filtering lower segmentation (predicted mask)? | lumen-only NLM -0.004 [-0.011, +0.003] (non-inferior); global NLM +0.001 (non-inferior); ROI v2 -0.010 [-0.024, -0.000] (not non-inferior). Oracle mask: lumen-only NLM +0.015 [+0.009, +0.023] | `realSeg_summary.txt` |
| Wrong mask (eroded/dilated 3-6 px, shifted 4 px) | lumen-only NLM edge and tissue kept >= 0.994; ROI v2 edge 0.81-0.94, tissue about 0.69 | `maskrob_summary.txt` |
| Added noise (segmenter trained on original frames) | global filters rescue Dice more than lumen-only NLM (level 0.30: global NLM 0.808, lumen-only NLM 0.685, no filter 0.640) | `realNoise_summary.txt` |
| Device, no barrier vs barrier (600 simulated scenes) | false structure +11.4 vs -0.4 grey levels; detection change -0.190 vs +0.008 | `extra_summary.txt` |
| Detect-then-protect (296 fresh scenes) | device contrast kept 0.94 vs 0.69 (barrier only) at equal noise removal 0.39; median 86% of true device pixels protected | `dp_summary.txt` |
| Device span within 20% (397 scenes) | raw 0.42, barrier-guarded SRAD 0.52 (+0.104 [+0.053, +0.160]); detect-then-protect not better than barrier only (-0.044 [-0.097, +0.024]) | `measure_summary_v3.txt`, `dp_summary.txt` |
| Repeatability of device span | wCV about 0.24 in every condition, limits of agreement about +/-70-80% | `agreement_summary.txt` |
| Lesion texture kept (case images with a usable lumen mask, lesion measured in 5) | lumen-only NLM 1.00 (min 0.96), global NLM 0.74 (min 0.45), global SRAD 0.36 | `external25_summary.txt` |
| Quality grade (leave-one-image-out, my 14 graded images) | exact 0.07, within one grade 0.86, majority baseline 0.36; shuffled labels give within-one about 0.82, so feasibility only | `gradeCV_summary.txt`, `gradePerm_summary.txt` |

## What this does and does not show
* Lumen-only denoising keeps edges and tissue and does not lower lumen segmentation on unseen videos (non-inferiority, not improvement).
* Its price: wall and outer-tissue noise stay, so under heavy added noise global filters help segmentation more.
* Device results are simulation only; there are no real device-in-bladder frames. About 14% of device pixels (median) are not protected. Span measurement is imprecise.
* The dataset is small (67 source frames, one dominant video); the lesion check is small; the quality grader is a pilot and is not better than chance on within-one accuracy.
* Hand-drawn masks come from the dataset authors; I did not re-annotate them.

## Reproducibility notes
* Seeds are fixed in each script; long runs use 2 workers and take minutes to under an hour (times above are for 2 cores).
* Hard-coded local paths to change: `/tmp/u/bladder pic/` (case images) and `/tmp/resnet50_a1.pth` (ResNet50 weights).
* Defense notes and Q&A are in `DEFENSE_OUTLINE.md` (written for the earlier title; the numbers are in `outputs/`).
* The device figure for the slides is `outputs/deck_device.png`; the preprocessing figure is `outputs/deck_prep.png`.

## References
1. D. Raina, S. H. Chandrashekhara, R. Voyles, J. Wachs, S. K. Saha, "Deep-learning model for quality assessment of urinary bladder ultrasound images using multiscale and higher-order processing," IEEE TUFFC 71(11):1451-1463, 2024, doi 10.1109/TUFFC.2024.3386919.
2. O. V. Michailovich, A. Tannenbaum, "Despeckling of medical ultrasound images," IEEE TUFFC 53(1):64-78, 2006, doi 10.1109/TUFFC.2006.1588392.
3. Y. Yu, S. T. Acton, "Speckle reducing anisotropic diffusion," IEEE Trans. Image Process. 11(11):1260-1270, 2002, doi 10.1109/TIP.2002.804276.
4. A. Buades, B. Coll, J.-M. Morel, "A non-local algorithm for image denoising," CVPR 2005, pp. 60-65, doi 10.1109/CVPR.2005.38.
5. D. Raina, S. H. Chandrashekhara, S. K. Saha, "Slim U-Net: Efficient anatomical feature preserving U-Net architecture for ultrasound image segmentation," ACM, 2023, doi 10.1145/3574198.3574205.
6. N. D. Shore et al., "Phase 3 study of TAR-210 vs intravesical chemotherapy in BCG-treated high-risk NMIBC," AUA Annual Meeting, 2025.
