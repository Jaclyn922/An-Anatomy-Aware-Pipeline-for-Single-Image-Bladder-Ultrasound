# Device-aware despeckling for single-frame bladder ultrasound

Question: when only one frame is available, can despeckling raise device visibility and
quality-assessment stability without erasing the device or inventing structure that is not there?

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put the Roboflow export (images + `_annotations.coco.json`) directly in `data/roboflow/` (no extra `train/` folder).
Keep copyrighted pathology images in `data/pathology_local/` and do not commit them.
Run every command from the project root.

## Step 1 - data and masks
```
python src/prepare_data.py --root data/roboflow --out data/masks --raw_out data/masks_raw
python src/compare_masks.py      # outputs/mask_cleaning.png
python src/make_previews.py      # outputs/mask_check.png, outputs/device_preview.png
```
`data/manifest.csv` has `frame` and `video` columns. Always split by `video` (or at least `frame`),
never by file: the 200 files are only 67 source frames, 53 of them from one video.
Masks are hand-drawn and cover the dark lumen somewhat tightly; the cleaning step only fills holes and removes specks.

## Step 2 - ROI despeckling (the method under test)
```
python tests/test_srad.py
```
`src/roi.py` holds the corrected SRAD and the three-region filter (SRAD in lumen, bilateral on the wall band, light Gaussian outside).

## Step 3 - device test object
`src/device.py`: a pretzel-like 8-shaped tube (inspired by the public TAR-210 schematic; size, material and
echo strength are assumptions, so call it "device-like"), sliced by a random imaging plane. The 2D cross-section is the ground truth.

## Step 4 - ultrasound simulator with a clean reference
```
python tests/test_simulate.py
python src/calibrate.py --looks 2     # fits simulator parameters to the real frames (about 3 minutes)
python src/make_sim_figure.py         # outputs/sim_vs_real.png
```
`src/simulate.py`:
* `build_scene()` makes the expected-reflectivity map from a lumen mask (smooth wall from the signed distance,
  textured tissue, attenuation with TGC, optional device with shadow and reverberation).
* `render()` turns it into a noisy image and a clean reference: complex-Gaussian speckle blurred by the PSF,
  `looks` independent patterns averaged (spatial compounding), log compression, black level.
* `calibrate.py` fits gain, dynamic range, wall and tissue level, PSF size on the Apache video and reports the
  same measurements on the other 8 sources as a held-out check.
Calibrated values are in `outputs/sim_calibration.json`.

## Known limits of the simulator
* Tissue is speckle plus smooth texture only: no fascia layers, no bright posterior structures.
* The wall is a uniform ring; real walls vary around the lumen.
* Levels were fitted on one video. Other sources have different gain (real wall level 72 vs 124), so later
  experiments randomise gain and dynamic range instead of using one setting.
* Device shape, size and echo strength are assumptions.

## Step 5 - does despeckling hurt device visibility? (core experiment)
```
python tests/test_metrics.py          # metrics must catch a deliberate eraser and a deliberate hallucinator
python src/run_experiment.py --n 600  # about 9 minutes on 2 cores -> outputs/results.csv
python src/summarize.py               # outputs/summary.csv, sweep.csv, results_figure.png
python src/make_example_figure.py     # outputs/example_conditions.png
```
* `src/degrade.py` samples random acquisition conditions (gain, dynamic range, resolution, speckle looks, device echo,
  shadow, touching the wall) and renders a PRESENT and an ABSENT image per scene with the same anatomy.
* `src/conditions.py`: raw, global NLM, global SRAD, lumen-only SRAD, ROI method (30 it), plus ROI at 10/100/200 it.
  OBNLM is not included (not in the original repository, not re-implemented).
* `src/metrics.py`: contrast retention, 2AFC detection with a difference-of-Gaussians blob observer, extra false
  structure on no-device images, PSNR/SSIM against the clean reference.
* All intervals are cluster bootstraps over source frames.

Mistakes found and fixed during this step (kept here so nobody repeats them): wavelet `estimate_sigma` made NLM a
no-op on correlated speckle; a pooled ROC mixed scene-to-scene noise with the device signal and a max-brightness
observer was fooled by wall echo leaking into the lumen; both replaced (matched 2AFC, DoG observer chosen after
checking raw images only).

## Next
Step 6: IQA stability (same scene, different noise seed), mask-error robustness, a device-aware variant that stops
diffusion at the lumen boundary, and the real-frame sanity check.

## Step 6 — guard (diffusion barrier), mask robustness, IQA stability, real-frame check
Run order: `run_experiment.py --set guard --out outputs/results_guard.csv`; `run_mask_robustness.py --n 300 --conds srad_lumen roi_full roi_g30 roi_d30`; `run_iqa.py`; `run_real_check.py`; then `summarize_extra.py` (tables → outputs/extra_summary.txt, figures guard/mask/iqa/real). Conclusions and defense outline: DEFENSE_OUTLINE.md.
Bug fixed during the run: mask-robustness divided by zero clean contrast for invisible devices (retention now only for visible ones).

## Step 7 - automatic device measurement and a 1-5 quality/reliability grade (the usable output)
```
python tests/test_measure.py
python src/run_measure.py --ntrain 900 --ntest 400 --workers 2   # about 9 minutes -> outputs/measure_results.csv, outputs/segmenters/*.pkl
python src/summarize_measure.py                                   # outputs/measure_summary.txt, measure_figure.png
python src/validate_real.py --csv my_reference_measurements.csv --segmenter outputs/segmenters/segmenter_roi_g100.pkl
```
* Measurand: visible span (largest distance between two device pixels) in the imaging plane, and its centre. Errors are relative
  to the true span, so no pixel size is needed; multiply by mm/pixel for real data. A 2D cut does not give the 3D length.
* A simple threshold detector was tried first (parameters scanned on a separate development set, best Dice about 0.3) and
  is kept only as a baseline in `measure.py`. The experiment uses a per-pixel gradient-boosting segmenter (scikit-learn),
  trained per processing condition on the Apache video and tested on the 8 other videos with fresh seeds.
* Development history, stated for honesty: a first version (without ridge features, test seed 400) is saved as
  `outputs/measure_results_v1_dev.csv`. After seeing it, ridge features and reliability grading were added and the final run
  used a NEW test seed (500). Only the final run is reported.
* Two grades: (1) visibility grade = binned clean-image device contrast, predicted from the image (Spearman about 0.8);
  (2) reliability grade = predicted probability that the measurement is within 20%, trained leave-one-video-out.
  Both are simulation-defined; they are not the expert-labelled USQNet grade. A trained USQNet/ResNet scorer can replace them.
* `validate_real.py` is the script for real data with a reference length (bias, limits of agreement, within 20%).

### Step 7 v3 (final measurement numbers)
```
python src/run_measure_v3.py --ntrain 1600 --nval 200 --ntest 400 --workers 2   # about 20 minutes -> outputs/measure_results_v3.csv
python src/summarize_measure.py measure_results_v3.csv _v3                      # outputs/measure_summary_v3.txt, measure_figure_v3.png
```
v3 = more training scenes, context features, dominant-cluster post-processing, probability threshold chosen on 11 validation frames
(not used for training), scored once on fresh test scenes (seed 600). Final numbers in the defense outline come from v3.
The earlier run (seed 500, weaker model) is kept for comparison; both give the same ranking of methods.

## Step 8 - detect, then protect (ROI v3) and Step 9 - technical performance
```
python src/run_dp.py --nA 800 --nB 700 --nval 120 --ntest 300 --workers 2     # about 45 min, spawn-based pool
python src/summarize_dp.py                                                     # outputs/dp_summary.txt, dp_figure.png
python src/run_agreement.py --n 150 --k 4 --workers 2                          # about 40 min -> outputs/agreement_rows.csv
python src/summarize_agreement.py                                              # outputs/agreement_summary.txt, agreement_figure.png
```
* `roi.roi_filter_protected` walls a detected device candidate off from the diffusion (tests in `tests/test_protect.py`).
* Detector (stage 1) and measurement models (stage 2) are trained on disjoint Apache frames; thresholds are chosen on validation frames;
  scoring uses fresh test scenes from the 8 other videos.
* The agreement study images each scene K=4 times with new noise and reports bias, limits of agreement, repeatability (wCV, RC), ICC,
  linearity and failure rate. Findings: bias about 0, limits of agreement about +/-70-80%, wCV about 0.24, identical across processing.
* Pool start method is `spawn`: forking after gradient boosting (OpenMP) deadlocked the workers.
