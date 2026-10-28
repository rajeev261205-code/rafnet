# Training plan — 5-class voice-origin detector (CPU-first)

Hardware here: 6-core Ryzen, 15.2 GB RAM, NO GPU, 50 GB free. Full encoder
fine-tuning is NOT feasible locally. Plan:

## Local (CPU) path — heads only, cached embeddings

1. `tools/make_manifest.py` → `data/manifest.csv` (speaker/generator-disjoint).
2. Embedding cache: WavLM-base frozen, 4 s windows (`preprocess.fixed_window`),
   layer-weighted mean → 768-d vectors, saved as `data/embed_cache/*.pt`.
   On CPU ≈ 2–4 s per window; 10k windows ≈ overnight batch job. Script:
   `tools/cache_embeddings.py` (CPU, resumable; generated with the notebooks).
3. Heads (sklearn/PyTorch-CPU, minutes): origin MLP (5 classes, class weights
   + focal loss), family classifier + open-set energy threshold fit on
   validation, IVR binary head. Early stopping on val macro-F1, seeds fixed.
4. Stacking: logistic regression on [new heads + RAFNet p_c/p_q/p_fused] fit
   on VALIDATION only. Temperature scaling on validation. UNCERTAIN band from
   validation selective-accuracy/coverage tradeoff. Test sets NEVER touched.
5. Artifacts: `models/origin_classifier.pt` + model card in the train report.

## GPU path — notebooks/

- `notebooks/extract_embeddings_kaggle.ipynb`: Kaggle GPU (T4 free), WavLM-base
  or wav2vec2-xls-r-300m, manifest upload, embedding extraction + augmentation,
  download shards back to `data/embed_cache/`.
- `notebooks/train_colab.ipynb`: Colab GPU (or Kaggle), head training +
  stacking + calibration + eval plots, download `models/origin_classifier.pt`.

## What is NOT done here

No encoder fine-tuning locally; no training run without the Phase-1 datasets;
no thresholds from test data. `reports/model_eval.md` stays UNEVALUATED until
a real run lands.
