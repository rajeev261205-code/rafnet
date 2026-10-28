# Private data splits (counts only — no filenames, no contact names)

Segments: 763 (files ok=64, failed/skipped=41)
Raw cuts=764 dropped_silence=1 dropped_clipped=0 quarantined_tonal=0 balance_moved=618

## By split|source|group
- adapt|phone|call_A: 60
- adapt|whatsapp|whatsapp_early: 15
- quarantine|phone|call_A: 618
- test|phone|call_B: 40
- test|whatsapp|whatsapp_late: 30

TEST rule: whatsapp_late dates + whole smaller contact + 2 ogg; never train/tune/select on test
Adapt share: adapt pool is the SMALL domain share (<=15% of human train at Phase 4)

## Speaker-leak check
{"cv_accuracy": 0.676, "n_groups": 4, "chance": 0.25, "verdict": "HIGH \u2014 reduce private share / add public speakers"}

## Acoustic gap (private real vs clean LibriSpeech)
{
 "rms_dbfs": {
  "private": {
   "median": -15.73,
   "mean": -16.86
  },
  "clean_public": {
   "median": -27.05,
   "mean": -27.05
  }
 },
 "snr_db": {
  "private": {
   "median": 28.4,
   "mean": 28.56
  },
  "clean_public": {
   "median": 39.64,
   "mean": 39.64
  }
 },
 "rolloff_hz": {
  "private": {
   "median": 2156.9,
   "mean": 2187.65
  },
  "clean_public": {
   "median": 4184.05,
   "mean": 4184.05
  }
 }
}

Training coverage consequence: see reports/acoustic_gap.md (Phase 1 must add noisy/band-limited public real + codec-fair augmentation both classes).

Manifest (local-only): samples/real/manifest.csv + manifest_summary.json; segments: samples/real/segments/<test|adapt|quarantine>/*.wav
