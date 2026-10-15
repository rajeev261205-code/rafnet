# Model evaluation — stacked ensemble (frozen VAL-fit, TEST + my_samples)

Generated: 2026-10-15T16:10:06.722913+00:00
Model: ensemble mo-thecreator + Bisher + RAFNet; stacking=models/stacking.json (VAL-fit, frozen here).
Taus: [0.05, 0.55] (VAL-chosen, frozen).

> TEST and my_samples were never used for training or band selection. Small-n snapshot, NOT a general benchmark.

## TEST (unseen speakers/setting, n=80)
accuracy=0.9375 balanced=0.9375 AUC=0.9953 EER={"eer": 0.0375, "threshold": 0.9845, "far": 0.05, "frr": 0.025}
FAR on human (most important)=0.1 binary={"accuracy": 0.9375, "precision": 0.907, "recall": 0.975, "f1": 0.9398, "tp": 39, "tn": 36, "fp": 4, "fn": 1, "n": 80}
confusion(band)={"real": {"REAL": 0, "UNCERTAIN": 36, "SYNTHETIC": 4}, "fake": {"REAL": 0, "UNCERTAIN": 1, "SYNTHETIC": 39}} coverage=0.5375 selective=0.907

## my_samples (unseen generators, n=19)
accuracy=1.0 balanced=1.0 AUC=1.0 EER={"eer": 0.0, "threshold": 0.9846, "far": 0.0, "frr": 0.0}
FAR on human=0.0 binary={"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 15, "tn": 4, "fp": 0, "fn": 0, "n": 19}
confusion(band)={"real": {"REAL": 0, "UNCERTAIN": 4, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 15}}

## Per group
- clean_human_librispeech: n=44 acc=0.9091 bal=None AUC=None EER={"eer": null, "threshold": null, "note": "needs both classes"} FAR_human=0.0909 conf={"real": {"REAL": 0, "UNCERTAIN": 40, "SYNTHETIC": 4}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
- voice_conversion_crudeVC: n=40 acc=0.975 bal=None AUC=None EER={"eer": null, "threshold": null, "note": "needs both classes"} FAR_human=None conf={"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 1, "SYNTHETIC": 39}}
- windows_sapi_tts: n=4 acc=1.0 bal=None AUC=None EER={"eer": null, "threshold": null, "note": "needs both classes"} FAR_human=None conf={"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 4}}
- assistant_voice_chatgpt: n=1 acc=1.0 bal=None AUC=None EER={"eer": null, "threshold": null, "note": "needs both classes"} FAR_human=None conf={"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 1}}
- windows_tts: n=1 acc=1.0 bal=None AUC=None EER={"eer": null, "threshold": null, "note": "needs both classes"} FAR_human=None conf={"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 1}}
- windows_sapi_tts_extended: n=9 acc=1.0 bal=None AUC=None EER={"eer": null, "threshold": null, "note": "needs both classes"} FAR_human=None conf={"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 9}}

## Per codec (test-real only, clean p_cal -> codec p_cal)
- data\raw\LibriSpeech\test-clean\1284\1181\1284-1181-0008.flac opus_24k: 0.0558->0.0558 d=0.0 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\1284\1181\1284-1181-0008.flac mp3_128k: 0.0558->0.0558 d=0.0 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\1284\1181\1284-1181-0008.flac amr_nb: 0.0558->0.0558 d=0.0 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\1284\1181\1284-1181-0008.flac g711_alaw: 0.0558->0.0558 d=0.0 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\2300\131720\2300-131720-0027.flac opus_24k: 0.0566->0.0567 d=0.0001 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\2300\131720\2300-131720-0027.flac mp3_128k: 0.0566->0.0567 d=0.0001 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\2300\131720\2300-131720-0027.flac amr_nb: 0.0566->0.0567 d=0.0001 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\2300\131720\2300-131720-0027.flac g711_alaw: 0.0566->0.0567 d=0.0001 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0024.flac opus_24k: 0.0558->0.0558 d=0.0 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0024.flac mp3_128k: 0.0558->0.0558 d=0.0 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0024.flac amr_nb: 0.0558->0.0571 d=0.0013 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0024.flac g711_alaw: 0.0558->0.0558 d=0.0 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\6829\68771\6829-68771-0019.flac opus_24k: 0.9836->0.9844 d=0.0008 SYNTHETIC 
- data\raw\LibriSpeech\test-clean\6829\68771\6829-68771-0019.flac mp3_128k: 0.9836->0.9839 d=0.0003 SYNTHETIC 
- data\raw\LibriSpeech\test-clean\6829\68771\6829-68771-0019.flac amr_nb: 0.9836->0.0557 d=-0.9279 UNCERTAIN 
- data\raw\LibriSpeech\test-clean\6829\68771\6829-68771-0019.flac g711_alaw: 0.9836->0.7685 d=-0.2151 SYNTHETIC 

## Reliability (calibrated p bins vs binary accuracy, TEST+my)
| bin | n | accuracy | avg_p |
| --- | --- | --- | --- |
| 0.0-0.1 | 41 | 0.9756 | 0.0563 |
| 0.1-0.3 | 0 | None | None |
| 0.3-0.5 | 0 | None | None |
| 0.5-0.7 | 0 | None | None |
| 0.7-0.9 | 0 | None | None |
| 0.9-1.0 | 58 | 0.931 | 0.9923 |

## Honest reading
- Targets: none pre-registered; nothing tuned here. If a deployment bar (e.g. FAR_human<=0.05 on WhatsApp audio) is required, it is NOT met by evidence yet — missing groups below.
- Missing (unmeasured): whatsapp_opus_human, phone_line_human, Hindi/Tamil, commercial TTS, real VC. The History failure (WhatsApp human flagged SYNTHETIC) cannot be closed without those clips.
- Train fake was one crude VC family; TEST fake is its held-out setting — expect degradation on true unseen generators; my_samples SAPI/ChatGPT is the only genuine unseen-generator gate and is tiny (n=15 fake).
- Next data step: 10-20 WhatsApp Opus notes + Indian-language + noisy real; public WaveFake/ASVspoof subset via GPU notebook (needs approval >15 GB). Next model step: run notebooks/finetune_gpu_phase3.ipynb, then re-run this script unchanged.

## Files (per-file TEST/my)
- data\raw\LibriSpeech\test-clean\1284\1181\1284-1181-0008.flac y=0 [clean_human_librispeech] p_cal=0.0558 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.7835)
- data\raw\LibriSpeech\test-clean\2300\131720\2300-131720-0027.flac y=0 [clean_human_librispeech] p_cal=0.0566 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.895)
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0024.flac y=0 [clean_human_librispeech] p_cal=0.0558 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.7848)
- data\raw\LibriSpeech\test-clean\6829\68771\6829-68771-0019.flac y=0 [clean_human_librispeech] p_cal=0.9836 -> SYNTHETIC votes=2/3 (mo=0.9933 bi=0.0021 ra=0.7623)
- data\raw\LibriSpeech\test-clean\1284\1180\1284-1180-0011.flac y=0 [clean_human_librispeech] p_cal=0.0562 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8399)
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0041.flac y=0 [clean_human_librispeech] p_cal=0.0558 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.7856)
- data\raw\LibriSpeech\test-clean\2830\3980\2830-3980-0014.flac y=0 [clean_human_librispeech] p_cal=0.056 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8127)
- data\raw\LibriSpeech\test-clean\2830\3980\2830-3980-0005.flac y=0 [clean_human_librispeech] p_cal=0.0564 -> UNCERTAIN votes=1/3 (mo=0.0001 bi=0.002 ra=0.8626)
- data\raw\LibriSpeech\test-clean\1284\1181\1284-1181-0001.flac y=0 [clean_human_librispeech] p_cal=0.0564 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.0021 ra=0.8567)
- data\raw\LibriSpeech\test-clean\5105\28233\5105-28233-0006.flac y=0 [clean_human_librispeech] p_cal=0.0558 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.7832)
- data\raw\LibriSpeech\test-clean\1284\1181\1284-1181-0018.flac y=0 [clean_human_librispeech] p_cal=0.0563 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8512)
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0012.flac y=0 [clean_human_librispeech] p_cal=0.0561 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.0021 ra=0.8252)
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0033.flac y=0 [clean_human_librispeech] p_cal=0.0563 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.0021 ra=0.8461)
- data\raw\LibriSpeech\test-clean\2830\3980\2830-3980-0053.flac y=0 [clean_human_librispeech] p_cal=0.056 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8133)
- data\raw\LibriSpeech\test-clean\1580\141084\1580-141084-0037.flac y=0 [clean_human_librispeech] p_cal=0.0558 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.7871)
- data\raw\LibriSpeech\test-clean\2830\3979\2830-3979-0011.flac y=0 [clean_human_librispeech] p_cal=0.0561 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8278)
- data\raw\LibriSpeech\test-clean\1284\1180\1284-1180-0024.flac y=0 [clean_human_librispeech] p_cal=0.0561 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8278)
- data\raw\LibriSpeech\test-clean\2300\131720\2300-131720-0026.flac y=0 [clean_human_librispeech] p_cal=0.0567 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.0021 ra=0.8984)
- data\raw\LibriSpeech\test-clean\2300\131720\2300-131720-0033.flac y=0 [clean_human_librispeech] p_cal=0.0566 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.0021 ra=0.8886)
- data\raw\LibriSpeech\test-clean\1580\141084\1580-141084-0024.flac y=0 [clean_human_librispeech] p_cal=0.056 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.0021 ra=0.8053)
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0020.flac y=0 [clean_human_librispeech] p_cal=0.9434 -> SYNTHETIC votes=2/3 (mo=0.8067 bi=0.002 ra=0.8856)
- data\raw\LibriSpeech\test-clean\1284\134647\1284-134647-0002.flac y=0 [clean_human_librispeech] p_cal=0.0564 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.0021 ra=0.8611)
- data\raw\LibriSpeech\test-clean\1284\1181\1284-1181-0021.flac y=0 [clean_human_librispeech] p_cal=0.0561 -> UNCERTAIN votes=1/3 (mo=0.0001 bi=0.002 ra=0.8212)
- data\raw\LibriSpeech\test-clean\2830\3980\2830-3980-0036.flac y=0 [clean_human_librispeech] p_cal=0.056 -> UNCERTAIN votes=1/3 (mo=0.0002 bi=0.002 ra=0.7962)
- data\raw\LibriSpeech\test-clean\6829\68769\6829-68769-0053.flac y=0 [clean_human_librispeech] p_cal=0.9846 -> SYNTHETIC votes=2/3 (mo=0.9997 bi=0.0021 ra=0.8727)
- data\raw\LibriSpeech\test-clean\1221\135766\1221-135766-0001.flac y=0 [clean_human_librispeech] p_cal=0.0566 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.0021 ra=0.8901)
- data\raw\LibriSpeech\test-clean\1221\135767\1221-135767-0006.flac y=0 [clean_human_librispeech] p_cal=0.0562 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8331)
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0013.flac y=0 [clean_human_librispeech] p_cal=0.0568 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.913)
- data\raw\LibriSpeech\test-clean\2830\3980\2830-3980-0054.flac y=0 [clean_human_librispeech] p_cal=0.056 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8125)
- data\raw\LibriSpeech\test-clean\2830\3979\2830-3979-0010.flac y=0 [clean_human_librispeech] p_cal=0.0564 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8682)
- data\raw\LibriSpeech\test-clean\1580\141083\1580-141083-0026.flac y=0 [clean_human_librispeech] p_cal=0.0561 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8254)
- data\raw\LibriSpeech\test-clean\6829\68771\6829-68771-0020.flac y=0 [clean_human_librispeech] p_cal=0.9845 -> SYNTHETIC votes=2/3 (mo=0.9996 bi=0.002 ra=0.8279)
- data\raw\LibriSpeech\test-clean\5105\28240\5105-28240-0004.flac y=0 [clean_human_librispeech] p_cal=0.056 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8102)
- data\raw\LibriSpeech\test-clean\5105\28241\5105-28241-0006.flac y=0 [clean_human_librispeech] p_cal=0.0565 -> UNCERTAIN votes=1/3 (mo=0.0002 bi=0.0021 ra=0.8652)
- data\raw\LibriSpeech\test-clean\2830\3980\2830-3980-0029.flac y=0 [clean_human_librispeech] p_cal=0.056 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8149)
- data\raw\LibriSpeech\test-clean\2300\131720\2300-131720-0037.flac y=0 [clean_human_librispeech] p_cal=0.056 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8127)
- data\raw\LibriSpeech\test-clean\1284\1180\1284-1180-0010.flac y=0 [clean_human_librispeech] p_cal=0.0564 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8648)
- data\raw\LibriSpeech\test-clean\2830\3980\2830-3980-0063.flac y=0 [clean_human_librispeech] p_cal=0.0562 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8332)
- data\raw\LibriSpeech\test-clean\1221\135767\1221-135767-0013.flac y=0 [clean_human_librispeech] p_cal=0.0563 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8468)
- data\raw\LibriSpeech\test-clean\5105\28240\5105-28240-0022.flac y=0 [clean_human_librispeech] p_cal=0.0559 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.7927)
- data/synth_vc/vc_ts_fast_spk-1284_000.wav y=1 [voice_conversion_crudeVC] p_cal=0.9988 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9368 ra=0.862)
- data/synth_vc/vc_ts_fast_spk-1284_001.wav y=1 [voice_conversion_crudeVC] p_cal=0.9928 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.2781 ra=0.917)
- data/synth_vc/vc_ts_fast_spk-1284_002.wav y=1 [voice_conversion_crudeVC] p_cal=0.9865 -> SYNTHETIC votes=2/3 (mo=0.9999 bi=0.0584 ra=0.7489)
- data/synth_vc/vc_ts_fast_spk-1580_003.wav y=1 [voice_conversion_crudeVC] p_cal=0.9887 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.1147 ra=0.9007)
- data/synth_vc/vc_ts_fast_spk-6829_004.wav y=1 [voice_conversion_crudeVC] p_cal=0.999 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9954 ra=0.8943)
- data/synth_vc/vc_ts_fast_spk-1580_005.wav y=1 [voice_conversion_crudeVC] p_cal=0.9968 -> SYNTHETIC votes=3/3 (mo=0.9999 bi=0.5943 ra=0.6348)
- data/synth_vc/vc_ts_fast_spk-2830_006.wav y=1 [voice_conversion_crudeVC] p_cal=0.9847 -> SYNTHETIC votes=2/3 (mo=0.9999 bi=0.0036 ra=0.8664)
- data/synth_vc/vc_ts_fast_spk-2300_007.wav y=1 [voice_conversion_crudeVC] p_cal=0.9988 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9273 ra=0.9321)
- data/synth_vc/vc_ts_fast_spk-2300_008.wav y=1 [voice_conversion_crudeVC] p_cal=0.9974 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.6532 ra=0.9118)
- data/synth_vc/vc_ts_fast_spk-5105_009.wav y=1 [voice_conversion_crudeVC] p_cal=0.9887 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.1151 ra=0.9093)
- data/synth_vc/vc_ts_fast_spk-2830_010.wav y=1 [voice_conversion_crudeVC] p_cal=0.9949 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.4075 ra=0.8781)
- data/synth_vc/vc_ts_fast_spk-2830_011.wav y=1 [voice_conversion_crudeVC] p_cal=0.9886 -> SYNTHETIC votes=2/3 (mo=0.9999 bi=0.1161 ra=0.8255)
- data/synth_vc/vc_ts_fast_spk-5105_012.wav y=1 [voice_conversion_crudeVC] p_cal=0.9848 -> SYNTHETIC votes=2/3 (mo=0.9997 bi=0.0066 ra=0.8945)
- data/synth_vc/vc_ts_fast_spk-1284_013.wav y=1 [voice_conversion_crudeVC] p_cal=0.9986 -> SYNTHETIC votes=3/3 (mo=0.9992 bi=0.8889 ra=0.9213)
- data/synth_vc/vc_ts_fast_spk-2300_014.wav y=1 [voice_conversion_crudeVC] p_cal=0.9961 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.5056 ra=0.9226)
- data/synth_vc/vc_ts_fast_spk-1284_015.wav y=1 [voice_conversion_crudeVC] p_cal=0.9845 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.0027 ra=0.8251)
- data/synth_vc/vc_ts_fast_spk-5105_016.wav y=1 [voice_conversion_crudeVC] p_cal=0.985 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.0107 ra=0.8801)
- data/synth_vc/vc_ts_fast_spk-1580_017.wav y=1 [voice_conversion_crudeVC] p_cal=0.9884 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.1071 ra=0.8499)
- data/synth_vc/vc_ts_fast_spk-1580_018.wav y=1 [voice_conversion_crudeVC] p_cal=0.9919 -> SYNTHETIC votes=2/3 (mo=0.9999 bi=0.2397 ra=0.8523)
- data/synth_vc/vc_ts_fast_spk-2300_019.wav y=1 [voice_conversion_crudeVC] p_cal=0.9977 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.7023 ra=0.8923)
- data/synth_vc/vc_ts_fast_spk-1580_020.wav y=1 [voice_conversion_crudeVC] p_cal=0.9976 -> SYNTHETIC votes=3/3 (mo=0.9999 bi=0.6957 ra=0.677)
- data/synth_vc/vc_ts_fast_spk-5105_021.wav y=1 [voice_conversion_crudeVC] p_cal=0.9987 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9017 ra=0.8941)
- data/synth_vc/vc_ts_fast_spk-6829_022.wav y=1 [voice_conversion_crudeVC] p_cal=0.9983 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.8191 ra=0.893)
- data/synth_vc/vc_ts_fast_spk-6829_023.wav y=1 [voice_conversion_crudeVC] p_cal=0.9989 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9725 ra=0.8983)
- data/synth_vc/vc_ts_fast_spk-1580_024.wav y=1 [voice_conversion_crudeVC] p_cal=0.9845 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.0039 ra=0.7802)
- data/synth_vc/vc_ts_fast_spk-6829_025.wav y=1 [voice_conversion_crudeVC] p_cal=0.9989 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.977 ra=0.9085)
- data/synth_vc/vc_ts_fast_spk-2830_026.wav y=1 [voice_conversion_crudeVC] p_cal=0.9847 -> SYNTHETIC votes=2/3 (mo=0.9998 bi=0.0036 ra=0.8932)
- data/synth_vc/vc_ts_fast_spk-5105_027.wav y=1 [voice_conversion_crudeVC] p_cal=0.9846 -> SYNTHETIC votes=2/3 (mo=0.9999 bi=0.005 ra=0.8301)
- data/synth_vc/vc_ts_fast_spk-2830_028.wav y=1 [voice_conversion_crudeVC] p_cal=0.9848 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.0073 ra=0.84)
- data/synth_vc/vc_ts_fast_spk-2830_029.wav y=1 [voice_conversion_crudeVC] p_cal=0.9989 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9744 ra=0.8956)
- data/synth_vc/vc_ts_fast_spk-1284_030.wav y=1 [voice_conversion_crudeVC] p_cal=0.999 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9972 ra=0.895)
- data/synth_vc/vc_ts_fast_spk-6829_031.wav y=1 [voice_conversion_crudeVC] p_cal=0.9987 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.8968 ra=0.9064)
- data/synth_vc/vc_ts_fast_spk-1580_032.wav y=1 [voice_conversion_crudeVC] p_cal=0.9989 -> SYNTHETIC votes=3/3 (mo=0.9999 bi=0.9686 ra=0.6678)
- data/synth_vc/vc_ts_fast_spk-1580_033.wav y=1 [voice_conversion_crudeVC] p_cal=0.9884 -> SYNTHETIC votes=2/3 (mo=0.9999 bi=0.1158 ra=0.7125)
- data/synth_vc/vc_ts_fast_spk-6829_034.wav y=1 [voice_conversion_crudeVC] p_cal=0.9989 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9812 ra=0.912)
- data/synth_vc/vc_ts_fast_spk-1221_035.wav y=1 [voice_conversion_crudeVC] p_cal=0.9982 -> SYNTHETIC votes=3/3 (mo=0.9999 bi=0.7905 ra=0.8234)
- data/synth_vc/vc_ts_fast_spk-2300_036.wav y=1 [voice_conversion_crudeVC] p_cal=0.9964 -> SYNTHETIC votes=3/3 (mo=0.9999 bi=0.5326 ra=0.8985)
- data/synth_vc/vc_ts_fast_spk-2300_037.wav y=1 [voice_conversion_crudeVC] p_cal=0.0598 -> UNCERTAIN votes=1/3 (mo=0.0021 bi=0.0183 ra=0.8861)
- data/synth_vc/vc_ts_fast_spk-5105_038.wav y=1 [voice_conversion_crudeVC] p_cal=0.9985 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.8488 ra=0.9318)
- data/synth_vc/vc_ts_fast_spk-2830_039.wav y=1 [voice_conversion_crudeVC] p_cal=0.9853 -> SYNTHETIC votes=2/3 (mo=0.9999 bi=0.0193 ra=0.8658)
- samples/real/260-123286-0012.flac y=0 [clean_human_librispeech] p_cal=0.0562 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8323)
- samples/real/4446-2273-0021.flac y=0 [clean_human_librispeech] p_cal=0.0563 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.8439)
- samples/real/5142-33396-0059.flac y=0 [clean_human_librispeech] p_cal=0.0561 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.0021 ra=0.818)
- samples/real/6930-81414-0025.flac y=0 [clean_human_librispeech] p_cal=0.0559 -> UNCERTAIN votes=1/3 (mo=0.0 bi=0.002 ra=0.7917)
- samples/fake/sapi_david_2.wav y=1 [windows_sapi_tts] p_cal=0.9981 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.7684 ra=0.8238)
- samples/fake/sapi_david_4.wav y=1 [windows_sapi_tts] p_cal=0.9847 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.0062 ra=0.8099)
- samples/fake/sapi_zira_2.wav y=1 [windows_sapi_tts] p_cal=0.9988 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9333 ra=0.8498)
- samples/fake/sapi_zira_4.wav y=1 [windows_sapi_tts] p_cal=0.9846 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.0046 ra=0.8077)
- samples/fake/chatgpt_voice/rafnet_ai_voice_test.wav y=1 [assistant_voice_chatgpt] p_cal=0.999 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9969 ra=0.8603)
- samples/fake/windows_tts/tts_test.wav y=1 [windows_tts] p_cal=0.999 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9955 ra=0.8262)
- samples/win_tts/sapi_david_0.wav y=1 [windows_sapi_tts_extended] p_cal=0.9989 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9615 ra=0.8239)
- samples/win_tts/sapi_david_2.wav y=1 [windows_sapi_tts_extended] p_cal=0.9981 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.7684 ra=0.8238)
- samples/win_tts/sapi_david_4.wav y=1 [windows_sapi_tts_extended] p_cal=0.9847 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.0062 ra=0.8099)
- samples/win_tts/sapi_zira_0.wav y=1 [windows_sapi_tts_extended] p_cal=0.9988 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9532 ra=0.8236)
- samples/win_tts/sapi_zira_1.wav y=1 [windows_sapi_tts_extended] p_cal=0.9988 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9337 ra=0.814)
- samples/win_tts/sapi_zira_2.wav y=1 [windows_sapi_tts_extended] p_cal=0.9988 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9333 ra=0.8498)
- samples/win_tts/sapi_zira_3.wav y=1 [windows_sapi_tts_extended] p_cal=0.999 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.9981 ra=0.8158)
- samples/win_tts/sapi_zira_4.wav y=1 [windows_sapi_tts_extended] p_cal=0.9846 -> SYNTHETIC votes=2/3 (mo=1.0 bi=0.0046 ra=0.8077)
- samples/win_tts/sapi_zira_5.wav y=1 [windows_sapi_tts_extended] p_cal=0.9989 -> SYNTHETIC votes=3/3 (mo=1.0 bi=0.976 ra=0.8195)
