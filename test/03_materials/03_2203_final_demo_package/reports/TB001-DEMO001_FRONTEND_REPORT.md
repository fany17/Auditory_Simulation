# TB001-DEMO001 convolutional frontend experiment

## Question

For the same pitch-preserving 2× speech input, does reducing wav2vec2 convolutional temporal downsampling change or improve the final CTC transcript without retraining?

## Intervention

- Standard: convolution strides `[5, 2, 2, 2, 2, 2, 2]`, total stride `320` samples, measured `49.6` CTC frames/s.
- 100 Hz variant: convolution strides `[5, 2, 2, 2, 2, 2, 1]`, total stride `160` samples, measured `99.2` CTC frames/s.
- 200 Hz variant: convolution strides `[5, 2, 2, 2, 2, 1, 1]`, total stride `80` samples, measured `198.8` CTC frames/s.
- All pretrained weights remain unchanged; no fine-tuning or retraining was performed.

## Real 2× results

| Frontend | CTC frames | Transcript | CER vs reference |
|---|---:|---|---:|
| Standard ~50 Hz | 127 | `AS RITINGS THERE ARE TWO KINDS BRISH AND VON` | 31.6% |
| Denser ~100 Hz | 254 | `AS FOR ETCHINGS THEY ARE OF TWO KINDS GRITISH AND FOREIGN` | 1.8% |
| Denser ~200 Hz | 509 | `AS FOR AGINGS THERE ARE F FEW KINDS GRUISH AND FOREIGN` | 22.8% |

The 100 Hz variant substantially improved this utterance, while 200 Hz was worse than 100 Hz. Therefore changing the convolutional frontend can change and sometimes improve the output, but higher temporal density is not monotonically better.

## Boundary

Speed conversion may already remove acoustic detail. Denser convolutional sampling cannot reconstruct missing waveform information; it only reduces additional temporal downsampling inside the model. Cross-rate hidden/CTC distances use linear interpolation over normalized utterance time and are engineering comparison metrics. This single-utterance result does not establish general fast-speech performance.

## Execution and QA

- 5 speeds × 3 frontend settings = 15 real inferences; failed = 0.
- Executed by ordinary Python over controller-driven SSH; remote Codex Agent tokens = 0.
- Desktop browser QA passed frontend selection (50/100/200 Hz), return to Transformer L9/α=0.5, explanatory text, transcript diff, and audio readiness (`readyState=4`, 2.56 s for 2× input).

## Standalone CER figure and limited compensation demo

- Standalone files: `reports/figures/frontend_speed_cer.svg` and `reports/figures/frontend_speed_cer.png`.
- Auditable source table: `reports/figures/frontend_speed_cer_source.csv`.
- The displayed same-sample policy uses 50 Hz at ≤1.5× and 100 Hz at ≥1.75×, yielding CER (%) `[0.0, 0.0, 0.0, 0.0, 1.7543859649122806]` across the five tested speeds.
- This curve is selected from the same 15 real inference results; it adds no inference and is neither held-out validation nor a general optimum.
