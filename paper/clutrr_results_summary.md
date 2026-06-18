# Original CLUTRR 15-epoch paper results

Configuration: `gen_train234_test2to10`  
Device: CPU  
Epochs: 15

| Model | Parameters | Best validation | Final validation | Reported test |
|---|---:|---:|---:|---:|
| ARM | 758,024 | 0.9810 | 0.9810 | 0.2710 |
| Enhanced ARM | 1,053,630 | 0.9710 | 0.9710 | 0.2653 |
| Dot memory | 625,792 | 0.9817 | 0.9797 | 0.2662 |
| Prototype memory | 609,281 | 0.9833 | 0.9827 | 0.2643 |
| RBF memory | 609,298 | 0.9840 | 0.9840 | 0.2634 |
| Hopfield memory | 611,603 | 0.8907 | 0.8907 | 0.2366 |
| Transformer | 781,842 | 0.8080 | 0.8077 | 0.2309 |

Interpretation used in the paper: this is a preliminary single-run result. ARM is competitive with the strongest metric-memory baselines on validation and obtains the highest reported test value in this run, but the low absolute test accuracies show that this setting remains a hard generalization problem and should not be treated as a conclusive superiority claim.

Use `paper/make_paper_plots.py` to regenerate the validation curve and summary bar figures from `paper/results_original_clutrr_15ep.json`.
