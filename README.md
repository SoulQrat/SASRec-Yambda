# SASRec on Yambda

This repository contains my solution for **Homework 5** in the **Deep Learning 2 course at HSE FCS**: neural candidate generation for sequential recommendations.

**Assignment**: [link](https://github.com/thecrazymage/DL2_HSE/blob/main/2025/homeworks/homework_05/task.ipynb)

## Project Structure

```
├── task.ipynb          # Data pipeline, SASRec, losses, metrics, training and evaluation
├── sasrec.png          # SASRec architecture
├── requirments.txt
└── README.md
```

## Data

[Yambda](https://huggingface.co/datasets/yandex/yambda) is a dataset of Yandex Music listening logs. This project uses the `sequential/50m` listens split, preprocessed with Polars:

- users with `uid % 200 == 0`; only recommended (non-organic) listens played for at least 50% of the track;
- track ids remapped to `0..N-1`;
- time split without gaps: the last day is test, the day before it is validation, everything earlier is train. Validation and test samples are built incrementally: the full history up to an event is used to predict that event.

| | users | events |
|---|---|---|
| after filtering | 4.3K | 7.6M |
| train | 4.2K | 7.5M |
| validation | 1.4K | 36K |
| test | 1.4K | 41K |

The catalog has 305K tracks.

## Model

[SASRec](https://arxiv.org/abs/1808.09781) is a causal transformer encoder over the user's listening history: item embeddings plus learned positional embeddings, then 2 transformer layers with a causal mask and a padding mask. The score of item `i` at step `t` is the dot product of the hidden state with the item embedding (shared input/output embeddings).

Three training objectives are compared:

- **BCE**: one uniformly sampled negative per positive, as in the original paper.
- **In-batch**: sampled softmax with 256 negatives per positive drawn from the targets of the current batch.
- **In-batch + logQ**: the same, with logQ correction: `log Q(i)` is subtracted from negative logits, where `Q(i)` is the item's frequency among training targets. This removes the bias towards penalising popular items, which appear as in-batch negatives more often ([Yi et al., 2019](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)).

Hyperparameters: embedding dim 64, 2 heads, 2 layers, max sequence length 200, dropout 0.1, Adam with lr 1e-3, batch size 256, 18 epochs. The checkpoint with the best validation NDCG@1000 is kept.

## Results

Evaluation on the test day, ranking the full catalog of 305K tracks. Each sample has exactly one relevant item (the next track), so IDCG = 1, DCG@k equals NDCG@k, and HitRate@k equals Recall@k.

| Objective | NDCG@10 | NDCG@100 | HitRate@10 | HitRate@100 | HitRate@1000 |
|---|---|---|---|---|---|
| BCE, 1 random negative | 0.0018 | 0.0064 | 0.0041 | 0.0290 | 0.1445 |
| In-batch | **0.0074** | 0.0158 | 0.0134 | 0.0582 | 0.2223 |
| In-batch + logQ | 0.0071 | **0.0186** | **0.0139** | **0.0752** | **0.2630** |

Findings:
- Moving from one random negative to 256 in-batch negatives gives the largest gain: about 4× NDCG@10 and 2× HitRate@100. Random negatives from a 305K catalog are too easy, while in-batch negatives follow the popularity distribution and are harder.
- logQ correction matters most deeper in the ranking: HitRate@100 0.058 → 0.075 and HitRate@1000 0.222 → 0.263, with the top 10 essentially unchanged. Without the correction popular items are over-penalised as negatives; with it, more relevant tracks get into the candidate set, which is what a candidate generator is for.

## Known limitations

- **Accidental hits.** In-batch negatives are not filtered, so a negative can coincide with the positive (the same track for another user or at another position). The model is then penalised for the correct answer, most often on popular tracks. The standard fix is to mask such negatives with a `-inf` logit.
