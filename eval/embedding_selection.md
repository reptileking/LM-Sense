Gold pairs: 300 (dev 147 / test 153). Thresholds chosen on dev.

| variant | model | dtype | texts/s (4 CPU) | fine AUC | fine test P/R/F1 @thr | coarse AUC | coarse test P/R/F1 @thr |
|---|---|---|---|---|---|---|---|
| bge-small/gloss+ex | bge-small-en-v1.5 | float32 | 305.7 | 0.873 | 0.67/0.89/0.77 @0.637 | 0.904 | 0.78/0.86/0.82 @0.637 |
| bge-base/gloss | bge-base-en-v1.5 | float32 | 93.7 | 0.920 | 0.81/0.88/0.84 @0.676 | 0.935 | 0.81/0.95/0.87 @0.591 |
| bge-base/gloss+ex | bge-base-en-v1.5 | float32 | 93.7 | 0.902 | 0.78/0.86/0.82 @0.649 | 0.936 | 0.73/0.93/0.82 @0.561 |
| bge-base/lemma+gloss+ex | bge-base-en-v1.5 | float32 | 93.7 | 0.856 | 0.60/0.95/0.74 @0.717 | 0.898 | 0.71/0.94/0.81 @0.717 |
| bge-base/parent+gloss+ex | bge-base-en-v1.5 | float32 | 93.7 | 0.901 | 0.77/0.86/0.81 @0.649 | 0.932 | 0.73/0.93/0.82 @0.561 |
| bge-base/gloss+ex/bf16 | bge-base-en-v1.5 | bfloat16 | 246.5 | 0.902 | 0.77/0.86/0.81 @0.649 | 0.936 | 0.73/0.93/0.82 @0.562 |
| bge-large/gloss+ex | bge-large-en-v1.5 | float32 | 25.8 | 0.910 | 0.79/0.82/0.81 @0.646 | 0.941 | 0.87/0.84/0.85 @0.628 |
| bge-large/gloss+ex/bf16 | bge-large-en-v1.5 | bfloat16 | 87.8 | 0.910 | 0.79/0.82/0.81 @0.646 | 0.940 | 0.91/0.78/0.84 @0.646 |
| e5-large-v2/gloss+ex | e5-large-v2 | float32 | 25.2 | 0.894 | 0.76/0.83/0.80 @0.817 | 0.929 | 0.78/0.96/0.86 @0.793 |
| mpnet-base/gloss+ex | all-mpnet-base-v2 | float32 | 91.6 | 0.915 | 0.65/0.97/0.78 @0.341 | 0.933 | 0.78/0.95/0.85 @0.338 |
| bge-base/gloss/bf16 | bge-base-en-v1.5 | bfloat16 | 246.5 | 0.920 | 0.81/0.88/0.84 @0.676 | 0.935 | 0.80/0.95/0.87 @0.588 |
| bge-large/gloss/bf16 | bge-large-en-v1.5 | bfloat16 | 87.8 | 0.928 | 0.77/0.92/0.84 @0.645 | 0.937 | 0.75/0.94/0.83 @0.574 |
| e5-large-v2/gloss | e5-large-v2 | float32 | 25.2 | 0.924 | 0.77/0.97/0.86 @0.809 | 0.928 | 0.89/0.86/0.87 @0.817 |
| mpnet-base/gloss | all-mpnet-base-v2 | float32 | 91.6 | 0.918 | 0.72/0.95/0.82 @0.354 | 0.918 | 0.80/0.90/0.85 @0.342 |
| bge-base/parent+gloss/bf16 | bge-base-en-v1.5 | bfloat16 | 246.5 | 0.924 | 0.82/0.88/0.85 @0.676 | 0.935 | 0.87/0.82/0.85 @0.657 |
