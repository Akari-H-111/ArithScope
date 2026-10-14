**E1 periodic denoising, P = 12.** Accuracy (%) as mean ± 95% CI half-width over seeds. For N > 64 only positions t ≥ 64 (never seen in training) are scored.

| variant | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|---|---|---|
| NoPE | 53.5 ± 0.0 | 54.9 ± 0.1 | 55.1 ± 0.1 | 55.0 ± 0.1 | 55.4 ± 0.2 | 56.1 ± 0.4 | 57.0 ± 0.6 | 57.0 ± 0.8 |
| ALiBi | 53.6 ± 0.0 | 55.3 ± 0.1 | 55.4 ± 0.1 | 55.1 ± 0.0 | 55.5 ± 0.1 | 55.9 ± 0.1 | 55.9 ± 0.1 | 55.3 ± 0.1 |
| RoPE | 67.9 ± 0.2 | 66.1 ± 7.2 | 49.9 ± 9.1 | 39.5 ± 10.1 | 39.4 ± 11.4 | 42.5 ± 12.4 | 43.1 ± 14.7 | 42.3 ± 15.1 |
| ArithScope only | 68.1 ± 0.1 | 94.1 ± 0.1 | 97.6 ± 0.1 | 99.0 ± 0.0 | 99.4 ± 0.0 | 99.8 ± 0.0 | 99.9 ± 0.0 | 100.0 ± 0.0 |
| RoPE + ArithScope | 68.1 ± 0.1 | 94.1 ± 0.1 | 97.6 ± 0.1 | 99.0 ± 0.0 | 99.5 ± 0.0 | 99.8 ± 0.0 | 99.9 ± 0.0 | 100.0 ± 0.0 |
| RoPE + matched cosine bank | 68.2 ± 0.2 | 94.1 ± 0.1 | 97.5 ± 0.1 | 99.0 ± 0.0 | 99.4 ± 0.0 | 99.8 ± 0.0 | 99.9 ± 0.0 | 100.0 ± 0.0 |
| RoPE + full Fourier bank | 68.1 ± 0.1 | 94.1 ± 0.1 | 97.6 ± 0.1 | 99.0 ± 0.0 | 99.4 ± 0.0 | 99.8 ± 0.0 | 99.9 ± 0.0 | 100.0 ± 0.0 |
| RoPE + learned frequencies | 68.1 ± 0.1 | 94.1 ± 0.1 | 97.5 ± 0.1 | 99.0 ± 0.1 | 99.4 ± 0.1 | 99.7 ± 0.1 | 99.6 ± 0.4 | 97.5 ± 3.9 |
| RoPE + relative bias (clipped) | 68.1 ± 0.2 | 90.2 ± 0.1 | 89.2 ± 0.2 | 82.3 ± 1.2 | 69.4 ± 2.0 | 56.3 ± 1.9 | 43.0 ± 4.2 | 37.0 ± 6.9 |
| *Bayes oracle* | 68.1 | 94.1 | 97.5 | 99.0 | 99.4 | 99.8 | 99.9 | 99.9 |

**E1 paired differences, P = 12** (RoPE + ArithScope minus control; mean, 95% CI, seeds won):

| control | N=64 | N=256 | N=1024 | N=2048 |
|---|---|---|---|---|
| RoPE | +0.2 pp [-0.0, +0.5] (4/5) | +47.7 pp [+38.5, +56.8] (5/5) | +60.0 pp [+48.7, +71.4] (5/5) | +57.2 pp [+44.9, +69.6] (5/5) |
| RoPE + matched cosine bank | -0.0 pp [-0.1, +0.1] (2/5) | +0.0 pp [-0.0, +0.0] (4/5) | +0.0 pp [-0.0, +0.0] (4/5) | +0.0 pp [-0.0, +0.0] (2/5) |
| RoPE + full Fourier bank | +0.0 pp [-0.0, +0.1] (3/5) | +0.0 pp [-0.0, +0.0] (3/5) | +0.0 pp [-0.0, +0.0] (3/5) | +0.0 pp [-0.0, +0.0] (3/5) |
| RoPE + learned frequencies | +0.1 pp [-0.1, +0.2] (4/5) | +0.1 pp [-0.0, +0.1] (4/5) | +0.0 pp [-0.0, +0.1] (3/5) | +0.1 pp [-0.0, +0.1] (4/5) |
| RoPE + relative bias (clipped) | +0.0 pp [-0.1, +0.2] (2/5) | +8.4 pp [+8.2, +8.6] (5/5) | +30.0 pp [+28.1, +32.0] (5/5) | +43.5 pp [+41.6, +45.4] (5/5) |

**E1 periodic denoising, P = 13.** Accuracy (%) as mean ± 95% CI half-width over seeds. For N > 64 only positions t ≥ 64 (never seen in training) are scored.

| variant | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|---|---|---|
| NoPE | 53.3 ± 0.0 | 54.4 ± 0.0 | 54.1 ± 0.1 | 54.6 ± 0.1 | 54.4 ± 0.1 | 55.2 ± 0.2 | 55.3 ± 0.2 | 56.6 ± 1.1 |
| ALiBi | 53.4 ± 0.0 | 54.6 ± 0.0 | 54.5 ± 0.1 | 54.7 ± 0.1 | 54.5 ± 0.1 | 55.1 ± 0.1 | 54.6 ± 0.2 | 55.8 ± 0.4 |
| RoPE | 66.2 ± 0.3 | 61.5 ± 12.8 | 41.8 ± 18.4 | 36.3 ± 19.7 | 35.6 ± 21.0 | 43.2 ± 17.1 | 42.9 ± 13.1 | 38.4 ± 16.0 |
| ArithScope only | 66.3 ± 0.2 | 93.2 ± 0.1 | 97.2 ± 0.0 | 98.9 ± 0.0 | 99.5 ± 0.0 | 99.7 ± 0.0 | 99.8 ± 0.0 | 99.9 ± 0.0 |
| RoPE + ArithScope | 66.2 ± 0.2 | 93.2 ± 0.1 | 97.2 ± 0.0 | 98.9 ± 0.0 | 99.5 ± 0.0 | 99.7 ± 0.0 | 99.8 ± 0.0 | 99.9 ± 0.0 |
| RoPE + matched cosine bank | 66.2 ± 0.1 | 69.0 ± 13.2 | 58.2 ± 18.2 | 55.9 ± 21.0 | 59.3 ± 24.5 | 61.6 ± 28.8 | 65.7 ± 28.8 | 75.2 ± 21.9 |
| RoPE + full Fourier bank | 66.3 ± 0.2 | 93.1 ± 0.1 | 97.2 ± 0.0 | 98.9 ± 0.0 | 99.5 ± 0.0 | 99.7 ± 0.0 | 99.8 ± 0.0 | 99.9 ± 0.0 |
| RoPE + learned frequencies | 66.3 ± 0.2 | 92.9 ± 0.4 | 96.6 ± 0.4 | 98.4 ± 0.3 | 99.2 ± 0.2 | 99.5 ± 0.3 | 98.8 ± 1.5 | 93.3 ± 7.8 |
| RoPE + relative bias (clipped) | 66.3 ± 0.1 | 86.6 ± 0.1 | 85.3 ± 0.4 | 78.0 ± 1.4 | 65.5 ± 2.0 | 54.0 ± 2.6 | 40.1 ± 2.6 | 39.6 ± 3.5 |
| *Bayes oracle* | 66.3 | 93.1 | 97.2 | 98.9 | 99.5 | 99.7 | 99.8 | 99.9 |

**E1 paired differences, P = 13** (RoPE + ArithScope minus control; mean, 95% CI, seeds won):

| control | N=64 | N=256 | N=1024 | N=2048 |
|---|---|---|---|---|
| RoPE | +0.1 pp [-0.2, +0.3] (2/5) | +55.4 pp [+37.0, +73.9] (5/5) | +63.9 pp [+42.9, +84.9] (5/5) | +56.5 pp [+39.4, +73.6] (5/5) |
| RoPE + matched cosine bank | +0.0 pp [-0.2, +0.2] (2/5) | +39.0 pp [+20.8, +57.1] (5/5) | +40.2 pp [+15.7, +64.7] (5/5) | +38.0 pp [+9.3, +66.8] (5/5) |
| RoPE + full Fourier bank | -0.0 pp [-0.1, -0.0] (1/5) | -0.0 pp [-0.0, +0.0] (3/5) | -0.0 pp [-0.0, +0.0] (1/5) | -0.0 pp [-0.0, +0.0] (2/5) |
| RoPE + learned frequencies | -0.0 pp [-0.2, +0.1] (1/5) | +0.6 pp [+0.2, +0.9] (5/5) | +0.3 pp [+0.1, +0.6] (5/5) | +0.2 pp [-0.1, +0.5] (4/5) |
| RoPE + relative bias (clipped) | -0.0 pp [-0.1, +0.0] (2/5) | +11.9 pp [+11.5, +12.2] (5/5) | +34.0 pp [+32.0, +35.9] (5/5) | +45.6 pp [+43.0, +48.2] (5/5) |

**E1 position-shift robustness, P = 13, N = 256** (accuracy %, mean over seeds):

| variant | 0 | 1e+03 | 1e+04 | 1e+05 | 1e+06 | 1e+07 | 1e+08 | 1e+09 |
|---|---|---|---|---|---|---|---|---|
| RoPE + ArithScope | 89.4 | 89.4 | 89.4 | 89.4 | 89.4 | 89.5 | 89.4 | 89.5 |
| ArithScope only | 89.4 | 89.4 | 89.4 | 89.4 | 89.4 | 89.4 | 89.4 | 89.4 |
| RoPE | 47.8 | 47.8 | 47.8 | 47.8 | 47.8 | 47.3 | 43.2 | 51.1 |
| RoPE + matched cosine bank | 60.2 | 60.2 | 60.2 | 60.2 | 60.2 | 60.5 | 37.6 | 56.9 |
| RoPE + relative bias (clipped) | 80.5 | 80.5 | 80.5 | 80.5 | 80.5 | 80.5 | 80.5 | 80.5 |
| ALiBi | 54.2 | 54.2 | 54.2 | 54.2 | 54.2 | 54.2 | 54.2 | 54.2 |

**Post-hoc (not preregistered): absolute-position shift at the training length, P = 12, N = 64** (accuracy %, mean over seeds; RoPE angles in float32):

| variant | 0 | 10^3 | 10^4 | 10^5 | 10^6 | 10^7 | 10^8 | 10^9 |
|---|---|---|---|---|---|---|---|---|
| RoPE + ArithScope | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 |
| ArithScope only | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 |
| RoPE | 67.8 | 67.8 | 67.8 | 67.8 | 67.8 | 67.8 | 50.7 | 52.1 |
| RoPE + matched cosine bank | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 |
| RoPE + full Fourier bank | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 |
| RoPE + learned frequencies | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 |
| RoPE + relative bias (clipped) | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 | 68.1 |
| ALiBi | 53.4 | 53.4 | 53.4 | 53.4 | 53.4 | 53.4 | 53.4 | 53.4 |
| NoPE | 53.3 | 53.3 | 53.3 | 53.3 | 53.3 | 53.3 | 53.3 | 53.3 |

**Post-hoc (not preregistered): absolute-position shift at the training length, P = 13, N = 64** (accuracy %, mean over seeds; RoPE angles in float32):

| variant | 0 | 10^3 | 10^4 | 10^5 | 10^6 | 10^7 | 10^8 | 10^9 |
|---|---|---|---|---|---|---|---|---|
| RoPE + ArithScope | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 |
| ArithScope only | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 |
| RoPE | 66.0 | 66.0 | 66.0 | 66.0 | 66.0 | 65.7 | 51.3 | 52.4 |
| RoPE + matched cosine bank | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.0 | 40.8 | 52.2 |
| RoPE + full Fourier bank | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.2 |
| RoPE + learned frequencies | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 66.2 | 66.2 |
| RoPE + relative bias (clipped) | 66.2 | 66.2 | 66.2 | 66.2 | 66.2 | 66.2 | 66.1 | 66.2 |
| ALiBi | 53.3 | 53.3 | 53.3 | 53.3 | 53.3 | 53.3 | 53.3 | 53.3 |
| NoPE | 53.2 | 53.2 | 53.2 | 53.2 | 53.2 | 53.2 | 53.2 | 53.2 |

**Learned mixtures vs Theorem T3** (layer-0 heads, all seeds; exploratory):

| P | variant | heads | mass on divisors of P | TV distance to optimum | comb margin of mean α (optimum P/(P−1)) |
|---|---|---|---|---|---|
| 12 | ArithScope only | 20 | 1.000 | 0.047 | 1.012 (1.091) |
| 12 | RoPE + ArithScope | 20 | 1.000 | 0.024 | 1.058 (1.091) |
| 13 | ArithScope only | 20 | 1.000 | 0.000 | 1.083 (1.083) |
| 13 | RoPE + ArithScope | 20 | 1.000 | 0.000 | 1.083 (1.083) |

**Matched cosine bank, layer-0 mixtures** (exploratory; mean over heads and seeds). Mass on periods that divide P keeps the bias exactly P-periodic; mass elsewhere approximates the comb only inside the training window:

| P | heads | mass on periods dividing P | largest components |
|---|---|---|---|
| 12 | 20 | 1.000 | cos(2πn/12): 0.26, cos(2πn/3): 0.25, cos(2πn/4): 0.22, cos(2πn/6): 0.14 |
| 13 | 20 | 0.423 | cos(2πn/13): 0.42, cos(2πn/14): 0.33, cos(2πn/4): 0.09, cos(2πn/2): 0.06 |

**Post-hoc (not preregistered): comb selectivity of learned-frequency heads** (S1 mechanism). Comb margin divided by the margin at offset 0, per comb head, averaged over the block bin that contains the offset; median over heads:

| P | comb heads / all heads | n ≈ 64 | n ≈ 1,024 | n ≈ 4,096 | n ≈ 8,192 | comb lost (ratio ≤ 0) near 8,192 |
|---|---|---|---|---|---|---|
| 12 | 19 / 40 | 0.93 | -0.43 | -0.42 | -0.53 | 18 of 19 |
| 13 | 8 / 40 | 0.95 | -0.13 | -0.68 | -0.64 | 7 of 8 |

**E2 alias fixture** (period 12, query tokens, ArithScope only; accuracy % on unseen positions at N = 1024):

| bank | orders | lcm | model | alias oracle | Bayes oracle | preregistered check |
|---|---|---|---|---|---|---|
| {2, 3} | 2 | 6 | 50.5 ± 4.2 | 52.1 | 98.4 | at most alias + 2 pp: yes |
| {1, 2, 3, 6} | 4 | 6 | 51.4 ± 2.7 | 52.1 | 98.4 | at most alias + 2 pp: yes |
| {12} | 1 | 12 | 98.3 ± 0.1 | 98.4 | 98.4 | within 5 pp of Bayes: yes |
| {4, 6} | 2 | 12 | 97.4 ± 2.2 | 98.4 | 98.4 | within 5 pp of Bayes: yes |
| {3, 4} | 2 | 12 | 98.1 ± 0.4 | 98.4 | 98.4 | within 5 pp of Bayes: yes |
| {1, 2, 3, 4, 6, 12} | 6 | 12 | 98.1 ± 0.5 | 98.4 | 98.4 | within 5 pp of Bayes: yes |

**E3 tagged denoising (aperiodic control).** Accuracy (%) as mean ± 95% CI half-width over seeds. For N > 64 only positions t ≥ 64 (never seen in training) are scored.

| variant | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 |
|---|---|---|---|---|---|---|
| NoPE | 69.7 ± 0.1 | 93.0 ± 0.1 | 97.3 ± 0.1 | 98.7 ± 0.1 | 99.5 ± 0.1 | 99.7 ± 0.0 |
| ALiBi | 69.7 ± 0.1 | 93.0 ± 0.1 | 97.2 ± 0.1 | 98.6 ± 0.0 | 99.5 ± 0.0 | 99.7 ± 0.0 |
| RoPE | 69.7 ± 0.1 | 92.9 ± 0.1 | 96.7 ± 0.1 | 98.0 ± 0.1 | 99.1 ± 0.1 | 98.8 ± 0.5 |
| ArithScope only | 69.7 ± 0.1 | 93.0 ± 0.1 | 97.3 ± 0.0 | 98.7 ± 0.0 | 99.5 ± 0.0 | 99.7 ± 0.0 |
| RoPE + ArithScope | 69.7 ± 0.1 | 92.9 ± 0.1 | 96.9 ± 0.1 | 98.1 ± 0.0 | 99.1 ± 0.1 | 98.7 ± 0.3 |
| RoPE + matched cosine bank | 69.7 ± 0.1 | 92.9 ± 0.1 | 96.8 ± 0.1 | 98.1 ± 0.1 | 99.1 ± 0.1 | 98.8 ± 0.2 |
| RoPE + relative bias (clipped) | 69.7 ± 0.1 | 92.4 ± 0.2 | 96.1 ± 0.2 | 97.5 ± 0.2 | 98.5 ± 0.3 | 97.6 ± 1.0 |
| *Bayes oracle* | 69.7 | 93.0 | 97.3 | 98.7 | 99.5 | 99.7 |

*E3 tagged denoising (aperiodic control) — C4 check:* RoPE + ArithScope minus RoPE at N = 64: -0.0 pp [-0.1, +0.1] (3/5) → no harm (CI lower bound > −1 pp).

**E5 periodic denoising, P = 7/11/13.** Accuracy (%) as mean ± 95% CI half-width over seeds. For N > 64 only positions t ≥ 64 (never seen in training) are scored.

| variant | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 |
|---|---|---|---|---|---|---|
| NoPE | 53.7 ± 0.1 | 56.0 ± 0.1 | 56.7 ± 0.0 | 57.6 ± 0.3 | 58.6 ± 0.1 | 58.5 ± 0.2 |
| ALiBi | 54.0 ± 0.0 | 56.5 ± 0.1 | 57.2 ± 0.2 | 57.6 ± 0.1 | 58.3 ± 0.1 | 57.9 ± 0.0 |
| RoPE | 58.4 ± 0.8 | 55.3 ± 10.4 | 47.6 ± 12.4 | 40.0 ± 11.7 | 36.0 ± 9.0 | 30.8 ± 7.1 |
| ArithScope only | 63.1 ± 0.5 | 80.4 ± 1.3 | 80.5 ± 2.1 | 81.7 ± 2.5 | 82.7 ± 2.5 | 87.7 ± 1.3 |
| RoPE + ArithScope | 63.8 ± 0.5 | 84.4 ± 0.9 | 91.1 ± 1.3 | 95.9 ± 0.7 | 98.2 ± 0.3 | 99.2 ± 0.2 |
| RoPE + matched cosine bank | 62.8 ± 1.9 | 71.3 ± 17.0 | 73.3 ± 13.3 | 77.2 ± 9.7 | 79.9 ± 6.7 | 86.1 ± 3.9 |
| RoPE + full Fourier bank | 63.6 ± 0.6 | 81.6 ± 10.1 | 81.6 ± 28.8 | 87.6 ± 28.1 | 92.8 ± 19.4 | 96.1 ± 11.5 |
| RoPE + learned frequencies | 64.2 ± 0.2 | 87.8 ± 0.9 | 92.6 ± 1.0 | 95.5 ± 1.0 | 96.4 ± 0.1 | 96.0 ± 3.7 |
| RoPE + relative bias (clipped) | 63.7 ± 0.1 | 79.2 ± 0.5 | 78.4 ± 0.5 | 76.4 ± 0.8 | 72.5 ± 0.9 | 65.1 ± 2.5 |
| *Bayes oracle* | 71.6 | 96.0 | 98.1 | 99.3 | 99.6 | 99.9 |

**E5 paired differences, P = 7/11/13** (RoPE + ArithScope minus control; mean, 95% CI, seeds won):

| control | N=64 | N=256 | N=1024 | N=2048 |
|---|---|---|---|---|
| RoPE | +5.4 pp [+5.1, +5.8] (3/3) | +43.6 pp [+31.0, +56.2] (3/3) | +62.2 pp [+53.2, +71.2] (3/3) | +68.4 pp [+61.2, +75.5] (3/3) |
| RoPE + matched cosine bank | +1.0 pp [-1.4, +3.4] (2/3) | +17.9 pp [+5.9, +29.9] (3/3) | +18.3 pp [+12.0, +24.7] (3/3) | +13.2 pp [+9.4, +16.9] (3/3) |
| RoPE + full Fourier bank | +0.1 pp [-0.6, +0.9] (2/3) | +9.5 pp [-20.3, +39.3] (2/3) | +5.4 pp [-14.2, +25.0] (2/3) | +3.1 pp [-8.5, +14.8] (2/3) |
| RoPE + learned frequencies | -0.4 pp [-1.1, +0.2] (0/3) | -1.5 pp [-2.4, -0.6] (0/3) | +1.8 pp [+1.5, +2.1] (3/3) | +3.2 pp [-0.5, +7.0] (3/3) |
| RoPE + relative bias (clipped) | +0.1 pp [-0.3, +0.5] (2/3) | +12.8 pp [+11.1, +14.5] (3/3) | +25.7 pp [+24.4, +26.9] (3/3) | +34.1 pp [+31.4, +36.8] (3/3) |

**ArithScope gate magnitudes** (exploratory; largest |γ| over the heads of each layer, all layers and seeds; γ starts at 0):

| suite | variant | layers | median | min | max |
|---|---|---|---|---|---|
| e1_periodic | ArithScope only | 20 | 6.89 | 4.69 | 9.58 |
| e1_periodic | RoPE + ArithScope | 20 | 6.71 | 2.56 | 8.94 |
| e2_alias | ArithScope only | 60 | 6.54 | 0.33 | 10.90 |
| e3_tagged | ArithScope only | 10 | 5.00 | 0.29 | 11.07 |
| e3_tagged | RoPE + ArithScope | 10 | 0.56 | 0.22 | 6.39 |
| e5_mixed | ArithScope only | 6 | 9.34 | 8.70 | 12.29 |
| e5_mixed | RoPE + ArithScope | 6 | 6.75 | 2.74 | 10.27 |

**Matched-baseline metadata** (E1; parameters from the run files, CPU time per training step):

| variant | total params | positional params | ms / step (mean ± sd) |
|---|---|---|---|
| ALiBi | 101,632 | 0 | 55.4 ± 2.5 |
| ArithScope only | 101,768 | 136 | 55.0 ± 2.7 |
| NoPE | 101,632 | 0 | 61.6 ± 3.5 |
| RoPE | 101,632 | 0 | 62.3 ± 4.6 |
| RoPE + ArithScope | 101,768 | 136 | 63.2 ± 2.0 |
| RoPE + full Fourier bank | 101,968 | 336 | 62.9 ± 3.0 |
| RoPE + matched cosine bank | 101,768 | 136 | 62.1 ± 2.5 |
| RoPE + learned frequencies | 101,888 | 256 | 63.1 ± 3.5 |
| RoPE + relative bias (clipped) | 102,664 | 1,032 | 64.2 ± 4.1 |

**Overhead, llm_d64_B16, forward** (median ms; peak RSS increase MB in parentheses; one layer, batch 1, CPU, 4 threads):

| backend | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|
| RoPE only, fused SDPA (floor) | 7.7 (19) | 17.9 (29) | 52.2 (49) | 122.2 (84) | 344.9 (161) |
| ArithScope, explicit logits | 24.4 (60) | 130.8 (153) | 450.7 (555) | 1497.1 (1354) | 6040.8 (2738) |
| ArithScope, SDPA + dense bias mask | 21.1 (47) | 72.4 (120) | 235.5 (426) | 915.6 (1641) | 4321.6 (6348) |
| ArithScope, folded into q/k (fused) | 12.0 (27) | 29.3 (46) | 67.1 (86) | 168.0 (151) | 599.3 (275) |
| ArithScope, compiled FlexAttention (forward only) | 11.8 (201) | 27.6 (206) | 82.5 (217) | 247.6 (256) | 869.3 (340) |

**Overhead, llm_d64_B16, forward+backward** (median ms; peak RSS increase MB in parentheses; one layer, batch 1, CPU, 4 threads):

| backend | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|
| RoPE only, fused SDPA (floor) | 23.6 (42) | 48.5 (55) | 121.2 (94) | 329.9 (194) | 1089.0 (315) |
| ArithScope, explicit logits | 59.8 (91) | 231.3 (195) | 922.6 (629) | 3339.1 (1838) | skipped |
| ArithScope, SDPA + dense bias mask | 43.4 (83) | 224.5 (198) | 756.9 (661) | 2626.4 (2467) | skipped |
| ArithScope, folded into q/k (fused) | 31.4 (50) | 79.9 (87) | 191.1 (152) | 535.0 (255) | 1905.8 (503) |
| ArithScope, compiled FlexAttention (forward only) | skipped | skipped | skipped | skipped | skipped |

**Overhead, tiny_d16_M1-16, forward** (median ms; peak RSS increase MB in parentheses; one layer, batch 1, CPU, 4 threads):

| backend | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|
| RoPE only, fused SDPA (floor) | 1.6 (8) | 3.0 (11) | 7.0 (14) | 20.8 (17) | 55.6 (25) |
| ArithScope, explicit logits | 11.0 (33) | 48.5 (82) | 272.0 (276) | 765.4 (672) | 3277.5 (1350) |
| ArithScope, SDPA + dense bias mask | 8.5 (26) | 27.6 (86) | 141.7 (212) | 506.3 (803) | 1649.4 (3166) |
| ArithScope, folded into q/k (fused) | 6.6 (14) | 12.9 (23) | 23.4 (32) | 54.4 (57) | 169.7 (103) |
| ArithScope, compiled FlexAttention (forward only) | 5.0 (194) | 9.4 (196) | 25.4 (197) | 91.1 (200) | 294.8 (205) |

**Overhead, tiny_d16_M1-16, forward+backward** (median ms; peak RSS increase MB in parentheses; one layer, batch 1, CPU, 4 threads):

| backend | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|
| RoPE only, fused SDPA (floor) | 4.2 (14) | 7.1 (20) | 21.4 (25) | 55.1 (35) | 195.6 (45) |
| ArithScope, explicit logits | 27.8 (45) | 85.6 (142) | 426.7 (317) | 1581.0 (957) | 5708.2 (3248) |
| ArithScope, SDPA + dense bias mask | 18.5 (42) | 84.2 (136) | 334.6 (329) | 1224.3 (1277) | 6222.6 (4976) |
| ArithScope, folded into q/k (fused) | 14.7 (24) | 26.5 (40) | 66.9 (65) | 163.6 (103) | 623.5 (177) |
| ArithScope, compiled FlexAttention (forward only) | skipped | skipped | skipped | skipped | skipped |
