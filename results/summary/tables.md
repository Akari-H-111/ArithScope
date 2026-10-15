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

**E1 NLL, P = 12** (nats per scored position; unseen positions for N > 64; mean ± 95% CI over seeds. NLL is recorded for every run but was not a preregistered decision metric):

| variant | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|---|---|---|
| NoPE | 1.839 ± 0.000 | 1.694 ± 0.001 | 1.667 ± 0.002 | 1.664 ± 0.002 | 1.642 ± 0.003 | 1.607 ± 0.004 | 1.582 ± 0.005 | 1.598 ± 0.006 |
| ALiBi | 1.812 ± 0.002 | 1.655 ± 0.003 | 1.635 ± 0.004 | 1.636 ± 0.004 | 1.617 ± 0.005 | 1.596 ± 0.005 | 1.582 ± 0.010 | 1.614 ± 0.011 |
| RoPE | 1.100 ± 0.009 | 1.202 ± 0.216 | 1.773 ± 0.281 | 2.175 ± 0.299 | 2.174 ± 0.280 | 2.042 ± 0.304 | 2.058 ± 0.385 | 2.077 ± 0.289 |
| ArithScope only | 1.079 ± 0.004 | 0.205 ± 0.005 | 0.119 ± 0.016 | 0.083 ± 0.023 | 0.072 ± 0.028 | 0.068 ± 0.029 | 0.066 ± 0.030 | 0.064 ± 0.032 |
| RoPE + ArithScope | 1.073 ± 0.001 | 0.210 ± 0.008 | 0.126 ± 0.016 | 0.085 ± 0.031 | 0.092 ± 0.049 | 0.149 ± 0.110 | 0.177 ± 0.093 | 0.219 ± 0.112 |
| RoPE + matched cosine bank | 1.073 ± 0.003 | 0.213 ± 0.008 | 0.157 ± 0.052 | 0.142 ± 0.105 | 0.142 ± 0.130 | 0.184 ± 0.163 | 0.206 ± 0.168 | 0.250 ± 0.109 |
| RoPE + full Fourier bank | 1.073 ± 0.003 | 0.209 ± 0.004 | 0.140 ± 0.041 | 0.092 ± 0.045 | 0.098 ± 0.067 | 0.158 ± 0.138 | 0.219 ± 0.196 | 0.254 ± 0.164 |
| RoPE + learned frequencies | 1.071 ± 0.003 | 0.208 ± 0.007 | 0.118 ± 0.016 | 0.084 ± 0.025 | 0.101 ± 0.062 | 0.185 ± 0.156 | 0.324 ± 0.206 | 0.658 ± 0.335 |
| RoPE + relative bias (clipped) | 1.074 ± 0.002 | 0.377 ± 0.003 | 0.509 ± 0.021 | 0.892 ± 0.051 | 1.324 ± 0.091 | 1.681 ± 0.081 | 1.953 ± 0.074 | 2.224 ± 0.123 |

Paired NLL difference, RoPE + ArithScope minus ArithScope only (positive: RoPE + ArithScope is worse): N = 1024: +0.019 [-0.026, +0.064] (4/5 seeds higher); N = 2048: +0.081 [-0.015, +0.177] (4/5 seeds higher); N = 4096: +0.111 [+0.031, +0.191] (5/5 seeds higher); N = 8192: +0.155 [+0.055, +0.256] (5/5 seeds higher).

**E1 NLL, P = 13** (nats per scored position; unseen positions for N > 64; mean ± 95% CI over seeds. NLL is recorded for every run but was not a preregistered decision metric):

| variant | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|---|---|---|---|
| NoPE | 1.862 ± 0.000 | 1.727 ± 0.001 | 1.708 ± 0.001 | 1.685 ± 0.003 | 1.690 ± 0.003 | 1.627 ± 0.003 | 1.671 ± 0.004 | 1.584 ± 0.008 |
| ALiBi | 1.840 ± 0.001 | 1.694 ± 0.001 | 1.683 ± 0.001 | 1.665 ± 0.002 | 1.676 ± 0.002 | 1.624 ± 0.002 | 1.669 ± 0.003 | 1.596 ± 0.007 |
| RoPE | 1.165 ± 0.017 | 1.300 ± 0.408 | 1.967 ± 0.600 | 2.120 ± 0.644 | 2.118 ± 0.685 | 1.840 ± 0.554 | 1.853 ± 0.432 | 1.996 ± 0.498 |
| ArithScope only | 1.147 ± 0.001 | 0.245 ± 0.012 | 0.137 ± 0.025 | 0.090 ± 0.030 | 0.074 ± 0.036 | 0.075 ± 0.040 | 0.066 ± 0.041 | 0.067 ± 0.042 |
| RoPE + ArithScope | 1.140 ± 0.002 | 0.246 ± 0.005 | 0.149 ± 0.026 | 0.106 ± 0.035 | 0.098 ± 0.045 | 0.110 ± 0.055 | 0.153 ± 0.050 | 0.220 ± 0.087 |
| RoPE + matched cosine bank | 1.143 ± 0.002 | 0.977 ± 0.375 | 1.332 ± 0.560 | 1.450 ± 0.586 | 1.337 ± 0.538 | 1.265 ± 0.555 | 1.173 ± 0.587 | 0.969 ± 0.397 |
| RoPE + full Fourier bank | 1.139 ± 0.004 | 0.244 ± 0.005 | 0.141 ± 0.024 | 0.090 ± 0.027 | 0.079 ± 0.032 | 0.116 ± 0.050 | 0.166 ± 0.117 | 0.260 ± 0.182 |
| RoPE + learned frequencies | 1.135 ± 0.003 | 0.270 ± 0.020 | 0.167 ± 0.026 | 0.121 ± 0.034 | 0.116 ± 0.044 | 0.175 ± 0.107 | 0.319 ± 0.174 | 0.569 ± 0.153 |
| RoPE + relative bias (clipped) | 1.140 ± 0.002 | 0.501 ± 0.012 | 0.650 ± 0.027 | 0.997 ± 0.076 | 1.424 ± 0.104 | 1.751 ± 0.086 | 2.011 ± 0.092 | 2.103 ± 0.147 |

Paired NLL difference, RoPE + ArithScope minus ArithScope only (positive: RoPE + ArithScope is worse): N = 1024: +0.024 [-0.040, +0.088] (3/5 seeds higher); N = 2048: +0.035 [-0.042, +0.112] (3/5 seeds higher); N = 4096: +0.087 [+0.006, +0.168] (5/5 seeds higher); N = 8192: +0.152 [+0.081, +0.223] (5/5 seeds higher).

**Post-hoc (not preregistered): NLL by query position** (4 fresh sequences of 8,192 tokens per checkpoint, data seed 20261015; mean over 5 seeds). Median, 99th percentile, and "top 1%" (the share of the mean carried by the worst 1% of positions) describe the per-position distribution; "from errors" is the part of the mean contributed by wrong predictions, the rest comes from correct ones:

| P | variant | positions | accuracy | NLL | median | 99th pct. | top 1% | from errors | Bayes NLL |
|---|---|---|---|---|---|---|---|---|---|
| 12 | ArithScope only | 64–1,024 | 99.34% | 0.075 | 0.048 | 0.363 | 25.5% | 0.016 | 0.0207 |
| 12 | ArithScope only | 1,024–2,048 | 100.00% | 0.065 | 0.056 | 0.152 | 2.4% | 0.000 | < 0.0001 |
| 12 | ArithScope only | 2,048–4,096 | 100.00% | 0.067 | 0.059 | 0.155 | 2.3% | 0.000 | < 0.0001 |
| 12 | ArithScope only | 4,096–8,192 | 100.00% | 0.069 | 0.062 | 0.159 | 2.3% | 0.000 | < 0.0001 |
| 12 | RoPE + ArithScope | 64–1,024 | 99.37% | 0.084 | 0.037 | 0.558 | 24.4% | 0.015 | 0.0207 |
| 12 | RoPE + ArithScope | 1,024–2,048 | 100.00% | 0.153 | 0.125 | 0.429 | 3.7% | 0.000 | < 0.0001 |
| 12 | RoPE + ArithScope | 2,048–4,096 | 100.00% | 0.179 | 0.148 | 0.506 | 3.2% | 0.000 | < 0.0001 |
| 12 | RoPE + ArithScope | 4,096–8,192 | 100.00% | 0.233 | 0.205 | 0.555 | 2.6% | 0.000 | < 0.0001 |
| 12 | RoPE + full Fourier bank | 64–1,024 | 99.34% | 0.100 | 0.054 | 0.638 | 24.7% | 0.015 | 0.0207 |
| 12 | RoPE + full Fourier bank | 1,024–2,048 | 100.00% | 0.183 | 0.177 | 0.396 | 2.7% | 0.000 | < 0.0001 |
| 12 | RoPE + full Fourier bank | 2,048–4,096 | 100.00% | 0.249 | 0.255 | 0.497 | 2.5% | 0.000 | < 0.0001 |
| 12 | RoPE + full Fourier bank | 4,096–8,192 | 100.00% | 0.285 | 0.293 | 0.543 | 2.8% | 0.000 | < 0.0001 |
| 13 | ArithScope only | 64–1,024 | 99.53% | 0.081 | 0.058 | 0.400 | 19.2% | 0.010 | 0.0126 |
| 13 | ArithScope only | 1,024–2,048 | 100.00% | 0.073 | 0.066 | 0.162 | 2.3% | 0.000 | < 0.0001 |
| 13 | ArithScope only | 2,048–4,096 | 100.00% | 0.074 | 0.068 | 0.160 | 2.3% | 0.000 | < 0.0001 |
| 13 | ArithScope only | 4,096–8,192 | 100.00% | 0.075 | 0.069 | 0.160 | 2.2% | 0.000 | < 0.0001 |
| 13 | RoPE + ArithScope | 64–1,024 | 99.57% | 0.115 | 0.056 | 0.603 | 13.9% | 0.009 | 0.0126 |
| 13 | RoPE + ArithScope | 1,024–2,048 | 100.00% | 0.140 | 0.092 | 0.507 | 4.2% | 0.000 | < 0.0001 |
| 13 | RoPE + ArithScope | 2,048–4,096 | 100.00% | 0.205 | 0.150 | 0.632 | 3.2% | 0.000 | < 0.0001 |
| 13 | RoPE + ArithScope | 4,096–8,192 | 100.00% | 0.258 | 0.204 | 0.658 | 2.6% | 0.000 | < 0.0001 |
| 13 | RoPE + full Fourier bank | 64–1,024 | 99.56% | 0.072 | 0.034 | 0.600 | 20.6% | 0.009 | 0.0126 |
| 13 | RoPE + full Fourier bank | 1,024–2,048 | 100.00% | 0.125 | 0.093 | 0.427 | 3.7% | 0.000 | < 0.0001 |
| 13 | RoPE + full Fourier bank | 2,048–4,096 | 100.00% | 0.199 | 0.156 | 0.555 | 3.3% | 0.000 | < 0.0001 |
| 13 | RoPE + full Fourier bank | 4,096–8,192 | 100.00% | 0.254 | 0.222 | 0.617 | 2.8% | 0.000 | < 0.0001 |

Calibration over positions t ≥ 64 (top-label ECE with 15 bins, Brier score; mean over seeds): P = 12, ArithScope only: ECE 0.063, Brier 0.0075; P = 12, RoPE + ArithScope: ECE 0.162, Brier 0.0467; P = 12, RoPE + full Fourier bank: ECE 0.196, Brier 0.0677; P = 13, ArithScope only: ECE 0.070, Brier 0.0085; P = 13, RoPE + ArithScope: ECE 0.177, Brier 0.0552; P = 13, RoPE + full Fourier bank: ECE 0.169, Brier 0.0517.

Readout limit (`experiments/posthoc_readout.py`, weights only): the lowest NLL that the final LayerNorm and linear readout can produce for the hardest target symbol, searched over all LayerNorm outputs (mean over seeds): P = 12, ArithScope only: 0.0053; P = 12, RoPE + ArithScope: 0.0057; P = 12, RoPE + full Fourier bank: 0.0057; P = 13, ArithScope only: 0.0056; P = 13, RoPE + ArithScope: 0.0073; P = 13, RoPE + full Fourier bank: 0.0066.

**Post-hoc: limiting attention to the last 1,024 keys** (same sequences; positions below 1,024 are unaffected by construction). NLL, then accuracy:

| P | variant | positions | NLL, no limit | all layers | layer 0 only | layer 1 only | accuracy, no limit | all layers |
|---|---|---|---|---|---|---|---|---|
| 12 | ArithScope only | 1,024–2,048 | 0.065 | 0.066 | 0.067 | 0.064 | 100.00% | 100.00% |
| 12 | ArithScope only | 2,048–4,096 | 0.067 | 0.066 | 0.070 | 0.063 | 100.00% | 100.00% |
| 12 | ArithScope only | 4,096–8,192 | 0.069 | 0.067 | 0.073 | 0.064 | 100.00% | 100.00% |
| 12 | RoPE + ArithScope | 1,024–2,048 | 0.153 | 0.142 | 0.153 | 0.142 | 100.00% | 100.00% |
| 12 | RoPE + ArithScope | 2,048–4,096 | 0.179 | 0.146 | 0.181 | 0.144 | 100.00% | 100.00% |
| 12 | RoPE + ArithScope | 4,096–8,192 | 0.233 | 0.146 | 0.235 | 0.145 | 100.00% | 100.00% |
| 12 | RoPE + full Fourier bank | 1,024–2,048 | 0.183 | 0.158 | 0.183 | 0.159 | 100.00% | 100.00% |
| 12 | RoPE + full Fourier bank | 2,048–4,096 | 0.249 | 0.162 | 0.251 | 0.160 | 100.00% | 100.00% |
| 12 | RoPE + full Fourier bank | 4,096–8,192 | 0.285 | 0.161 | 0.286 | 0.160 | 100.00% | 100.00% |
| 13 | ArithScope only | 1,024–2,048 | 0.073 | 0.076 | 0.075 | 0.074 | 100.00% | 100.00% |
| 13 | ArithScope only | 2,048–4,096 | 0.074 | 0.076 | 0.077 | 0.074 | 100.00% | 100.00% |
| 13 | ArithScope only | 4,096–8,192 | 0.075 | 0.075 | 0.078 | 0.073 | 100.00% | 100.00% |
| 13 | RoPE + ArithScope | 1,024–2,048 | 0.140 | 0.136 | 0.141 | 0.135 | 100.00% | 100.00% |
| 13 | RoPE + ArithScope | 2,048–4,096 | 0.205 | 0.136 | 0.208 | 0.133 | 100.00% | 100.00% |
| 13 | RoPE + ArithScope | 4,096–8,192 | 0.258 | 0.137 | 0.260 | 0.134 | 100.00% | 100.00% |
| 13 | RoPE + full Fourier bank | 1,024–2,048 | 0.125 | 0.097 | 0.126 | 0.097 | 100.00% | 100.00% |
| 13 | RoPE + full Fourier bank | 2,048–4,096 | 0.199 | 0.099 | 0.202 | 0.096 | 100.00% | 100.00% |
| 13 | RoPE + full Fourier bank | 4,096–8,192 | 0.254 | 0.099 | 0.255 | 0.096 | 100.00% | 100.00% |

**Post-hoc: change of NLL with query position, per seed** (seeds in order; mean NLL at positions 4,096–8,192 minus positions 1,024–2,048; keys limited to the last 1,024):

| P | variant | no limit | all layers limited | layer 0 only | layer 1 only |
|---|---|---|---|---|---|
| 12 | ArithScope only | +0.003, +0.006, +0.004, +0.001, +0.006 | +0.001, +0.004, +0.000, +0.000, +0.002 | +0.004, +0.011, +0.006, +0.002, +0.009 | -0.000, -0.000, -0.001, -0.001, -0.000 |
| 12 | RoPE + ArithScope | +0.043, -0.079, +0.019, +0.232, +0.186 | +0.001, +0.007, +0.007, -0.000, +0.001 | +0.045, -0.080, +0.022, +0.230, +0.190 | -0.001, +0.011, +0.006, -0.000, -0.001 |
| 12 | RoPE + full Fourier bank | +0.087, +0.050, +0.165, +0.159, +0.046 | -0.001, +0.000, +0.001, +0.002, +0.011 | +0.088, +0.050, +0.169, +0.160, +0.048 | -0.001, -0.000, -0.001, -0.001, +0.008 |
| 13 | ArithScope only | +0.003, +0.004, -0.000, +0.006, +0.000 | -0.006, +0.002, -0.001, +0.000, +0.000 | +0.003, +0.006, +0.000, +0.006, +0.001 | -0.004, -0.001, -0.002, +0.001, -0.000 |
| 13 | RoPE + ArithScope | +0.147, +0.205, +0.100, +0.074, +0.060 | -0.000, -0.002, -0.004, +0.010, +0.000 | +0.147, +0.210, +0.101, +0.075, +0.061 | -0.003, -0.001, -0.003, +0.006, -0.004 |
| 13 | RoPE + full Fourier bank | +0.280, +0.051, +0.156, +0.081, +0.078 | +0.007, -0.000, -0.000, +0.004, -0.003 | +0.285, +0.047, +0.157, +0.082, +0.074 | +0.001, -0.001, -0.001, +0.002, -0.005 |

Paired NLL difference, RoPE + ArithScope minus ArithScope only, on the same sequences (seed-paired mean with 95% CI; then each of the 4 test sequences, averaged over seeds): P = 12, 1,024–2,048: +0.088 [-0.033, +0.208], sequences +0.106, +0.094, +0.074, +0.078; P = 12, 2,048–4,096: +0.112 [+0.007, +0.218], sequences +0.115, +0.130, +0.092, +0.111; P = 12, 4,096–8,192: +0.164 [+0.030, +0.298], sequences +0.165, +0.183, +0.143, +0.165; P = 13, 1,024–2,048: +0.068 [-0.054, +0.189], sequences +0.038, +0.022, +0.156, +0.054; P = 13, 2,048–4,096: +0.131 [+0.070, +0.192], sequences +0.093, +0.077, +0.232, +0.122; P = 13, 4,096–8,192: +0.182 [+0.121, +0.244], sequences +0.133, +0.141, +0.272, +0.183.

Rank correlation of per-position NLL with the true symbol's vote share on the comb, positions 64–1,024 / 4,096–8,192 (an attention average sees vote shares; the Bayes posterior depends on vote counts): P = 12, ArithScope only: ρ = -0.46 / -0.14; P = 12, RoPE + ArithScope: ρ = -0.27 / -0.10; P = 12, RoPE + full Fourier bank: ρ = -0.29 / -0.12; P = 13, ArithScope only: ρ = -0.50 / -0.25; P = 13, RoPE + ArithScope: ρ = -0.26 / -0.11; P = 13, RoPE + full Fourier bank: ρ = -0.41 / -0.11. The rank correlation with the Bayes NLL, defined only where the Bayes NLL is not exactly 0, is -0.07, +0.01, -0.02, +0.05, +0.01, +0.07 in the same order at positions 64–1,024.

Positions whose own token is corrupted make up 47% of positions 4,096–8,192. Their share of the NLL that the window removes (shown where the window removes more than 0.01 nats): P = 12, RoPE + ArithScope: 47%; P = 12, RoPE + full Fourier bank: 47%; P = 13, RoPE + ArithScope: 47%; P = 13, RoPE + full Fourier bank: 47%.

**Post-hoc: attention off the comb** (attention mass on keys whose offset is not a multiple of P, averaged over heads and the query positions 1,087, 1,599, …, 7,743) and the largest learned gate |γ| per layer:

| P | variant | layer | off the comb, all keys | off the comb, beyond the last 1,024 keys | max \|γ\| |
|---|---|---|---|---|---|
| 12 | ArithScope only | 0 | 1.3% | 0.9% | 7.35 |
| 12 | ArithScope only | 1 | 85.5% | 58.2% | 5.41 |
| 12 | RoPE + ArithScope | 0 | 0.2% | 0.2% | 8.35 |
| 12 | RoPE + ArithScope | 1 | 73.5% | 47.6% | 3.13 |
| 12 | RoPE + full Fourier bank | 0 | 0.2% | 0.2% | 8.41 |
| 12 | RoPE + full Fourier bank | 1 | 67.7% | 42.6% | 3.82 |
| 13 | ArithScope only | 0 | 0.9% | 0.6% | 7.55 |
| 13 | ArithScope only | 1 | 89.6% | 60.7% | 7.46 |
| 13 | RoPE + ArithScope | 0 | 0.2% | 0.2% | 8.62 |
| 13 | RoPE + ArithScope | 1 | 68.8% | 43.4% | 4.21 |
| 13 | RoPE + full Fourier bank | 0 | 0.3% | 0.2% | 8.42 |
| 13 | RoPE + full Fourier bank | 1 | 74.1% | 46.6% | 3.80 |

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

**Post-hoc sensitivity to multiplicity** (the preregistration used unadjusted 95% intervals; C3 uses fixed thresholds and is not included). One-sided paired t tests in the claimed direction:

| comparison | mean difference | one-sided p | Bonferroni 98.75% interval | holds after Bonferroni | holds after Holm |
|---|---|---|---|---|---|
| C1, P = 12 | +60.02 pp | 6.3e-05 | [+42.38, +77.66] pp | yes | yes |
| C1, P = 13 | +63.89 pp | 5.4e-04 | [+31.21, +96.56] pp | yes | yes |
| C2, P = 13 | +40.20 pp | 5.2e-03 | [+2.17, +78.24] pp | yes | yes |
| C4 (margin −1 pp) | -0.01 pp | 2.3e-06 | [-0.14, +0.12] pp | yes | yes |

**Replays with the current code** (`experiments/replay.py`):

- E1 checkpoints re-scored on the preregistered test sets at N = 1,024 and 8,192: 180 of 180 bitwise identical to the committed records.
- confirmatory runs retrained from scratch: 11 of 11 bitwise identical to the committed records.

**E6, long-length calibration (preregistered, Section 9).** N = 8,192; far region = positions 4,096–8,191; growth = NLL on positions 7,168–8,191 minus NLL on positions 1,024–2,047. Mean ± 95% CI half-width over 5 seeds.

| P | arm | far NLL (nats) | far accuracy (%) | growth (nats) |
|---|---|---|---|---|
| 12 | ArithScope only | 0.050 ± 0.023 | 100.0 ± 0.0 | 0.000 ± 0.003 |
| 12 | RoPE + ArithScope | 0.231 ± 0.113 | 100.0 ± 0.0 | 0.107 ± 0.058 |
| 12 | RoPE + ArithScope, 2nd layer limited to 64 keys | 0.009 ± 0.002 | 100.0 ± 0.0 | -0.001 ± 0.000 |
| 12 | RoPE + ArithScope, trained at length 256 | 0.022 ± 0.010 | 100.0 ± 0.0 | 0.010 ± 0.009 |
| 13 | ArithScope only | 0.059 ± 0.065 | 100.0 ± 0.0 | 0.003 ± 0.008 |
| 13 | RoPE + ArithScope | 0.354 ± 0.139 | 100.0 ± 0.0 | 0.168 ± 0.040 |
| 13 | RoPE + ArithScope, 2nd layer limited to 64 keys | 0.012 ± 0.001 | 100.0 ± 0.0 | -0.001 ± 0.000 |
| 13 | RoPE + ArithScope, trained at length 256 | 0.020 ± 0.009 | 100.0 ± 0.0 | 0.010 ± 0.008 |

Paired differences against RoPE + ArithScope (mean [95% CI], seeds lower / 5):

- P = 12, RoPE + ArithScope, 2nd layer limited to 64 keys: far NLL -0.222 [-0.335, -0.109] (5/5); far accuracy +0.00 pp [+0.00, +0.00]; growth -0.108 [-0.166, -0.050]; far NLL minus ArithScope only -0.041 [-0.065, -0.017].
- P = 12, RoPE + ArithScope, trained at length 256: far NLL -0.209 [-0.326, -0.093] (5/5); far accuracy +0.00 pp [+0.00, +0.00]; growth -0.097 [-0.160, -0.034]; far NLL minus ArithScope only -0.028 [-0.049, -0.007].
- P = 13, RoPE + ArithScope, 2nd layer limited to 64 keys: far NLL -0.342 [-0.480, -0.204] (5/5); far accuracy +0.00 pp [+0.00, +0.00]; growth -0.169 [-0.208, -0.129]; far NLL minus ArithScope only -0.047 [-0.112, +0.018].
- P = 13, RoPE + ArithScope, trained at length 256: far NLL -0.334 [-0.471, -0.198] (5/5); far accuracy +0.00 pp [+0.00, +0.00]; growth -0.158 [-0.202, -0.113]; far NLL minus ArithScope only -0.039 [-0.104, +0.025].

Holm-adjusted one-sided p-values for the four far-NLL comparisons (sensitivity only): long_train, P = 12: 0.0055; long_train, P = 13: 0.0046; window, P = 12: 0.0055; window, P = 13: 0.0046.

**E7, the NLL floor of ArithScope alone (preregistered, Section 10).** N = 8,192; far region = positions 4,096–8,191; growth = NLL on positions 7,168–8,191 minus NLL on positions 1,024–2,047. Mean ± 95% CI half-width over 5 seeds.

| P | arm | far NLL (nats) | far accuracy (%) | growth (nats) |
|---|---|---|---|---|
| 12 | ArithScope only | 0.0713 ± 0.0666 | 100.0 ± 0.0 | 0.0013 ± 0.0051 |
| 12 | ArithScope only, 2nd layer limited to 64 keys | 0.0135 ± 0.0042 | 100.0 ± 0.0 | -0.0010 ± 0.0006 |
| 12 | RoPE + ArithScope, 2nd layer limited to 64 keys | 0.0089 ± 0.0026 | 100.0 ± 0.0 | -0.0008 ± 0.0001 |
| 13 | ArithScope only | 0.0657 ± 0.0383 | 100.0 ± 0.0 | 0.0035 ± 0.0031 |
| 13 | ArithScope only, 2nd layer limited to 64 keys | 0.0144 ± 0.0024 | 100.0 ± 0.0 | -0.0004 ± 0.0003 |
| 13 | RoPE + ArithScope, 2nd layer limited to 64 keys | 0.0115 ± 0.0009 | 100.0 ± 0.0 | -0.0005 ± 0.0002 |

Paired differences in far NLL (mean [95% CI], seeds lower / 5):

- P = 12: ArithScope only, limited minus unlimited: -0.0578 [-0.1205, +0.0049] (5/5); far accuracy +0.00 pp [+0.00, +0.00]. ArithScope only limited minus RoPE + ArithScope limited: +0.0046 [-0.0009, +0.0100] (1/5). RoPE + ArithScope limited minus ArithScope only: -0.0624 [-0.1300, +0.0053] (5/5). Share of the gap between ArithScope only and RoPE + ArithScope limited that the limit closes on ArithScope only (from means, descriptive): 93%.
- P = 13: ArithScope only, limited minus unlimited: -0.0512 [-0.0885, -0.0139] (5/5); far accuracy +0.00 pp [+0.00, +0.00]. ArithScope only limited minus RoPE + ArithScope limited: +0.0029 [+0.0001, +0.0057] (0/5). RoPE + ArithScope limited minus ArithScope only: -0.0541 [-0.0924, -0.0158] (5/5). Share of the gap between ArithScope only and RoPE + ArithScope limited that the limit closes on ArithScope only (from means, descriptive): 95%.

Holm-adjusted one-sided p-values for W1's two far-NLL comparisons (sensitivity only): P = 12: 0.0314; P = 13: 0.0189.

**E8, a powered retest of E7's W1 (preregistered, Section 11).** N = 8,192; far region = positions 4,096–8,191; growth = NLL on positions 7,168–8,191 minus NLL on positions 1,024–2,047. Mean ± 95% CI half-width over 21 seeds (seeds 6–26; 0 failed runs). Seed-to-seed SD of far NLL in the last column.

| P | arm | far NLL (nats) | far accuracy (%) | growth (nats) | SD of far NLL |
|---|---|---|---|---|---|
| 12 | ArithScope only | 0.0636 ± 0.0146 | 100.0 ± 0.0 | 0.0040 ± 0.0011 | 0.0321 |
| 12 | ArithScope only, 2nd layer limited to 64 keys | 0.0126 ± 0.0010 | 100.0 ± 0.0 | -0.0004 ± 0.0001 | 0.0022 |
| 13 | ArithScope only | 0.0618 ± 0.0191 | 100.0 ± 0.0 | 0.0027 ± 0.0015 | 0.0419 |
| 13 | ArithScope only, 2nd layer limited to 64 keys | 0.0193 ± 0.0018 | 100.0 ± 0.0 | -0.0005 ± 0.0002 | 0.0039 |

Paired differences, limited minus unlimited (mean [95% CI], seeds lower / seeds):

- P = 12: far NLL -0.0510 [-0.0656, -0.0363] (21/21); far accuracy +0.00 pp [+0.00, +0.00]; growth -0.0043 [-0.0054, -0.0033] (21/21).
- P = 13: far NLL -0.0425 [-0.0606, -0.0244] (21/21); far accuracy +0.00 pp [+0.00, +0.00]; growth -0.0032 [-0.0047, -0.0017] (21/21).

Holm-adjusted one-sided p-values for W2's two far-NLL comparisons (sensitivity only): P = 12: 5e-07; P = 13: 4.4e-05.
