# Validation report

Produced by `python scripts/run_validation.py` (seed 0). Base case: S=K=100, T=1, r=5%, sigma=20% unless stated.

## 1. Black-Scholes closed form

| check | result |
|---|---|
| Textbook call / put (S=42, K=40, T=0.5, r=10%, sigma=20%) | 4.7594 / 0.8086 (book: 4.76 / 0.81) |
| Max abs gap vs numerical integration of the risk-neutral expectation (60 random parameter sets, calls and puts, with dividends) | 1.4e-13 |
| Max put-call parity gap | 1.4e-14 |

## 2. Binomial tree (Cox-Ross-Rubinstein)

| check | result |
|---|---|
| European call error at N=10 / 160 / 2560 steps | 1.97e-01 / 1.25e-02 / 7.81e-04 |
| Log-log convergence slope (raw / averaging N and N+1) | -1.00 / -1.06 (first order = -1) |
| Error at N=2560 after averaging N and N+1 | 4.8e-05 |
| Max put-call parity gap inside the tree | 2.8e-12 |
| Hull two-step example, European / American put | 4.1927 / 5.0896 (book: 4.1923 / 5.0894; book rounds p to 4 d.p.) |
| American minus European call, no dividends | 0.0e+00 |
| American call premium, dividend yield 8% (K=80) | 2.353 |
| American put, N = 250 / 1000 / 4000 | 6.0872 / 6.0896 / 6.0902 |
| Early-exercise premium of the ATM put (N=4000) | 0.5172 |

## 3. Monte Carlo (500 replications x 20,000 paths, ATM call, exact price 10.4506)

| variant | 95% CI coverage | RMSE | variance reduction vs plain |
|---|---|---|---|
| plain | 94.6% | 0.1040 | 1.0x |
| antithetic | 94.4% | 0.0751 | 1.9x |
| control variate | 94.0% | 0.0402 | 6.6x |
| antithetic + control variate | 96.4% | 0.0186 | 31.0x |

## 4. Greeks

Scaled error = |numerical - analytic| / max(|analytic|, 0.01).

| method | delta | gamma | vega | theta | rho |
|---|---|---|---|---|---|
| Finite difference on closed form, worst over 40 random sets x call/put | 2.5e-04 | 4.2e-05 | 2.7e-06 | 7.5e-07 | 2.5e-07 |
| CRR tree Greeks, N=250 (relative error, base case) | 2.0e-04 | 3.4e-03 | - | 1.9e-03 | - |
| CRR tree Greeks, N=1000 (relative error, base case) | 5.0e-05 | 8.5e-04 | - | 4.8e-04 | - |
| CRR tree Greeks, N=4000 (relative error, base case) | 1.3e-05 | 2.1e-04 | - | 1.2e-04 | - |
| Monte Carlo bump-and-revalue, common random numbers, 400k paths | 1.1e-04 | - | 9.3e-04 | - | - |

Gamma by finite difference is U-shaped in the bump size: relative error 5.1e-01 at a bump of 1e-08 x S (round-off), minimum 8.6e-10 at 3e-05 x S, 1.2e-02 at 1e-01 x S (truncation).
