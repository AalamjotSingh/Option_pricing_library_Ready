# Run and explore on Windows

Extract the ZIP first. Open the extracted option-pricing-lab folder in VS Code, then open Terminal > New Terminal (PowerShell). You must be in the folder containing this file.

Use Python 3.11 or 3.12 for this walkthrough. If `py` is not found, install Python with its Windows launcher and reopen your terminal.

## First run

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/play.py
```

Calling the virtual environment's Python directly avoids PowerShell activation-policy issues. Dependencies need an internet connection for installation. The experiments then run locally using simulated prices.

Open results/play/experiment.png and experiment.csv. Each run overwrites this experiment's outputs; copy that folder before running again to keep a comparison. inputs.json records the exact inputs.

## Change one thing at a time

Run each command from the same project folder:

```powershell
# No trading costs
.\.venv\Scripts\python.exe scripts/play.py --cost 0
# A 0.5% fee on traded stock notional
.\.venv\Scripts\python.exe scripts/play.py --cost 0.005
# Higher annual volatility
.\.venv\Scripts\python.exe scripts/play.py --vol 0.40
# Six-month option, out of the money
.\.venv\Scripts\python.exe scripts/play.py --years 0.5 --strike 110
# More simulation paths for a less noisy estimate
.\.venv\Scripts\python.exe scripts/play.py --paths 50000
```

Inputs: --spot and --strike are dollars per share; --years is time until expiry; --rate is a continuously compounded annual rate; --vol is annual volatility; --cost is a fraction of stock notional traded, not of option premium. 0.20 means 20%. --seed controls reproducibility.

The 260 simulation steps span the entire option life: if --years changes, they are no longer daily steps. The table reports hedge intervals; one interval means buying the initial hedge and holding until liquidation at expiry. It does not mean one interior rebalance. Total cost includes the initial purchase and final liquidation.

## What the experiment means

You sell one European call, collect the model premium, buy delta shares, and finance the hedge through a cash account accruing at the risk-free rate. You adjust shares over simulated price paths, liquidate at expiry, and pay the option payoff. P&L is discounted to the starting date. Values are per share; multiply dollar values by 100 for a conventional 100-share contract.

With matching model volatility and no costs, average P&L should approach zero. Its standard deviation measures residual hedging uncertainty, not an expected return. More frequent hedging usually lowers this uncertainty while increasing costs. A lower standard deviation alone does not make a strategy better: inspect mean P&L and cost too.

Learn in this order:
1. Read optionlab/black_scholes.py: price and delta, then gamma and vega. Vega is per 1.00 volatility; divide by 100 for one percentage point. Theta is per year.
2. Compare the three call prices printed by play.py. Monte Carlo has sampling error; its confidence interval describes numerical estimation, not future investment returns.
3. Read optionlab/hedging.py beside the experiment table. Trace premium, shares, cash, fees, final payoff.
4. Read tests/test_hedging.py to understand the numerical claims being checked.
5. Explore binomial.py and monte_carlo.py, then validation.py.

## Reproduce the supplied reports

```powershell
.\.venv\Scripts\python.exe scripts/run_hedging.py
.\.venv\Scripts\python.exe scripts/run_validation.py
```

These overwrite the matching outputs supplied in results/. The original README records Claude's supplied run, not a guarantee of identical results across environments.

## Optional machine learning stage

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.2"
.\.venv\Scripts\python.exe scripts/train_surrogate.py
.\.venv\Scripts\python.exe -m pytest -q
```

Without torch, the surrogate test module is skipped. The neural network approximates Black-Scholes prices from generated training data; it does not predict markets. Start with the numerical experiments before this stage. Do not load unknown PyTorch checkpoint files.

## Limits

This is a working educational numerical project, not a live trading application. It uses constant-volatility GBM, no jumps, no dividends in hedging, no bid/ask model, and no market feeds. The basic pricing functions expect positive spot, strike, maturity and volatility; expiry and zero-volatility limits are not generally implemented. Supplied historical results and trained checkpoint are retained for reference.
