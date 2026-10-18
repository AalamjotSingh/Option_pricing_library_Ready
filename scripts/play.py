"""Editable-input pricing and hedging experiment. See START_HERE.md."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optionlab import black_scholes as bs, binomial, monte_carlo, hedging

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in [("spot",100.),("strike",100.),("years",1.),("rate",.05),("vol",.2),("cost",.001)]:
        p.add_argument("--"+name, type=float, default=default)
    p.add_argument("--paths", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    vals = [a.spot,a.strike,a.years,a.rate,a.vol,a.cost]
    if not np.isfinite(vals).all() or min(a.spot,a.strike,a.years,a.vol)<=0 or a.cost<0 or not 100<=a.paths<=100000:
        p.error("Use finite inputs, positive spot/strike/years/vol, nonnegative cost, and 100..100000 paths.")
    args=(a.spot,a.strike,a.years,a.rate,a.vol)
    print("European call prices, dollars per share:")
    print("Black-Scholes:", round(bs.price(*args),6))
    print("CRR (1000 steps):", round(binomial.crr_price(*args,steps=1000),6))
    mc=monte_carlo.mc_european(*args,n_paths=a.paths,control_variate=True,seed=a.seed)
    print("Monte Carlo:", round(mc.price,6), "95% CI:", mc.ci95)
    print("Greeks:", bs.greeks(*args))
    paths=hedging.simulate_gbm(a.spot,a.rate,a.vol,a.years,260,a.paths,np.random.default_rng(a.seed))
    rows=[]
    for every in [260,130,65,20,10,5,2,1]:
        res=hedging.delta_hedge_short_call(paths,a.strike,a.years,a.rate,a.vol,every=every,cost_rate=a.cost)
        rows.append(hedging.summarize(res))
    df=pd.DataFrame(rows)
    print("\nHedging results (% of initial option premium):\n",df.round(3).to_string(index=False))
    out=Path(__file__).resolve().parents[1]/"results"/"play"
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"experiment.csv",index=False)
    (out/"inputs.json").write_text(__import__("json").dumps(vars(a),indent=2))
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(df.n_intervals,df.std_pnl_pct,"o-")
    ax[0].set(xlabel="Hedge intervals",ylabel="P&L std (% of premium)",xscale="log")
    ax[1].plot(df.mean_cost_pct,df.std_pnl_pct,"o-")
    ax[1].set(xlabel="Mean total cost (% of premium)",ylabel="P&L std (% of premium)")
    fig.tight_layout(); fig.savefig(out/"experiment.png",dpi=140); plt.close(fig)
    print("\nSaved CSV, inputs and chart to",out)
if __name__=="__main__":
    main()
