import numpy as np
import pickle
from config import *
from tqdm import tqdm
import pandas as pd

num_prices = 19
priceset = np.linspace(1 / (num_prices + 1), 1, num_prices, endpoint=False)


def estimate_regret(transcript, assumed_cost, true_allocation=False):
    estimated_allocation = np.zeros((T, num_prices))
    for i, t in enumerate(transcript):
        p, pi, x, _, opponent = t
        empirical_dist = pi
        if true_allocation:  # true regret
            estimated_allocation[i] = np.array(
                [cdf(priceset[j], priceset[opponent]) for j in range(num_prices)]
            )
        else:  # estimated regret
            estimated_allocation[i][p] = x / empirical_dist[p]

    swap = 0
    for i in range(num_prices):  # find the best swaps
        best_swap = 0
        for j in range(num_prices):
            if i == j:
                continue
            sw = 0
            for k, t in enumerate(transcript):
                sw += (
                    (priceset[j] - assumed_cost) * estimated_allocation[k][j]
                    - (priceset[i] - assumed_cost) * estimated_allocation[k][i]
                ) * empirical_dist[i]
            best_swap = max(best_swap, sw)
        swap += best_swap
    return swap / len(transcript)


def job(agent_type, s, i, c):
    # print(f"started agent_type={agent_type}, s={s}, i={i}, c={c}")
    with open(f"transcripts/{agent_type}-{i}-seller{s}.pkl", "rb") as f:
        transcript = pickle.load(f)
    res = [
        T,
        c,
        agent_type,
        s,
        estimate_regret(
            transcript,
            c,
            true_allocation=True,
        ),
        estimate_regret(
            transcript,
            c,
            true_allocation=False,
        ),
    ]
    # print(f"finihsed agent_type={agent_type}, s={s}, i={i}, c={c}")
    return res


repeat = list(range(N))

columns = [
    "t",
    "c",
    "agent_type",
    "seller",
    "true_regret",
    "estimated_regret",
]
res = []
params = []
for agent_type in ["q"]:
    for s in [1]:
        for i in repeat:
            for c in [
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
            ]:
                params.append((agent_type, s, i, c))


def execute_job(p):
    return job(*p)


import multiprocess as mp

p = mp.Pool(mp.cpu_count())

res = list(tqdm(p.imap(execute_job, params), total=len(params)))
p.close()
p.join()

df = pd.DataFrame(res, columns=columns)
df.to_csv(f"res{min(repeat)}-{max(repeat)}.csv")
