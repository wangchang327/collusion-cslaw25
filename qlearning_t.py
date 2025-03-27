import numpy as np
from tqdm import tqdm
import pickle
import random
from config import *

random.seed(1851)
np.random.seed(1851)

num_prices = 19
priceset = np.linspace(1 / (num_prices + 1), 1, num_prices, endpoint=False)


gamma = 0.99
alpha = 0.05
beta = 0.0002


class Seller:
    def __init__(self, cost=0, T=0):
        self.qtable = np.random.uniform(size=num_prices) * 100
        self.history = []
        self.t = 0
        self.gamma = gamma
        self.alpha = alpha
        self.cost = cost
        self.eps = 1 / (T ** (1 / 3))

    # select the action using epsilon-greedy
    def act(self):
        self.pi = np.ones(num_prices) * self.eps / num_prices
        self.pi[np.argmax(self.qtable)] += 1 - self.eps
        if np.random.random() < self.eps:
            return np.random.choice(num_prices)
        else:
            return np.argmax(self.qtable)

    # update the Q-table and the transcript
    def update(self, p, x, reward, opponent):
        self.t += 1
        self.history.append((p, self.pi, x, reward, opponent))
        self.qtable[p] = self.qtable[p] + self.alpha * (
            reward + self.gamma * max(self.qtable) - self.qtable[p]
        )


heatmap = np.zeros((num_prices, num_prices))
cost1 = 0.1
cost2 = 0.2

# compute and print the normal form of the pricing game
normal_form = []
for i in range(num_prices):
    row = []
    for j in range(num_prices):
        row.append(
            (
                round(
                    float((cdf(priceset[i], priceset[j])) * (priceset[i] - cost1)), 3
                ),
                round(float(cdf(priceset[j], priceset[i]) * (priceset[j] - cost2)), 3),
            )
        )
    normal_form.append(row)
from tabulate import tabulate

print(tabulate(normal_form))


# run the experiment and save the transcripts
for i in tqdm(range(N)):
    for T in tqdm(T_range):
        seller1 = Seller(cost=cost1, T=T)
        seller2 = Seller(cost=cost2, T=T)

        for t in range(T):
            seller1_act = seller1.act()
            seller2_act = seller2.act()

            price1 = priceset[seller1_act]
            price2 = priceset[seller2_act]

            x1 = cdf(price1, price2)
            x2 = cdf(price2, price1)

            reward1 = (price1 - cost1) * x1
            reward2 = (price2 - cost2) * x2

            seller1.update(seller1_act, x1, reward1, seller2_act)
            seller2.update(seller2_act, x2, reward2, seller1_act)
        with open(f"transcripts/q-{i}-{T}-seller1.pkl", "wb") as f:
            pickle.dump(seller1.history, f)
        with open(f"transcripts/q-{i}-{T}-seller2.pkl", "wb") as f:
            pickle.dump(seller2.history, f)

        for x, y in zip(seller1.history[-10:], seller2.history[-10:]):
            heatmap[x[0]][y[0]] += 1


heatmap /= N * 10

import seaborn as sns
import matplotlib.pylab as plt

# draw the heatmap of the final strategies
ax = sns.heatmap(
    heatmap,
    cmap="Purples",
    xticklabels=np.round(priceset, 2),
    yticklabels=np.round(priceset, 2),
    linewidths=0.5,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 64 / num_prices},
)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("qlearning-heatmap.pdf")
