import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import glob

csv_files = glob.glob("t_res*.csv")
dataframes = [pd.read_csv(file) for file in csv_files]
ddf = pd.concat(dataframes)

for s in [1]:
    df = ddf[ddf["seller"] == s]
    df = df[df["agent_type"] == "q"]

    true = df[df["c"] == 0.1]
    plausible = df[df["c"] == 0.25]
    df = pd.concat(
        [true, plausible], axis=0, keys=["cost 0.1", "cost 0.25"]
    ).reset_index()
    df = df.rename(columns={"level_0": "cost", "level_1": "x"})
    sns.lineplot(
        data=df,
        x="t",
        y="true_regret",
        markers=True,
        dashes=False,
        hue="cost",
        marker="o",
        palette=["#00A933", "r"],
    )

    plt.tight_layout()
    plt.ylabel("regret")
    plt.xlabel("T (time horizon)")
    plt.savefig(f"figure_t.pdf")
    plt.close()
