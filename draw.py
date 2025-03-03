import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("twilight_shifted")

import glob

csv_files = glob.glob("res*.csv")
dataframes = [pd.read_csv(file) for file in csv_files]
ddf = pd.concat(dataframes)

for s in [1]:
    df = ddf[ddf["seller"] == s]
    df = df[df["agent_type"] == "q"]
    # df = df[df["c"] <= 0.4]

    sns.lineplot(
        x="c",
        y="true_regret",
        data=df,
        markers=True,
        dashes=False,
        label="true_regret",
        marker="o",
    )

    sns.lineplot(
        x="c",
        y="estimated_regret",
        data=df,
        markers=True,
        dashes=False,
        label="estimated_regret",
        marker="o",
    )

    plt.tight_layout()
    plt.ylabel("regret")
    plt.savefig(f"figure.pdf")
    plt.close()
