import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("final_result.json", "r") as f:
    results = json.load(f)

# Convert to DataFrame and melt for plotting
df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "method"})
df_melted = df.melt(id_vars="method", var_name="metric", value_name="score")

# Shorten method names for readability
df_melted["method"] = df_melted["method"].str.replace(
    r"squality_", "", regex=True
).str.replace(
    r"_openai-gpt.*", "", regex=True
)

# Use shortened names for color palette
methods_short = df_melted["method"].unique()
palette = sns.dark_palette("red", n_colors=len(methods_short), reverse=True)
color_map = dict(zip(methods_short, palette))

sns.set_theme(style="whitegrid", context="talk")

g = sns.catplot(
    data=df_melted,
    x="method",
    y="score",
    col="metric",
    kind="bar",
    hue="method",
    palette=color_map,
    height=5,
    aspect=0.8,
    sharex=False,
    legend=False,
)

# Beautify each facet
for ax in g.axes.flat:
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=45)

g.fig.suptitle("Model Comparison by Metric", fontsize=18, y=1.05)
g.despine(left=True)

# Save to file
g.savefig("method_comparison_by_metric.png", dpi=300)
plt.show()