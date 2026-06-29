import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

algorithm_pairs = pd.read_csv(
    "HT_LLM/similarity/algorithm_similarity_results/AS_BOO_weighted_jaccard_pairs.csv"
)

feature_pairs = pd.read_csv(
    "HT_LLM/similarity/feature_similarity_results/FS_pairs.csv"
)

scatter_df = feature_pairs.merge(
    algorithm_pairs[["pair_key", "algorithm_similarity"]],
    on="pair_key",
    how="inner",
)

plt.figure(figsize=(7, 6))

ax = sns.scatterplot(
    data=scatter_df,
    x="algorithm_similarity",
    y="feature_similarity",
    alpha=0.7,
    
)


# Label certain points by their method pair name for identification
counter = 0
for idx, row in scatter_df.iterrows():
    # Only label points satisfying this condition
    if row['algorithm_similarity'] > 0.5:
        ax.text(
            row['algorithm_similarity'] + 0.01,             # Add a slight x-offset manually
            row['feature_similarity'] + 0.01,             # Add a slight y-offset manually
            row['pair_key'],                   # The labelled text is the method pair's name
            color='red',                    # Highlight color
            weight='bold'
        )
    counter += 1

# Sanity checks for correct number of method pairs (not missing any)
print("Feature pairs:", len(feature_pairs))
print("Algorithm pairs:", len(algorithm_pairs))
print("Merged pairs:", len(scatter_df))
print("Plotted points:", counter)

plt.xlim(0, 1)
plt.ylim(-0.2, 1)
plt.axvline(x=0.5)
plt.axhline(y=0.5)

plt.xlabel("Algorithm similarity")
plt.ylabel("dFC feature similarity")
plt.title("Algorithm similarity vs. dFC feature similarity for method pairs")

plt.tight_layout()
plt.savefig("HT_LLM/similarity/algorithm_vs_feature_similarity_scatter.png", dpi=600)
plt.savefig("HT_LLM/similarity/algorithm_vs_feature_similarity_scatter.pdf")
plt.show()