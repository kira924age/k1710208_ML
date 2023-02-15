import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.inspection import (PartialDependenceDisplay, partial_dependence,
                                permutation_importance)


class Analytics:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def plot_true_vs_predicted_graph(self, y_true, y_pred, title, filename):
        y_max = max(y_pred.max(), y_true.max())
        y_min = min(y_pred.min(), y_true.min())

        # plot diagonal line from min-5% to max+5%
        plt.plot(
            [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
            [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
            "k-",
        )

        sns.scatterplot(
            x=y_true,
            y=y_pred,
        )

        plt.title(title, fontsize=20)
        plt.xlabel("DFT calculated / eV", fontsize=16)
        plt.ylabel("ML predicted / eV", fontsize=16)

        plt.savefig(f"{self.output_dir}/{filename}.png")

        plt.clf()
        plt.cla()
        plt.close()

    def plot_pfi(self, model, X_val, y_val, columns, n_samples, filename):
        show_feature_num = 20

        result = permutation_importance(
            estimator=model,
            X=X_val,
            y=y_val,
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        perm_sorted_idx = result.importances_mean.argsort()[::-1]
        features = [columns[idx] for idx in perm_sorted_idx[:show_feature_num]]

        d = {"feature": [], "decrease": []}
        for feature, importance in zip(
            features,
            result.importances[perm_sorted_idx[:show_feature_num]],
        ):
            d["feature"] += [feature] * len(importance)
            d["decrease"] += list(np.array(importance).flatten())

        sns.boxplot(
            data=pl.DataFrame(d).to_pandas(),
            x="decrease",
            y="feature",
            orient="h",
            linewidth=0.8,
            fliersize=0.5,
        )
        plt.subplots_adjust(left=0.55)
        plt.savefig(f"{self.output_dir}/{filename}.png")

        plt.clf()
        plt.cla()
        plt.close()

    def plot_pdp(self, model, X, filename):
        target_idx = 0
        # target_idx = X.columns.index("weighted_mean_unfilled_d_states")
        # target_idx = X.columns.index("weighted_mean_group")

        features, feature_names = [(target_idx,)], X.columns

        X = X.to_pandas()

        deciles = {0: np.linspace(0, 1, num=5)}
        pd_results = partial_dependence(
            model, X, features=target_idx, kind="average", grid_resolution=100
        )
        display = PartialDependenceDisplay(
            [pd_results],
            features=features,
            feature_names=feature_names,
            target_idx=0,
            deciles=deciles,
        )
        display.plot()

        plt.savefig(f"{self.output_dir}/{filename}.png")

        plt.clf()
        plt.cla()
        plt.close()
