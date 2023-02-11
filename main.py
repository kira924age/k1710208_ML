#!/usr/bin/env python3

import datetime
import logging
import os
import time

import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from analytics.analytics import Analytics
from ml_model.ml_model import ML_model


def init():
    current_time = datetime.datetime.now().isoformat()
    os.mkdir(f"result/{current_time}")

    logging.basicConfig(
        filename=f"result/{current_time}/log.txt",
        encoding="utf-8",
        level=logging.INFO,
    )

    return current_time


def main():
    start_time = time.time()
    directory_name = init()

    dataset_df = pl.read_csv("data/dataset.csv").sample(n=10000, seed=42)
    # dataset_df = pl.read_csv("data/dataset.csv")

    n_samples = len(dataset_df)
    logging.info(f"number of all samples: {n_samples}")

    X = dataset_df.select(dataset_df.columns[2:])
    y = dataset_df.select(["y_relaxed"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    logging.info(f"training sample number: {len(X_train)}")
    logging.info(f"testing sample number: {len(X_test)}")
    logging.info(f"number of features: {len(X.columns)}")
    logging.info(
        "features:\n"
        + "\n".join([" " * 4 + column for column in X_train.columns])
    )

    model = ML_model(
        X_train=X_train.to_numpy(), y_train=y_train.to_series().to_numpy()
    )
    logging.info(f"TRAIN MAE: {model.train_mae:.4f}")
    logging.info(f"TRAIN R^2: {model.train_r2_score:.4f}")

    print(f"train MAE = {model.train_mae:.4f}")
    print(f"train r^2 = {model.train_r2_score:.4f}")

    y_test_pred = model.predict(X_test.to_numpy())
    test_mae = mean_absolute_error(
        y_true=y_test.to_numpy(), y_pred=y_test_pred
    )
    test_r2_score = r2_score(y_true=y_test.to_numpy(), y_pred=y_test_pred)

    logging.info(f"TEST MAE: {test_mae:.4f}")
    logging.info(f"TEST R^2: {test_r2_score:.4f}")

    print(f"test MAE = {test_mae:.4f}")
    print(f"test r^2 = {test_r2_score:.4f}")

    analytics = Analytics(output_dir=f"result/{directory_name}")
    analytics.plot_true_vs_predicted_graph(
        y_true=model.y_train,
        y_pred=model.y_train_pred,
        filename="(train)true_vs_predicted_plot_graph",
        title=f"[TRAIN] n={n_samples}",
    )
    analytics.plot_true_vs_predicted_graph(
        y_true=y_test.to_numpy().ravel(),
        y_pred=y_test_pred,
        filename="(test)true_vs_predicted_plot_graph",
        title=f"[TEST] n={n_samples}",
    )
    analytics.plot_pfi(
        model=model.model,
        X_val=X_test.to_numpy(),
        y_val=y_test.to_numpy().ravel(),
        columns=X_test.columns,
        n_samples=n_samples,
        filename="pfi.png",
    )

    end_time = time.time()
    logging.info(f"Execution Time: {(end_time - start_time) / 60:.2f} min")


main()
