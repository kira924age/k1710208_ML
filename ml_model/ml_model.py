import lightgbm as lgb
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class ML_model:
    def __init__(self, X_train, y_train, algorithm):
        self.X_train = X_train
        self.y_train = y_train

        self.algorithm = algorithm

        if algorithm == "LightGBM":
            self.model = lgb.LGBMRegressor()
            self.model.fit(
                X=X_train.to_numpy(),
                y=y_train.to_series().to_numpy(),
                eval_metric="l1",
            )

            self.y_train_pred = self.model.predict(X_train.to_numpy())
            self.train_mae = mean_absolute_error(
                y_true=y_train.to_series().to_numpy(), y_pred=self.y_train_pred
            )
            self.train_r2_score = r2_score(
                y_true=y_train.to_series().to_numpy(), y_pred=self.y_train_pred
            )

        if algorithm == "Lasso":
            self.model = Lasso(alpha=1.0)
            self.scaler = StandardScaler()

            scaled_X_train = self.scaler.fit_transform(X_train.to_numpy())
            self.model.fit(
                X=scaled_X_train,
                y=y_train.to_series().to_numpy(),
            )

            self.y_train_pred = self.model.predict(scaled_X_train)
            self.train_mae = mean_absolute_error(
                y_true=y_train.to_series().to_numpy(), y_pred=self.y_train_pred
            )
            self.train_r2_score = r2_score(
                y_true=y_train.to_series().to_numpy(), y_pred=self.y_train_pred
            )

    def predict(self, X):
        if self.algorithm == "LightGBM":
            y_pred = self.model.predict(X.to_numpy())
            return y_pred

        if self.algorithm == "Lasso":
            y_pred = self.model.predict(self.scaler.transform(X.to_numpy()))
            return y_pred
