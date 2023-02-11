import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score


class ML_model:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.model = lgb.LGBMRegressor()
        self.model.fit(X=X_train, y=y_train, eval_metric="l1")

        self.y_train_pred = self.model.predict(X_train)
        self.train_mae = mean_absolute_error(
            y_true=y_train, y_pred=self.y_train_pred
        )
        self.train_r2_score = r2_score(
            y_true=y_train, y_pred=self.y_train_pred
        )

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
