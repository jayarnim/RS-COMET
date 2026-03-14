import pandas as pd
from .metric.builder import metric_builder


class MetricsComputer:
    def __init__(
        self, 
        criterion,
        k: int,
        schema,
    ):
        self.criterion = criterion
        self.k = k
        self.schema = schema

    def __call__(
        self,
        result: pd.DataFrame,
    ):
        rating_true, rating_pred = self._seperator(result)
        score = self._computer(rating_true, rating_pred)
        return score

    def _seperator(self, result):
        TRUE_COL_LIST = [self.schema.col_user, self.schema.col_item, self.schema.col_rating]
        PRED_COL_LIST = [self.schema.col_user, self.schema.col_item, self.schema.col_prediction]

        rating_true = (
            result[TRUE_COL_LIST]
            [result[self.schema.col_rating]==1]
            .sort_values(
                by=self.schema.col_user, 
                ascending=True,
            )
        )

        rating_pred = (
            result[PRED_COL_LIST]
            .sort_values(
                by=[self.schema.col_user, self.schema.col_prediction], 
                ascending=[True, False], 
                kind='stable',
            ).groupby(self.schema.col_user)
        )

        return rating_true, rating_pred

    def _computer(self, rating_true, rating_pred):
        kwargs = dict(
            rating_true=rating_true,
            rating_pred=rating_pred.head(self.k),
            col_user=self.schema.col_user,
            col_item=self.schema.col_item,
            col_rating=self.schema.col_rating,
            col_prediction=self.schema.col_prediction,
            k=self.k,
        )
        return self.criterion(**kwargs)