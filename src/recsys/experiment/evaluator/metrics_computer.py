import pandas as pd
from ...msr.python_evaluation import (
    hit_ratio_at_k,
    precision_at_k, 
    recall_at_k,
    map_at_k, 
    ndcg_at_k, 
)


class MetricsComputer:
    def __init__(
        self, 
        k: list,
        schema,
    ):
        self.k = k
        self.schema = schema

    def __call__(
        self, 
        result: pd.DataFrame, 
    ):
        rating_true, rating_pred = self._seperator(result)

        eval_list = []

        for k in self.k:
            kwargs = dict(
                rating_true=rating_true, 
                rating_pred=rating_pred, 
                k=k,
            )
            eval = self._computer(**kwargs)
            eval_list.append(eval)

        return pd.DataFrame(eval_list)

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

    def _computer(self, rating_true, rating_pred, k):
        kwargs = dict(
            rating_true=rating_true,
            rating_pred=rating_pred.head(k),
            col_user=self.schema.col_user,
            col_item=self.schema.col_item,
            col_rating=self.schema.col_rating,
            col_prediction=self.schema.col_prediction,
            k=k,
        )
        return dict(
            k=k,
            hit_ratio=hit_ratio_at_k(**kwargs), 
            precision=precision_at_k(**kwargs), 
            recall=recall_at_k(**kwargs), 
            map=map_at_k(**kwargs), 
            ndcg=ndcg_at_k(**kwargs),
        )