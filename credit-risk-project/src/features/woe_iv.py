from xverse.transformer import WOE
import pandas as pd

def apply_woe(df: pd.DataFrame, target_col: str):
    """
    Apply Weight of Evidence transformation.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    woe = WOE()
    X_woe = woe.fit_transform(X, y)

    return X_woe, woe
