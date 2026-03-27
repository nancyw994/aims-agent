def run_component(df, features, target, task_type='regression'):
    import pandas as pd
    # Guard against invalid inputs
    if not isinstance(df, pd.DataFrame) or not features:
        return {"features": features, "note": "removed constant features"}
    # Identify constant features (including NaN-only columns)
    constant_cols = [
        col for col in features
        if col in df.columns and df[col].nunique(dropna=False) <= 1
    ]
    filtered_features = [col for col in features if col not in constant_cols]
    return {"features": filtered_features, "note": "removed constant features"}