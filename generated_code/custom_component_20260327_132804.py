import pandas as pd

def run_component(df, features, target, task_type='regression'):
    """
    Remove constant (zero-variance) features from the feature list.
    Parameters
    ----------
    df : pd.DataFrame        Input data.
    features : list of str
        Candidate feature column names.
    target : str
        Target column name (not used in filtering).
    task_type : str, optional        Type of task ('regression' or 'classification'), default 'regression'.
    Returns
    -------
    dict
        {'features': filtered feature list, 'note': 'removed constant features'}
    """
    # Guard against invalid inputs
    if not isinstance(df, pd.DataFrame) or not features:
        return {"features": features, "note": "removed constant features"}

    filtered = []
    for col in features:
        if col in df.columns:
            # Consider a feature constant if it has 0 or 1 unique values
            if df[col].nunique() > 1:
                filtered.append(col)
        else:
            # Column missing from dataframe; keep it (could be handled elsewhere)
            filtered.append(col)

    note = "removed constant features"
    return {"features": filtered, "note": note}