def run_component(df, features, target, task_type='regression'):
    import pandas as pd
    present = [f for f in features if f in df.columns]
    filtered = []
    for col in present:
        series = df[col].dropna()
        if series.nunique() > 1:
            filtered.append(col)
    note = f"Removed {len(present)-len(filtered)} constant features"
    return {"features": filtered, "note": note}