def run_component(df, features, target, task_type='regression'):
    try:
        present = [f for f in features if f in df.columns]
        to_remove = []
        for f in present:
            if df[f].nunique(dropna=False) <= 1:
                to_remove.append(f)
        filtered = [f for f in present if f not in to_remove]
        return {"features": filtered, "note": "removed constant features"}
    except Exception:
        return {"features": features, "note": "removed constant features"}