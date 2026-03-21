import pandas as pd
import numpy as np
import warnings
from pyfixest.estimation import feols

def ssaggregate(data: pd.DataFrame, vars_list: list, n: str, s: str, 
                shares: pd.DataFrame = None, l: str = None, t: str = None, 
                weights: str = None, controls: str = "1", addmissing: bool = False) -> pd.DataFrame:
    """
    Converts "location-level" variables in a shift-share IV dataset to a dataset 
    of exposure-weighted "industry-level" aggregates.
    """
    df = data.copy()
    sh = shares.copy() if shares is not None else None

    # 1. Wide vs Long format
    wideformat = sh is None
    if wideformat and l is not None:
        raise ValueError("Option 'l' may not be used with shares in wide format")

    if wideformat:
        l = "location_ids"
        df["location_ids"] = range(len(df))
        
        keep_cols = ["location_ids"]
        if t: keep_cols.append(t)
        share_cols = [c for c in df.columns if s in c]
        keep_cols.extend(share_cols)

        sh = df[keep_cols].copy()
        
        id_vars = ["location_ids"]
        if t: id_vars.append(t)
        
        sh = sh.melt(id_vars=id_vars, value_vars=share_cols, var_name=n, value_name=s)
        sh[n] = sh[n].astype(str).str.replace(s, "")

    # 2. Check controls / sum of shares varying
    check_controls = sh[s].std() > 1e-5
    groupby_cols = [l] if t is None else [l, t]

    sh_missing = sh.groupby(groupby_cols, as_index=False)[s].sum()
    sh_missing[s] = 1.0 - sh_missing[s]

    if addmissing:
        sh_missing[n] = np.nan
        sh = pd.concat([sh, sh_missing], ignore_index=True)

    if check_controls:
        temp = df.merge(sh_missing, on=groupby_cols)
        formula = f"{s} ~ {controls}"
        try:
            fit = feols(formula, data=temp)
            rsq = fit._r2  # Accessing R-squared from pyfixest
        except Exception:
            rsq = 1.0

        if rsq < 0.9999 and not addmissing:
            warnings.warn(
                "You are in the incomplete share case (the sum of exposure shares varies) "
                "and you have not controlled for the sum of shares. You should either include "
                "the missing industry or add the sum-of-share control."
            )

    # 3. Residualize variables using pyfixest
    for var in vars_list:
        formula = f"{var} ~ {controls}"
        if weights:
            fit = feols(formula, data=df, weights=weights)
        else:
            fit = feols(formula, data=df)
        df[var] = fit.resid()

    # 4. Merge and calculate exposure-weighted aggregates
    if wideformat:
        df = df.drop(columns=[c for c in df.columns if s in c])

    merged = df.merge(sh, on=groupby_cols)

    if weights:
        merged[s] = merged[s] * merged[weights]

    agg_groupby = [n] if t is None else [n, t]

    def compute_weighted_aggregates(grp):
        # Calculate sum of shares (s_n)
        s_sum = grp[s].sum()
        res = {'s_n': s_sum}
        # Calculate weighted average for each residualized variable
        for v in vars_list:
            res[v] = (grp[v] * grp[s]).sum() / s_sum if s_sum != 0 else np.nan
        return pd.Series(res)

    collapsed = merged.groupby(agg_groupby, dropna=False).apply(compute_weighted_aggregates).reset_index()

    # Normalize s_n so it sums to 1 globally
    collapsed['s_n'] = collapsed['s_n'] / collapsed['s_n'].sum()

    # Final cleanup
    collapsed = collapsed.sort_values(by=agg_groupby).reset_index(drop=True)
    return collapsed