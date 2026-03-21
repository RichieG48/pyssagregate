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
    Parameters
    ----------
    data : pd.DataFrame
        The main dataset containing location-level variables to be aggregated.
        Conditional on any sample restrictions, there should be no missing values 
        for the location-level variables being aggregated.

    vars_list : list of str
        A list of column names in `data` representing the variables (e.g., 
        outcomes and endogenous variables) that will be residualized and aggregated.
    
    n : str
        The variable name indicating industry (or shock) identifiers. 
        In "long" format, this is a column in the `shares` dataset. 
        In "wide" format, this will be the name of the newly created industry 
        identifier column in the output.
    
    s : str
        The variable name indicating the exposure weight. 
        In "long" format, this is the column name in `shares` containing the weights. 
        In "wide" format, this should denote the common prefix (stub name) of the 
        exposure weight columns in `data` (e.g., "share_" if columns are named 
        "share_101", "share_102").
    
    shares : pd.DataFrame, optional
        The shares dataset, used only for the "long" format. Each row should be 
        uniquely indexed by the variables in `l` and `n` (and `t`, when specified). 
        If None, the function assumes shares are in "wide" format inside `data`.
    
    l : str, optional
        The variable name indicating location identifiers. Required if using the 
        "long" format. Must be None if using the "wide" format.
    
    t : str, optional
        The variable name indicating period (time) identifiers. Required if the 
        data is a panel or repeated cross-section.
    
    weights : str, optional
        The variable name in `data` indicating population or regression weights 
        used to weight locations during residualization and final aggregation.
    
    controls : str, default "1"
        A `pyfixest` formula string specifying control variables and fixed effects 
        that will be partialled out from the variables in `vars_list` prior to 
        aggregation. Fixed effects should be specified after a "|". 
        Example: "demographic_var + i(state) | year"
    
    addmissing : bool, default False
        If True, creates a "missing industry" observation with exposure weights 
        equal to one minus the sum of a location's exposure weights. Recommended 
        when the sum of exposure weights varies across locations and the sum of 
        shares is not explicitly controlled for.

    Returns
    -------
    pd.DataFrame
        A DataFrame of exposure-weighted "industry-level" aggregates. The dataset 
        will contain the control-residualized, exposure-weighted averages of the 
        location-level variables, along with the global normalized average exposure 
        weight `s_n`. It is indexed by the variables in `n` (and `t`, if specified).
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