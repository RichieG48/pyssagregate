# ssaggregate (Python)

A Python port of `ssaggregate`, the command accompanying
[Borusyak, Hull and Jaravel (2022), "Quasi-Experimental Shift-Share Research Designs"](https://doi.org/10.1093/restud/rdab030),
*Review of Economic Studies* 89(1): 181–213. The original is available for Stata and R.

`ssaggregate` converts a **location-level** shift-share IV dataset into a dataset of
exposure-weighted **industry-level** (shock-level) aggregates. BHJ show that the
location-level shift-share IV estimate is identical to a shock-level IV regression
on these aggregates, weighted by average exposure `s_n`. Running the regression at
the shock level makes it straightforward to get valid standard errors (clustered or
robust at the shock level), add shock-level controls, and run balance tests.

Residualization is done with [pyfixest](https://github.com/py-econometrics/pyfixest),
so controls and high-dimensional fixed effects use the familiar formula syntax.

## Installation

```bash
pip install git+https://github.com/RichieG48/ssaggregate.git
```

Depends on `pandas`, `numpy` and `pyfixest` (recent pyfixest releases require Python ≥ 3.10).

## What it does

Given location-level variables $y_\ell$ (outcome, endogenous regressor, ...),
exposure shares $s_{\ell n}$, optional location weights $e_\ell$ and controls $w_\ell$,
`ssaggregate`:

1. Residualizes each variable on the controls (weighted by $e_\ell$ if given):
   $y_\ell^\perp$.
2. Computes, for each industry $n$ (and period $t$), the exposure-weighted average
   $\bar y_n^\perp = \sum_\ell e_\ell s_{\ell n} y_\ell^\perp / \sum_\ell e_\ell s_{\ell n}$.
3. Computes the industry's average exposure $s_n \propto \sum_\ell e_\ell s_{\ell n}$,
   normalized to sum to one.

The output has one row per industry (per period), with the aggregated variables
under their original names and the weight `s_n`.

## Usage

### Shares in long format

Pass a separate `shares` DataFrame with one row per location × industry (× period):

```python
import pandas as pd
from ssaggregate import ssaggregate

# data:   one row per location (x period) with y, x, controls, weights
# shares: columns czone, year, sic87dd, ind_share

industry = ssaggregate(
    data=data,
    vars_list=["y", "x"],
    shares=shares,
    l="czone",          # location identifier
    t="year",           # period identifier (omit for a cross-section)
    n="sic87dd",        # industry identifier in `shares`
    s="ind_share",      # share column in `shares`
    weights="wei",      # location weights (optional)
    controls="t2 + Lsh_manuf | czone",  # pyfixest syntax, FEs after "|"
)
```

### Shares in wide format

If the shares are columns of `data` sharing a common prefix (e.g. `emp_101`,
`emp_102`, ...), omit `shares` and `l`, and pass the prefix as `s`. `n` is then the
name of the industry column to create; industry IDs are taken from the column
suffixes (as strings).

```python
industry = ssaggregate(
    data=data,
    vars_list=["y", "x"],
    s="emp_",           # prefix of the share columns
    n="industry",       # name for the new industry column
    t="year",
    controls="t2",
)
```

### Running the shock-level IV

Merge your shocks onto the output and run a weighted IV regression with `s_n` as
weights and heteroskedasticity-robust (or shock-clustered) standard errors:

```python
import pyfixest as pf

industry = industry.merge(shocks, on=["sic87dd", "year"])

fit = pf.feols("y ~ 1 | year | x ~ g", data=industry,
               weights="s_n", vcov="hetero")
fit.summary()
```

See BHJ (2022), Section 4, for the choice of shock-level controls and standard errors.

## Arguments

| Argument | Description |
|---|---|
| `data` | Location-level DataFrame. No missing values in the aggregated variables (after sample restrictions). |
| `vars_list` | List of variables to residualize and aggregate. |
| `n` | Industry identifier. Long format: column in `shares`. Wide format: name of the column to create. |
| `s` | Long format: share column in `shares`. Wide format: prefix of the share columns in `data`. |
| `shares` | Long-format shares DataFrame, unique by `l`, `n` (and `t`). `None` for wide format. |
| `l` | Location identifier. Required for long format; must be `None` for wide format. |
| `t` | Period identifier, for panels or repeated cross-sections. |
| `weights` | Location weights used in residualization and aggregation. |
| `controls` | pyfixest formula RHS for controls and fixed effects, e.g. `"x1 + x2 \| state"`. Default `"1"` (intercept only). |
| `addmissing` | If `True`, adds a "missing industry" (`n` = NaN) with share $1 - \sum_n s_{\ell n}$. |

## Incomplete shares

When shares do not sum to the same value in every location (e.g. manufacturing
shares of total employment), BHJ recommend either controlling for the sum of
shares at the location level, or adding the "missing industry" with
`addmissing=True`. `ssaggregate` warns if the sum of shares varies and is not
spanned by `controls` while `addmissing=False`.

## Testing

```bash
pip install pytest
pytest
```

## Citation

If you use this package, please cite the original paper:

```bibtex
@article{borusyak2022quasi,
  title   = {Quasi-Experimental Shift-Share Research Designs},
  author  = {Borusyak, Kirill and Hull, Peter and Jaravel, Xavier},
  journal = {The Review of Economic Studies},
  volume  = {89},
  number  = {1},
  pages   = {181--213},
  year    = {2022}
}
```

## Author

Gabriel Richard. Python port written for a dissertation; issues and pull requests welcome.
