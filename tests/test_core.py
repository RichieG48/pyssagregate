import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import pytest
import pandas as pd
import numpy as np
from ssaggregate import ssaggregate

@pytest.fixture
def sample_data():
    """Creates a dummy shift-share dataset for testing."""
    # Location-level data
    df = pd.DataFrame({
        'l': [1, 2, 1, 2],
        't': [1, 1, 2, 2],
        'y': [10.0, 20.0, 15.0, 25.0],
        'x': [2.0, 4.0, 3.0, 5.0],
        'c': [1, 0, 1, 0],        # Control variable
        'wei': [100, 200, 100, 200] # Population weights
    })
    
    # Long format exposure shares
    shares = pd.DataFrame({
        'l': [1, 1, 2, 2, 1, 1, 2, 2],
        't': [1, 1, 1, 1, 2, 2, 2, 2],
        'n': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        's': [0.8, 0.2, 0.3, 0.7, 0.8, 0.2, 0.3, 0.7]
    })
    
    return df, shares

def test_long_format(sample_data):
    """Test aggregation using the separate 'long' shares dataset."""
    df, shares = sample_data
    
    res = ssaggregate(
        data=df, 
        vars_list=['y', 'x'], 
        n='n', 
        s='s', 
        shares=shares, 
        l='l', 
        t='t',
        weights='wei',
        controls="1" # No controls, just intercept
    )
    
    # Check shape and expected columns
    assert not res.empty
    assert set(['n', 't', 's_n', 'y', 'x']).issubset(res.columns)
    
    # Check that s_n (sum of normalized shares) adds up to 1 for each period
    assert np.isclose(res['s_n'].sum(), 1.0)

def test_wide_format(sample_data):
    """Test aggregation using the 'wide' shares format inside the main dataframe."""
    df, shares = sample_data
    
    # Reshape our test shares to wide format to mimic the wide case
    shares_wide = shares.pivot(index=['l', 't'], columns='n', values='s').reset_index()
    shares_wide = shares_wide.rename(columns={'A': 'share_A', 'B': 'share_B'})
    df_wide = df.merge(shares_wide, on=['l', 't'])
    
    res = ssaggregate(
        data=df_wide, 
        vars_list=['y', 'x'], 
        n='industry_id', 
        s='share_', 
        l=None, # 'l' must be None for wide format per the R documentation
        t='t',
        weights='wei',
        controls="1"
    )
    
    assert not res.empty
    assert 'industry_id' in res.columns
    # Check that it extracted the industry names correctly from the column stubs
    assert set(res['industry_id']) == {'A', 'B'}

def test_addmissing(sample_data):
    """Test the creation of the 'missing industry' observation when shares don't sum to 1."""
    df, shares = sample_data
    
    # Artificially make the sum of shares < 1 for location 1
    shares.loc[(shares['l'] == 1) & (shares['n'] == 'A'), 's'] = 0.5 
    
    res = ssaggregate(
        data=df, 
        vars_list=['y'], 
        n='n', 
        s='s', 
        shares=shares, 
        l='l', 
        t='t',
        addmissing=True
    )
    
    # If addmissing works, there should be rows where the industry 'n' is NaN
    assert res['n'].isna().any()

def test_residualization_with_controls(sample_data):
    """Test that specifying controls actually alters the aggregated outcome via pyfixest."""
    df, shares = sample_data
    
    res_no_controls = ssaggregate(
        data=df, vars_list=['y'], n='n', s='s', shares=shares, l='l', t='t', controls="1"
    )
    
    res_with_controls = ssaggregate(
        data=df, vars_list=['y'], n='n', s='s', shares=shares, l='l', t='t', controls="c"
    )
    
    # The aggregated y values should differ because 'y' was residualized against 'c'
    assert not np.allclose(res_no_controls['y'], res_with_controls['y'])
def test_no_warning_with_complete_shares(sample_data):
    """Shares summing to one in every location should not trigger the incomplete-share warning."""
    df, shares = sample_data

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ssaggregate(data=df, vars_list=['y'], n='n', s='s', shares=shares, l='l', t='t')

def test_wide_format_stub_is_prefix(sample_data):
    """Only columns starting with the stub are treated as shares."""
    df, shares = sample_data

    shares_wide = shares.pivot(index=['l', 't'], columns='n', values='s').reset_index()
    shares_wide = shares_wide.rename(columns={'A': 'emp_A', 'B': 'emp_B'})
    df_wide = df.merge(shares_wide, on=['l', 't'])
    df_wide['unemp_rate'] = [0.1, 0.2, 0.3, 0.4]

    res = ssaggregate(
        data=df_wide, vars_list=['y', 'unemp_rate'], n='industry_id', s='emp_', t='t'
    )

    assert set(res['industry_id']) == {'A', 'B'}
    assert 'unemp_rate' in res.columns
