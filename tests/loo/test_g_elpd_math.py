import numpy as np
import pytest
import warnings
from arviz_base import from_dict
from arviz_stats.loo import loo

def generate_mock_idata():
    """Generates a mock InferenceData object with required groups."""
    # 4 chains, 100 draws, 10 observations
    log_lik = np.random.randn(4, 100, 10)
    
    # Use absolute values for predictive draws so we don't pass 
    # negative numbers into fractional beta powers
    post_pred = np.abs(np.random.randn(4, 100, 10))
    
    # Dummy posterior for reff calculations
    theta = np.random.randn(4, 100, 2)
    
    idata = from_dict({
        "posterior": {"theta": theta},
        "log_likelihood": {"y": log_lik},
        "posterior_predictive": {"y": post_pred}
    })
    return idata


def test_g_elpd_beta_warning():
    """Tests that beta <= 1.0 correctly issues a warning and falls back to log score."""
    idata = generate_mock_idata()
    
    with pytest.warns(UserWarning, match="Beta must be strictly greater than 1.0"):
        # This should trigger the warning and revert score_type to "log"
        # We also suppress the Pareto k warning that might co-occur
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Estimated shape parameter")
            result = loo(idata, score_type="beta", beta=1.0)
    
    assert result is not None
    assert result.kind == "loo"


def test_g_elpd_beta_execution():
    """Tests that the beta-divergence math executes without throwing shape or calculation errors."""
    idata = generate_mock_idata()
    
    with warnings.catch_warnings():
        # Ignore the expected Pareto k warnings caused by random mock data
        warnings.simplefilter("ignore")
        
        # Run the standard LOO as a baseline
        result_log = loo(idata, score_type="log")
        
        # Run the new generalized ELPD
        result_beta = loo(idata, score_type="beta", beta=1.05)
    
    assert result_beta is not None
    assert hasattr(result_beta, "elpd")
    
    # The calculated ELPDs should differ because the scoring rules differ
    assert result_beta.elpd != result_log.elpd


def test_g_elpd_missing_posterior_predictive():
    """Tests that the function properly catches missing posterior_predictive groups."""
    log_lik = np.random.randn(4, 100, 10)
    theta = np.random.randn(4, 100, 2)
    
    # Omit posterior_predictive
    idata_incomplete = from_dict({
        "posterior": {"theta": theta},
        "log_likelihood": {"y": log_lik}
    })
    
    with pytest.raises(ValueError, match="The posterior_predictive group"):
        loo(idata_incomplete, score_type="beta", beta=1.05)