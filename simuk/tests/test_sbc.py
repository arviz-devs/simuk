import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import pytest

import simuk

np.random.seed(1234)

data = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

with pm.Model() as centered_eight:
    mu = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfCauchy("tau", beta=5)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
    y_obs = pm.Normal("y", mu=theta, sigma=sigma, observed=data)

x = np.random.normal(0, 1, 20)
y = 2 + np.random.normal(x, 1)
df = pd.DataFrame({"x": x, "y": y})
bmb_model = bmb.Model("y ~ x", df)


@pytest.mark.parametrize("model", [centered_eight, bmb_model])
def test_sbc(model):
    sbc = simuk.SBC(
        model,
        num_simulations=10,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations
