---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Posterior Simulation-Based Calibration

**Posterior SBC** (Säilynoja et al., 2025) validates the inference algorithm
*conditional on observed data*, rather than averaging over the prior.

```{admonition} When to use Posterior SBC
:class: tip

Use **Prior SBC** when you want to check that your inference pipeline works
for a wide range of datasets generated under the prior.

Use **Posterior SBC** when you already have observed data and want to verify
that the inference algorithm is trustworthy *for that specific dataset*.
Posterior SBC focuses on the region of the parameter space that matters
for the observed data, making it more sensitive to local calibration issues.
```

```{jupyter-execute}

import pymc as pm
from arviz_plots import plot_ecdf_pit, style
import matplotlib.pyplot as plt
import numpy as np
import simuk

style.use("arviz-variat")
```

## How Posterior SBC works

Given a model $\pi(\theta, y) = \pi(\theta)\,\pi(y \mid \theta)$ and
observed data $y_{\text{obs}}$, Posterior SBC proceeds as follows:

1. **Fit the model** to $y_{\text{obs}}$ to obtain posterior draws
   $\theta'_i \sim \pi(\theta \mid y_{\text{obs}})$.
2. **Generate replicated data** from the posterior predictive:
   $y_i \sim \pi(y \mid \theta'_i)$.
3. **Augment** the observations: $y_{\text{aug}} = (y_{\text{obs}}, y_i)$.
4. **Re-fit the model** on the augmented data to get
   $\theta''_{i,1}, \ldots, \theta''_{i,S} \sim \pi(\theta \mid y_i, y_{\text{obs}})$.
5. **Compute the rank statistics** of $f(\theta'_i)$ among $f(\theta''_{i,1}), \ldots, f(\theta''_{i,S})$. Where $f$ is an optional test quantity applied to the parameters before computing ranks.

By the self-consistency of Bayesian updating, $\theta'_i$ is also a draw
from the augmented posterior $\pi(\theta \mid y_i, y_{\text{obs}})$.
Therefore the rank statistics should be **uniformly distributed** if the inference
is calibrated.

## Example: Linear Regression Model

### Define the model

```{admonition} Model requirements for Posterior SBC
:class: warning

Posterior SBC augments the observed data (concatenating original + replicated),
which changes its size. For this to work, store observed data in ``pm.Data``
containers, and specify size using the ``dims`` parameter instead of setting a static shape. 
If your model uses ``dims`` and ``coords``, you are also responsible for resizing them to the correct size corresponding to the new augmented dataset via the ``update_data`` callback.
Similarly, if your model has covariates, store them in ``pm.Data`` so they
can be resized in the same callback.
```

```{jupyter-execute}

random_seed = 42
np.random.seed(random_seed)

x_data = np.linspace(0, 10, 100)
y_data = np.random.normal(x_data ** 1.2, 1)

coords = {
    "obs_id": np.arange(len(x_data))
}

with pm.Model(coords=coords) as model:
    model_x_data = pm.Data("x_data", x_data, dims="obs_id")
    model_y_data = pm.Data("y_data", y_data, dims="obs_id")

    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)
    
    # pm.Deterministic forces PyMC to track this equation's output
    mu = pm.Deterministic("mu", alpha + beta * model_x_data)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=model_y_data)
```

### Fit the original posterior

First, we need the posterior samples from the observed data. These will
serve as the reference distribution for Posterior SBC.

```{jupyter-execute}

with model:
    idata = pm.sample(200, random_seed=random_seed, progressbar=False)
```

### Using `update_data` with covariates and `dims`

When your model uses `dims`/`coords` or has covariates stored in `pm.Data`,
you must provide an `update_data` callback that resizes everything to
match the augmented observations. The callback is called **before** the model
is re-conditioned, and runs inside the model context.

```{jupyter-execute}

def update_data(model, augmented_data, simulation_idx):
    with model:
        pm.set_data(
            {"x_data": np.concatenate([model["x_data"].get_value(), model["x_data"].get_value()])},
            coords={"obs_id": np.arange(len(augmented_data["y"]))},
        )
```

### Custom test quantities with `param_transform`

You can define a scalar test quantity applied to both the reference draw
and the posterior draws before computing the rank statistic. The function
receives `(param_name, param_value)` and should return a comparable value.

```{jupyter-execute}

def param_transform(param_name, param_value):
    return np.pow(param_value, 2)
```

### Run Posterior SBC

Pass `method="posterior"` and provide the `trace`. Each iteration
generates replicated data from the posterior predictive, augments it
with the original observations, and re-fits the model.

```{jupyter-execute}
sbc = simuk.SBC(
    model,
    method="posterior",
    trace=idata,
    param_transform=param_transform,
    update_data=update_data,
    num_simulations=50,
    seed=random_seed,
    sample_kwargs={"chains": 4, "draws": 50, "tune": 50},
    progress_bar=False,
)

sbc.run_simulations();
```

### Visualize the results

We expect the ECDF lines to fall inside the grey simultaneous confidence
band, indicating that the ranks are consistent with a uniform distribution.

```{jupyter-execute}

plot_ecdf_pit(sbc.simulations,
              group="posterior_sbc",
              visuals={"xlabel": False},
);
```

## Intentionally Skewing the Augmented Posterior Using Custom augmentation with `augment_observed`

We intentionally skew the augmented posterior by keeping only the last 25 original observations and concatenating them with the replicated data. This creates a mismatch between the reference draw (which is based on the full observed data) and the augmented posterior (which is based on a subset of the observed data), leading to skewed rank statistics.

```{jupyter-execute}

def augment_observed(model, observed_data, replicated_data, simulation_idx):
    """Keep only the last 25 original observations + replicated."""
    data = {"y": np.concatenate([observed_data["y"].values[-25:], replicated_data["y"]])}
    return data


def update_data(model, augmented_data, simulation_idx):
    with model:
        pm.set_data(
            {
                "x_data": np.concatenate(
                    [model["x_data"].get_value()[-25:], model["x_data"].get_value()]
                )
            },
            coords={"obs_id": np.arange(25 + len(model["x_data"].get_value()))},
        )


skewed_sbc = simuk.SBC(
    model,
    method="posterior",
    trace=idata,
    augment_observed=augment_observed,
    update_data=update_data,
    num_simulations=50,
    sample_kwargs={"chains": 4, "draws": 50, "tune": 50},
    progress_bar=False,
)

skewed_sbc.run_simulations()
```

### Visualize the skewed results

The results indicate a clear deviation from uniformity, with the ECDF lines falling outside the confidence band. This suggests that the self-consistency property of Bayesian updating does not hold.

```{jupyter-execute}

plot_ecdf_pit(skewed_sbc.simulations, group="posterior_sbc", visuals={"xlabel": False})
```

We shall also replot the original Posterior SBC results for comparison using `compute_rank_statistics` without need to re-run the simulations.

```{jupyter-execute}

sbc.compute_rank_statistics(lambda _, param_value: param_value)
plot_ecdf_pit(sbc.simulations, group="posterior_sbc", visuals={"xlabel": False})
```

## References

- Säilynoja, T., Schmitt, M., Bürkner, P.-C., & Vehtari, A. (2025).
  *Posterior SBC: Simulation-Based Calibration Checking Conditional on Data*.
  [arXiv:2502.03279](https://arxiv.org/abs/2502.03279)
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2020).
  *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*.
  [arXiv:1804.06788](https://arxiv.org/abs/1804.06788)
