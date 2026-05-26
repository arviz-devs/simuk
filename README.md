# Simuk

Simuk is a Python library for simulation-based calibration (SBC) and the generation of synthetic data.
Simulation-Based Calibration is a method for validating Bayesian inference by checking whether the posterior distributions align with the expected theoretical results derived from the prior.

Simuk works with [PyMC](http://docs.pymc.io), [Bambi](https://bambinos.github.io/bambi/) and [NumPyro](https://num.pyro.ai/en/latest/index.html) models.

## Installation

May be pip installed from github:

```bash
pip install simuk
```

## Quickstart

### Prior SBC

1. Define a PyMC or Bambi model. For example, the centered eight schools model:

    ```python
    import numpy as np
    import pymc as pm
    from arviz_plots import plot_ecdf_pit

    data = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as centered_eight:
        mu = pm.Normal('mu', mu=0, sigma=5)
        tau = pm.HalfCauchy('tau', beta=5)
        theta = pm.Normal('theta', mu=mu, sigma=tau, shape=8)
        y_obs = pm.Normal('y', mu=theta, sigma=sigma, observed=data)
    ```
2. Pass the model to the `SBC` class, and run the simulations. This will take a while, as it is running the model many times.
    ```python
    sbc = SBC(centered_eight,
            num_simulations=100, # ideally this should be higher, like 1000
            sample_kwargs={'draws': 25, 'tune': 50})

    sbc.run_simulations()
    ```
    ```python
    79%|███████▉  | 79/100 [05:36<01:29,  4.27s/it]
    ```

3. Plot the empirical CDF for the difference between prior and posterior. The lines
should be close to uniform and within the oval envelope.

    ```python
    plot_ecdf_pit(sbc.simulations,
                visuals={"xlabel":False},
    );
    ```

![Prior Simulation based calibration plots, ecdf](prior_ecdf.png)

### Posterior SBC

Posterior SBC evaluates validity locally, conditional on observed data. It is
currently implemented for PyMC. This requires storing observed data in
`pm.Data` containers, using `dims` instead of static shapes, and resizing
covariates and coords in an `update_data` callback to match the augmented data.

1. Define the model with `pm.Data` and `dims`:

    ```python
    import numpy as np
    import pymc as pm

    data = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model(coords={"school": np.arange(8)}) as centered_eight:
        school_idx = pm.Data("school_idx", np.arange(8))
        y_data = pm.Data("y_data", data)
        sigma_data = pm.Data("sigma_data", sigma)

        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, dims="school")
        y_obs = pm.Normal("y", mu=theta[school_idx], sigma=sigma_data, observed=y_data)
    ```

2. Sample once to obtain the original trace:

    ```python
    with centered_eight:
        idata = pm.sample(progressbar=False)
    ```

3. Define `update_data` to resize covariates and run Posterior SBC:

    ```python
    import simuk
    from arviz_plots import plot_ecdf_pit

    def update_data(model, augmented_data, simulation_idx):
        with model:
            pm.set_data({
                "sigma_data": np.concatenate([sigma, sigma]),
                "school_idx": np.concatenate([np.arange(8), np.arange(8)])
            })

    post_sbc = simuk.SBC(
        centered_eight,
        method="posterior",
        trace=idata,
        update_data=update_data,
        num_simulations=50,
        sample_kwargs={"draws": 25, "tune": 50},
        progress_bar=False
    )
    post_sbc.run_simulations()

    plot_ecdf_pit(post_sbc.simulations, group="posterior_sbc", visuals={"xlabel": False})
    ```

![Posterior Simulation based calibration plots, ecdf](posterior_ecdf.png)

## References

- Talts, S., Betancourt, M., Simpson, D., Vehtari A., and Gelman A. (2018). [Validating Bayesian Inference Algorithms with Simulation-Based Calibration](https://doi.org/10.48550/arXiv.1804.06788).
- Modrák, M., Moon, A, Kim, S., Bürkner, P., Huurre, N., Faltejsková, K., Gelman A and Vehtari, A.(2023). [Simulation-based calibration checking for Bayesian computation: The choice of test quantities shapes sensitivity](https://doi.org/10.1214/23-BA1404). Bayesian Analysis.
- Säilynoja, T., Marvin Schmitt, Paul-Christian Bürkner and Aki Vehtari (2025). [Posterior SBC: Simulation-Based Calibration Checking Conditional on Data](https://doi.org/10.48550/arXiv.2502.03279).
