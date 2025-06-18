Overview
========

Simuk is a Python library for simulation-based calibration (SBC) and the generation of synthetic data.
Simulation-Based Calibration (SBC) is a method for validating Bayesian inference by checking whether the
posterior distributions align with the expected theoretical results derived from the prior.

Quickstart
----------

This quickstart guide provides a simple example to help you get started. If you're looking for more examples
and use cases, be sure to check out the :doc:`examples` section.

To use SBC, you need to define a model function that generates simulated data and corresponding prior predictive
samples, then compare them to posterior samples obtained through inference.

In our case, we will take a PyMC model and pass it into our ``SBC`` class.

.. code-block:: python

    from arviz_plots import plot_ecdf_pit, style
    import numpy as np
    import pymc as pm

    data = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as centered_eight:
        mu = pm.Normal('mu', mu=0, sigma=5)
        tau = pm.HalfCauchy('tau', beta=5)
        theta = pm.Normal('theta', mu=mu, sigma=tau, shape=8)
        y_obs = pm.Normal('y', mu=theta, sigma=sigma, observed=data)

    # Pass it into the SBC class
    sbc = simuk.SBC(centered_eight, num_simulations=100, sample_kwargs={'draws': 25, 'tune': 50})

Now, we use the ``run_simulations`` method to generate and analyze simulated data, running the model multiple times to
compare prior and posterior distributions.

.. code-block:: python

    sbc.run_simulations()

Plot the empirical CDF to compare the differences between the prior and posterior.

.. code-block:: python

    plot_ecdf_pit(sbc.simulations,
                visuals={"xlabel":False},
    );

The lines should be nearly uniform and fall within the oval envelope. It suggests that the prior and posterior distributions
are properly aligned and that there are no significant biases or issues with the model.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   Overview <self>
   installation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API documentation

   api/index.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: References

   contributing
   changelog

References
----------

- Talts, Sean, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman. 2018. “Validating Bayesian Inference Algorithms with Simulation-Based Calibration.” `arXiv:1804.06788 <https://doi.org/10.48550/arXiv.1804.06788>`_.
- Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N., Faltejsková, K., … & Vehtari, A. (2023). Simulation-based calibration checking for Bayesian computation: The choice of test quantities shapes sensitivity. Bayesian Analysis, advance publication, DOI: `10.1214/23-BA1404 <https://projecteuclid.org/journals/bayesian-analysis/volume--1/issue--1/Simulation-Based-Calibration-Checking-for-Bayesian-Computation--The-Choice/10.1214/23-BA1404.full>`_
