
<!-- README.md is generated from README.Rmd. Please edit that file -->

# R/`dsldensify`

> **Discrete Super Learner Conditional Density Estimation**

**Author:**\
David Benkeser (Emory University)

------------------------------------------------------------------------

## What is `dsldensify`?

`dsldensify` is an R package for **cross-validated conditional density
estimation** of a continuous variable $A \mid W$, using a **discrete
Super Learner–style model selection framework**.

The package is designed for settings where:

- the full conditional distribution of $A \mid W$ is required (not just
  a mean),
- flexible machine-learning estimators are desired,
- and model selection is performed by **cross-validated negative
  log-density risk**.

The project is under active development and should currently be viewed
as **research-grade software** rather than a fully stable production
library.

------------------------------------------------------------------------

## Modeling strategies supported

At a high level, `dsldensify` supports **three complementary
approaches** to conditional density estimation under a common
cross-validation interface.

### 1. Hazard-based (discretized) density estimation

In this approach, the support of $A$ is discretized into bins, and the
conditional density is estimated by modeling **discrete-time hazards**:

- $A$ is binned using either equal-range or equal-mass binning.
- Long-format data are constructed with one row per observation–bin
  pair.
- Binary regression models estimate the hazard of falling into each bin.
- Estimated hazards are converted into a continuous density.

This strategy allows the use of a wide range of binary learners (e.g.,
logistic regression, penalized GLMs, random forests, gradient boosting).

### 2. Direct conditional density models

Direct models estimate $f(A \mid W)$ without discretization by
specifying a parametric or semi-parametric likelihood for the outcome,
for example:

- homoscedastic Gaussian regression,
- flexible GAMLSS models with covariate-dependent parameters,
- mixture-of-experts models with component-specific means and variances.

### 3. Quantile inversion methods

Quantile inversion methods represent the conditional distribution of
$A \mid W$ through its conditional quantile function $Q(p \mid W)$,
where $Q(p \mid W)$ denotes the $p$-th conditional quantile of $A$ given
$W$.

In this approach:

- Conditional quantile functions $Q(p \mid W)$ are estimated on a fixed
  grid of probability levels $p \in (p_{\min}, p_{\max})$ using quantile
  regression.
- The conditional density is recovered via inversion of the quantile
  function, using the identity
  $f(a \mid W) = \left( \frac{d}{dp} Q(p \mid W) \right)^{-1}
    \Bigg|_{p : Q(p \mid W) = a}.$
- Numerical differentiation and smoothing are used to stabilize
  estimates of the derivative $dQ(p \mid W)/dp$, and monotonicity of the
  quantile function.

For the time being, only parametric quantile regression is supported.

------------------------------------------------------------------------

## Hurdle Models

Hurdle models combine a point-mass learner for
$\pi(W) = P(A = 0 \mid W)$ with a positive-part density for $A > 0$. The
conditional distribution of $A$ is modeled as a two-part mixture:
$f(a \mid W = w) =   \pi(w)\,\mathbb{I}\{a = a_0\} +   \left( 1 - \pi(w)\right) f_+(a \mid w)\,\mathbb{I}\{a \ne a_0\} \ .$

If hurdle learners are specified for $\pi$, the function then uses the
supplied direct and/or hazard learners for the positive-part density
estimation. Direct density learners inputted in this way should respect
the requirement for positive bounds.

------------------------------------------------------------------------

## Installation

The development version of `dsldensify` can be installed from GitHub:

``` r
remotes::install_github("benkeser/dsldensify")
```

------------------------------------------------------------------------

## Example

Below is a simple example using synthetic data. We combine:

- a hazard-based learner using gradient boosting, and
- a direct Gaussian conditional density model.

A simple example illustrates how `haldensify` may be used to train a
discrete super learner on a small data set. In this example, we leverage
a hazard-based density learner that uses `xgboost` and a simple, direct
density estimate based on a homoscedastic, Gaussian linear model.

``` r
library(data.table)
library(dsldensify)
#> dsldensify v0.0.1: Discrete Super Learner Conditional Density Estimation

# generate data
set.seed(1)

n <- 200
W <- data.table(
  x1 = rnorm(n),
  x2 = rnorm(n)
)

A <- 0.7 * W$x1 - 0.3 * W$x2 + rnorm(n)

# define xgboost hazard learner
# see ?make_xgboost_runner for details
hazard_learners <- list(
  xgb = make_xgboost_runner(
    rhs_list = list(~ x1 + x2 + bin_id),
    max_depth_grid = c(2, 4),
    eta_grid = c(0.1, 0.05),
    nrounds_max = 100,
    early_stopping_rounds = 10
  )
)

# define Gaussian linear model direct density learner
# see ?make_gaussian_homosced_direct_runner for details
direct_learners <- list(
  gaussian = make_gaussian_homosced_direct_runner(
    rhs_list = list(~ x1 + x2)
  )
)

fit <- dsldensify(
  A = A,
  W = W,
  hazard_learners = hazard_learners,
  direct_learners = direct_learners,
  grid_type = c("equal_mass"),
  n_bins = c(5, 10),
  cv_folds = 3
)

fit
#> dsldensify fit
#>   grid_type:  direct
#>   learner:    gaussian
#>   best_tune:  1
#>   is_direct:  TRUE
#> 
#>   tune_row:
#>  .tune     rhs
#>      1 x1 + x2
#> 
#>   fits:
#>     full_fit: present
#>     cv_fit:   NULL/empty
#> 
#> 3-fold CV risk:    1.49741
```

------------------------------------------------------------------------

## Visualizing fitted densities

We can visualize the fit by plotting the conditional density estimates
for fixed covariate profiles.

``` r
W_grid <- data.table(
  x1 = c(0, 1),
  x2 = c(0, 0)
)

plot(fit, W_grid, mode = "conditional")
```

<img src="README-example-plot1-1.png" alt="" width="80%" />

We can also sanity check the fit by comparing the marginal density
estimate implied by the conditional model to a simple histogram.

``` r
hist(
  A, freq = FALSE, breaks = 30, col = "gray", border = "white",
  main = "Marginal distribution of A"
)
plot(fit, W_grid, mode = "marginal", add = TRUE, plot_args = list(lwd = 2))
```

<img src="README-example-plot2-1.png" alt="" width="80%" />

------------------------------------------------------------------------

## Prediction

We can obtain conditional density estimates from the trained model on
arbitrary new data.

``` r
A_new <- seq(-3, 3, length.out = 100)
W_new <- data.table(x1 = 0, x2 = 0)

dens <- predict(fit, A = A_new, W = W_new)
head(dens)
#> [1] 0.007923187 0.009265375 0.010799867 0.012547760 0.014531364 0.016774087
```

Cross-validated prediction is also supported (e.g., for DML/TMLE
workflows); see the function documentation for details.

------------------------------------------------------------------------

## Sampling from the fitted conditional density

In addition to evaluating densities, `dsldensify` can generate draws
from the fitted conditional distribution $\hat f(\cdot \mid W)$.

``` r
# draw 5 samples of A for each row of W_grid
A_samp <- rsample(fit, W = W_grid, n_samp = 5, type = "full", seed = 123)

A_samp
#>            [,1]      [,2]       [,3]       [,4]       [,5]
#> [1,] -0.6401785 1.6157852 0.09410472  0.4471377 -0.7747125
#> [2,]  0.4893415 0.8094347 2.56013685 -0.6123373  0.2599489
```

A quick sanity check is to compare the sampled values to the fitted
conditional density curve at the same covariate profile.

``` r
# overlay samples on the fitted conditional density for W_new = (0, 0)
W_new <- data.table(x1 = 0, x2 = 0)
A_samp0 <- rsample(fit, W = W_new, n_samp = 2000, type = "full", seed = 1)

hist(
  A_samp0[1, ], freq = FALSE, breaks = 30, col = "gray", border = "white",
  main = "Samples from estimated density + fitted density curve"
)

plot(
  fit, W_new, mode = "conditional", add = TRUE, plot_args = list(lwd = 2)
)
```

<img src="README-example-sample-1.png" alt="" width="80%" />

Cross-validated sampling is also supported; see the function
documentation for details.

------------------------------------------------------------------------

## Hurdle model example

In this example, we illustrate a simple hurdle model.

``` r
set.seed(2)

n_h <- 300
W_h <- data.table(
  x1 = rnorm(n_h),
  x2 = rnorm(n_h)
)

p_zero <- plogis(-0.5 + 0.8 * W_h$x1 - 0.4 * W_h$x2)
is_zero <- rbinom(n_h, size = 1, prob = p_zero)
A_pos <- exp(0.5 + 0.6 * W_h$x1 - 0.2 * W_h$x2 + rnorm(n_h, sd = 0.4))
A_h <- ifelse(is_zero == 1, 0, A_pos)

hurdle_learners <- list(
  glm = make_glm_hurdle_runner(
    rhs_list = list(~ x1 + x2)
  )
)

positive_learners <- list(
  lognormal = make_lognormal_homosced_pos_runner(
    rhs_list = list(~ x1 + x2)
  )
)

fit_hurdle <- dsldensify(
  A = A_h,
  W = W_h,
  hurdle_learners = hurdle_learners,
  direct_learners = positive_learners,
  cv_folds = 3
)
```

``` r
W_h_grid <- data.table(
  x1 = 0,
  x2 = 0
)

op <- par(mfrow = c(1, 2))
plot(
  fit_hurdle,
  W_h_grid,
  mode = "conditional",
  component = "hurdle",
  main = "Hurdle model"
)
plot(
  fit_hurdle,
  W_h_grid,
  mode = "conditional",
  component = "positive",
  main = "Positive-part density"
)
```

<img src="README-example-hurdle-plot-1.png" alt="" width="100%" />

------------------------------------------------------------------------

## Supported learners

All estimators in `dsldensify` are implemented as **runners**: adapters
that expose a common interface for fitting, prediction, tuning, and
cross-validation. This design allows hazard-based and direct density
learners to compete under the same cross-validated log-density risk and
allows for building out additional learners under this same framework.

### Hazard-based learners

| Learner                            | Runner function               |
|------------------------------------|-------------------------------|
| Logistic regression (`stats::glm`) | `make_glm_hazard_runner()`    |
| Penalized logistic (`glmnet`)      | `make_glmnet_hazard_runner()` |
| Gradient boosting (`xgboost`)      | `make_xgboost_runner()`       |
| Random forest (`ranger`)           | `make_rf_hazard_runner()`     |

### Direct density learners

| Learner | Runner function |
|----|----|
| Gaussian linear model, homoscedastic | `make_gaussian_homosced_direct_runner()` |
| GAMLSS (`gamlss`) | `make_gamlss_direct_runner()` |
| Bounded GAMLSS (`gamlss`) | `make_bounded_gamlss_direct_runner()` |
| Gaussian mixture of experts (EM, optional gating) | `make_gmm_direct_runner()` |
| Location–scale model with residual KDE | `make_locscale_kde_direct_runner()` |
| Mixture of location–scale residual KDE experts | `make_mix_locscale_kde_direct_runner()` |
| Quantile-function–based density (quantile regression) | `make_quantreg_direct_runner()` |

### Positive-part learners

| Learner | Runner function |
|----|----|
| Gamma GLM with log link | `make_gamma_glm_log_pos_runner()` |
| Log-normal homoscedastic direct model | `make_lognormal_homosced_pos_runner()` |

### Hurdle learners

| Learner                            | Runner function                |
|------------------------------------|--------------------------------|
| Logistic regression (`stats::glm`) | `make_glm_hurdle_runner()`     |
| Penalized logistic (`glmnet`)      | `make_glmnet_hurdle_runner()`  |
| Random forest (`ranger`)           | `make_rf_hurdle_runner()`      |
| Gradient boosting (`xgboost`)      | `make_xgboost_hurdle_runner()` |

------------------------------------------------------------------------

## Status

The API is still evolving and internal representations may change.

Unit tests are minimal for the time being.

Despite this, the package already supports:

- multiple learner backends,
- multiple binning strategies,
- attempts at memory-efficient cross-validation,

A package vignette is (probably) coming soon.

------------------------------------------------------------------------

## Issues

If you encounter any bugs or have any specific feature requests, please
[file an issue](https://github.com/benkeser/dsldensify/issues).

------------------------------------------------------------------------

## Contributions

Contributions are welcome. Please see [contribution
guidelines](https://github.com/benkeser/dsldensify/blob/master/CONTRIBUTING.md)
prior to submitting a pull request.

------------------------------------------------------------------------

<!-- ## Citation -->

## Related

- [R/`haldensify`](https://github.com/nhejazi/haldensify) – Earlier work
  that inspired parts of the hazard-based density machinery.

------------------------------------------------------------------------

## License

© 2019-2026 David Benkeser

The contents of this repository are distributed under the MIT license.
