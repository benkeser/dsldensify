#' Create a random forest runner for hurdle probability modeling
#'
#' Constructs a runner (learner adapter) for modeling the hurdle probability
#' \eqn{\pi(W) = P(A = a_0 \mid W)} using probabilistic random forests via
#' ranger::ranger(). The runner is compatible with the hurdle workflow in
#' dsldensify, where the hurdle component is fit on wide data with binary
#' outcome \code{in_hurdle}.
#'
#' Tuning is performed over a grid defined by:
#'   - multiple RHS feature specifications (rhs_list),
#'   - node-size parameters (min_node_size_grid),
#'   - optional feature-subset sizes (mtry_grid),
#'   - sampling behavior (bootstrap vs subsampling without replacement).
#'
#' RHS specifications (column selection only)
#'
#' The rhs_list argument defines feature sets by extracting variable names:
#'   - RHS may be provided as one-sided formulas (for example, ~ W1 + W2),
#'     or as character strings (for example, "W1 + W2").
#'   - Each RHS is converted to a formula internally and variable names are
#'     extracted via all.vars().
#'   - No transformations or interactions are evaluated; referenced columns
#'     must already exist in the wide data.
#'
#' Numeric-only requirement
#'
#' This runner is intended for use with numeric predictors only. All columns
#' referenced by rhs_list should already be numeric. Factors, characters, and
#' ordered factors are not supported and should be encoded upstream.
#'
#' @param rhs_list A list of RHS specifications, either one-sided formulas
#'   (for example, ~ W1 + W2) or character strings (for example, "W1 + W2").
#'
#' @param mtry_grid Optional integer vector of mtry values to tune over.
#'   If NULL, mtry is set per fit to floor(sqrt(p)) bounded to [1, p], where p
#'   is the number of selected features.
#'
#' @param min_node_size_grid Integer vector of minimum terminal node sizes
#'   passed to ranger::ranger(min.node.size = ...).
#'
#' @param num_trees Integer number of trees in each forest.
#'
#' @param sampling_grid Character vector specifying sampling schemes to include.
#'   Must be a subset of c("bootstrap", "subsample").
#'
#' @param subsample_fraction_grid Numeric vector of sampling fractions used
#'   only when sampling = "subsample".
#'
#' @param use_weights_col Logical. If TRUE and weights_col is present, weights
#'   are passed to ranger::ranger() via case.weights.
#'
#' @param weights_col Name of the weights column in the wide data.
#'
#' @param respect_unordered_factors Passed to ranger::ranger().
#'
#' @param importance Passed to ranger::ranger().
#'
#' @param seed Optional integer seed for deterministic fitting across tuning
#'   rows (set.seed(seed + .tune)).
#'
#' @param ... Additional arguments forwarded to ranger::ranger().
#'
#' @return A named list (runner) with the following elements:
#'   method: Character string "hurdle_rf".
#'   tune_grid: Data frame describing the tuning grid.
#'   fit: Function fit(train_set, ...) returning a fit bundle.
#'   fit_one: Function fit_one(train_set, tune, ...) fitting only the selected
#'     tuning index.
#'   logpi: Function logpi(fit_bundle, newdata, ...) returning log probabilities.
#'   log_density: Function log_density(fit_bundle, newdata, ...) returning
#'     Bernoulli negative log-likelihoods.
#'   sample: Function sample(fit_bundle, newdata, n_samp, ...) drawing hurdle
#'     indicators (assumes K = 1).
#'
#' Data requirements
#'
#' The runner expects train_set and newdata as wide data containing:
#'   - a binary outcome column in_hurdle,
#'   - covariates referenced in rhs_list,
#'   - an optional weight column wts (or weights_col).
#'
#' @examples
#' rhs_list <- list(~ W1 + W2)
#'
#' runner <- make_rf_hurdle_runner(
#'   rhs_list = rhs_list,
#'   min_node_size_grid = c(5L, 20L)
#' )
#'
#' @export
#' @importFrom ranger predict.ranger

make_rf_hurdle_runner <- function(
  rhs_list,
  mtry_grid = NULL,
  min_node_size_grid = c(5L, 20L, 50L),
  num_trees = 500L,
  sampling_grid = c("bootstrap", "subsample"),
  subsample_fraction_grid = c(0.6, 0.8),
  use_weights_col = TRUE,
  weights_col = "wts",
  respect_unordered_factors = "order",
  importance = "none",
  seed = NULL,
  ...
) {
  stopifnot(requireNamespace("ranger", quietly = TRUE))
  stopifnot(requireNamespace("data.table", quietly = TRUE))
  stopifnot(requireNamespace("stats", quietly = TRUE))

  # Fixed hurdle conventions (match make_hurdle_glm_runner)
  outcome_col <- "in_hurdle"

  # rhs parsing: formulas (~ ...) or strings
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  } else {
    rhs_chr <- as.character(rhs_list)
  }
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1")

  sampling_grid <- unique(as.character(sampling_grid))
  if (!all(sampling_grid %in% c("bootstrap", "subsample"))) {
    stop("sampling_grid must be subset of c('bootstrap','subsample')")
  }

  sampling_tbl <- do.call(
    rbind,
    lapply(sampling_grid, function(s) {
      if (s == "bootstrap") {
        data.frame(
          sampling = "bootstrap",
          replace = TRUE,
          sample_fraction = 1.0,
          stringsAsFactors = FALSE
        )
      } else {
        data.frame(
          sampling = "subsample",
          replace = FALSE,
          sample_fraction = as.numeric(subsample_fraction_grid),
          stringsAsFactors = FALSE
        )
      }
    })
  )

  tune_grid <- merge(
    expand.grid(
      rhs = rhs_chr,
      min_node_size = as.integer(min_node_size_grid),
      stringsAsFactors = FALSE,
      KEEP.OUT.ATTRS = FALSE
    ),
    sampling_tbl,
    by = NULL
  )

  if (!is.null(mtry_grid)) {
    tune_grid <- merge(
      tune_grid,
      data.frame(mtry = as.integer(mtry_grid), stringsAsFactors = FALSE),
      by = NULL
    )
  } else {
    tune_grid$mtry <- NA_integer_
  }

  tune_grid$.tune <- seq_len(nrow(tune_grid))

  rhs_to_cols <- function(rhs) setdiff(all.vars(stats::as.formula(paste0("~", rhs))), outcome_col)
  default_mtry <- function(p) max(1L, min(p, as.integer(floor(sqrt(p)))))

  # Strict probability checks: fail loudly if ranger gives non-probabilities.
  # We only "nudge" tiny float drift into [0,1] (like -1e-16, 1+1e-16).
  clamp01_strict <- function(p, tol = 1e-6) {
    p <- as.numeric(p)
    if (any(!is.finite(p))) stop("Non-finite probabilities returned by ranger.")
    if (any(p < -tol | p > 1 + tol)) {
      rg <- range(p, finite = TRUE)
      stop("ranger returned values outside [0,1]. Range: [", rg[1], ", ", rg[2], "].")
    }
    p[p < 0] <- 0
    p[p > 1] <- 1
    p
  }

  clip01 <- function(p, eps) pmin(pmax(p, eps), 1 - eps)

  # Predict pi(W) for all tuned fits in a fit bundle (returns n x K probs).
  # Uses the same pattern as make_rf_runner's predict_hazards() helper.
  predict_pi <- function(fits, newdata, eps = 1e-15, ...) {
    nd <- data.table::as.data.table(newdata)
    if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")

    n <- nrow(nd)
    out <- matrix(NA_real_, nrow = n, ncol = length(fits))

    for (k in seq_along(fits)) {
      cols <- fits[[k]]$cols
      df_new <- as.data.frame(nd[, cols, with = FALSE])

      pr <- ranger:::predict.ranger(
        fits[[k]]$model,
        data = df_new,
        type = "response",
        ...
      )$predictions

      # probability=TRUE => predictions should be an n x 2 matrix
      if (!(is.matrix(pr) || is.data.frame(pr))) {
        stop("Expected ranger to return class-probability matrix; got non-matrix predictions.")
      }
      if (!("1" %in% colnames(pr))) {
        stop("ranger probability predictions missing class '1'.")
      }

      p1 <- pr[, "1", drop = TRUE]
      p1 <- clamp01_strict(p1)
      out[, k] <- clip01(p1, eps)
    }

    out
  }

  list(
    method = "hurdle_rf",
    tune_grid = tune_grid,

    fit = function(train_set, ...) {
      dat <- data.table::as.data.table(train_set)
      if (!(outcome_col %in% names(dat))) stop("Missing outcome_col='", outcome_col, "' in train_set")
      if (use_weights_col && !(weights_col %in% names(dat))) stop("Missing weights_col='", weights_col, "' in train_set")

      fits <- vector("list", nrow(tune_grid))

      for (k in seq_len(nrow(tune_grid))) {
        tg <- tune_grid[k, , drop = FALSE]
        cols <- rhs_to_cols(tg$rhs)
        if (length(cols) < 1L) stop("RHS selects no columns: rhs='", tg$rhs, "'")

        # Enforce numeric-only predictors (you said enforced upstream, but be defensive).
        bad <- cols[!vapply(dat[, cols, with = FALSE], function(x) is.numeric(x) || is.integer(x) || is.logical(x), logical(1))]
        if (length(bad)) stop("Non-numeric predictors not allowed in hurdle RF: ", paste(bad, collapse = ", "))

        mtry_k <- tg$mtry
        if (is.na(mtry_k)) mtry_k <- default_mtry(length(cols))

        # Build df for ranger: outcome + selected cols only
        df <- as.data.frame(dat[, c(outcome_col, cols), with = FALSE])

        # Ensure factor outcome with levels c(0,1) so column '1' exists
        y <- df[[outcome_col]]
        if (is.logical(y)) y <- as.integer(y)
        if (is.numeric(y) || is.integer(y)) {
          y <- as.integer(y)
          if (anyNA(y) || any(!(y %in% c(0L, 1L)))) stop(outcome_col, " must be coded 0/1 with no NA.")
          df[[outcome_col]] <- factor(y, levels = c(0L, 1L))
        } else if (is.factor(y)) {
          # normalize factor to 0/1 levels if possible
          # (defensive: many user pipelines still store as factor)
          y2 <- as.integer(as.character(y))
          if (anyNA(y2) || any(!(y2 %in% c(0L, 1L)))) {
            stop(outcome_col, " factor outcome must correspond to 0/1.")
          }
          df[[outcome_col]] <- factor(y2, levels = c(0L, 1L))
        } else {
          stop(outcome_col, " must be numeric/integer/logical/factor representing 0/1.")
        }

        case_wts <- if (use_weights_col) dat[[weights_col]] else NULL
        if (!is.null(case_wts)) {
          if (!is.numeric(case_wts) || any(!is.finite(case_wts)) || any(case_wts < 0)) {
            stop("weights_col='", weights_col, "' must be nonnegative finite numeric.")
          }
        }

        if (!is.null(seed)) set.seed(as.integer(seed) + as.integer(tg$.tune))

        fit_obj <- ranger::ranger(
          formula = stats::as.formula(paste0(outcome_col, " ~ .")),
          data = df,
          num.trees = as.integer(num_trees),
          mtry = as.integer(mtry_k),
          min.node.size = as.integer(tg$min_node_size),
          probability = TRUE,
          classification = TRUE,
          respect.unordered.factors = respect_unordered_factors,
          importance = importance,
          case.weights = case_wts,
          replace = isTRUE(tg$replace),
          sample.fraction = as.numeric(tg$sample_fraction),
          write.forest = TRUE
        )

        fits[[k]] <- list(model = fit_obj, cols = cols, .tune = tg$.tune)
      }

      list(
        fits = fits,
        rhs_chr = rhs_chr,
        tune = seq_len(nrow(tune_grid))
      )
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..", nrow(tune_grid))
      }

      dat <- data.table::as.data.table(train_set)
      if (!(outcome_col %in% names(dat))) stop("Missing outcome_col='", outcome_col, "' in train_set")
      if (use_weights_col && !(weights_col %in% names(dat))) stop("Missing weights_col='", weights_col, "' in train_set")

      tg <- tune_grid[k, , drop = FALSE]
      cols <- rhs_to_cols(tg$rhs)
      if (length(cols) < 1L) stop("RHS selects no columns: rhs='", tg$rhs, "'")

      bad <- cols[!vapply(dat[, cols, with = FALSE], function(x) is.numeric(x) || is.integer(x) || is.logical(x), logical(1))]
      if (length(bad)) stop("Non-numeric predictors not allowed in hurdle RF: ", paste(bad, collapse = ", "))

      mtry_k <- tg$mtry
      if (is.na(mtry_k)) mtry_k <- default_mtry(length(cols))

      df <- as.data.frame(dat[, c(outcome_col, cols), with = FALSE])
      y <- df[[outcome_col]]
      if (is.logical(y)) y <- as.integer(y)
      if (is.numeric(y) || is.integer(y)) {
        y <- as.integer(y)
        if (anyNA(y) || any(!(y %in% c(0L, 1L)))) stop(outcome_col, " must be coded 0/1 with no NA.")
        df[[outcome_col]] <- factor(y, levels = c(0L, 1L))
      } else if (is.factor(y)) {
        y2 <- as.integer(as.character(y))
        if (anyNA(y2) || any(!(y2 %in% c(0L, 1L)))) stop(outcome_col, " factor outcome must correspond to 0/1.")
        df[[outcome_col]] <- factor(y2, levels = c(0L, 1L))
      } else {
        stop(outcome_col, " must be numeric/integer/logical/factor representing 0/1.")
      }

      case_wts <- if (use_weights_col) dat[[weights_col]] else NULL
      if (!is.null(case_wts)) {
        if (!is.numeric(case_wts) || any(!is.finite(case_wts)) || any(case_wts < 0)) {
          stop("weights_col='", weights_col, "' must be nonnegative finite numeric.")
        }
      }

      if (!is.null(seed)) set.seed(as.integer(seed) + as.integer(tg$.tune))

      fit_obj <- ranger::ranger(
        formula = stats::as.formula(paste0(outcome_col, " ~ .")),
        data = df,
        num.trees = as.integer(num_trees),
        mtry = as.integer(mtry_k),
        min.node.size = as.integer(tg$min_node_size),
        probability = TRUE,
        classification = TRUE,
        respect.unordered.factors = respect_unordered_factors,
        importance = importance,
        case.weights = case_wts,
        replace = isTRUE(tg$replace),
        sample.fraction = as.numeric(tg$sample_fraction),
        write.forest = TRUE
      )

      list(
        fits = list(list(model = fit_obj, cols = cols, .tune = tg$.tune)),
        rhs_chr = tg$rhs,
        tune = k
      )
    },

    # Negative log-likelihood (matches hurdle_glm_runner convention)
    log_density = function(fit_bundle, newdata, eps = 1e-15, ...) {
      nd <- data.table::as.data.table(newdata)
      if (!(outcome_col %in% names(nd))) {
        stop("hurdle rf runner requires `", outcome_col, "` column in newdata")
      }
      y <- as.integer(nd[[outcome_col]])
      if (anyNA(y) || any(!(y %in% c(0L, 1L)))) stop(outcome_col, " must be coded 0/1 with no NA.")

      p <- predict_pi(fits = fit_bundle$fits, newdata = nd, eps = eps, ...)

      # Bernoulli negative log-likelihood
      y_mat <- matrix(y, nrow = length(y), ncol = ncol(p))
      -(y_mat * log(p) + (1L - y_mat) * log1p(-p))
    },

    logpi = function(fit_bundle, newdata, eps = 1e-15, ...) {
      p <- predict_pi(fits = fit_bundle$fits, newdata = newdata, eps = eps, ...)
      log(p)
    },

    # Sample in_hurdle ~ Bernoulli(pi(W)); returns n x n_samp matrix
    # Assumes post-selection K=1.
    sample = function(fit_bundle, newdata, n_samp, seed = NULL, eps = 1e-15, ...) {
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
      n_samp <- as.integer(n_samp)
      if (!is.null(seed)) set.seed(seed)

      p_mat <- predict_pi(fits = fit_bundle$fits, newdata = newdata, eps = eps, ...)
      if (ncol(p_mat) != 1L) stop("sample() assumes K=1: fit_bundle must contain exactly one selected model.")

      p <- as.numeric(p_mat[, 1L])
      n <- length(p)
      matrix(stats::rbinom(n * n_samp, size = 1L, prob = p), nrow = n, ncol = n_samp)
    }
  )
}
