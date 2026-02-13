#' Create a GLM runner for discrete-time hazard modeling with frozen spline bases
#'
#' Constructs a runner (learner adapter) compatible with the
#' run_grid_setting() / summarize_and_select() workflow used in dsl_densify.
#' The runner fits one or more logistic regression models via stats::glm.fit()
#' on long-format discrete-time hazard data with binary outcome in_bin.
#'
#' Tuning is performed over a list of RHS model specifications (rhs_list);
#' exactly one model is fit per RHS. Each fitted model estimates per-bin
#' discrete-time hazards
#'   P(T in bin_j | T >= bin_j, W)
#' using a logistic likelihood, which is equivalent to maximizing the
#' discrete-time hazard likelihood under the long-data construction used
#' by dsl_densify.
#'
#' This runner closely mirrors the conventions of the xgboost hazard runner:
#' prediction and sampling share a common hazard-scoring path, sampling
#' assumes post-selection fits (K = 1), and sampling operates exclusively
#' on long hazard data constructed upstream.
#'
#' Spline handling
#'
#' Natural spline terms may be specified directly in the RHS using ns() or
#' splines::ns(), subject to the following restrictions:
#'   - Only ns(<symbol>, df = ...) is supported; the first argument must be
#'     a bare variable name.
#'   - splines::ns() is accepted but internally rewritten to ns().
#'   - Knot locations and boundary knots are computed once on the training
#'     data within each fold and then frozen.
#'
#' Frozen spline bases are enforced at prediction and sampling time by
#' evaluating model.matrix() in an environment where ns() is replaced by a
#' wrapper that injects the stored knot information. This guarantees
#' consistency of spline bases within each fold and between training,
#' validation, and sampling.
#'
#' Numeric-only requirement
#'
#' This runner is intended for use with numeric predictors only. All variables
#' referenced in rhs_list (including bin_id and all covariates in W) must
#' already be numeric. Factors, characters, and ordered factors are not
#' supported and are not coerced internally.
#'
#' Tuning grid and prediction layout
#'
#' The runner constructs a tune_grid with one row per RHS specification and
#' columns .tune and rhs. During cross-validation, predict() returns an
#' n_long x K matrix of predicted hazards, where K = length(rhs_list), with
#' columns aligned to the ordering of tune_grid.
#'
#' Prediction delegates to an internal predict_hazards() helper so that
#' prediction and sampling always use identical hazard estimates.
#'
#' Sampling from the fitted hazard model
#'
#' The runner provides a sample() method that generates draws
#'   A* ~ f_hat(· | W)
#' from the implied conditional density under the discrete-time hazard
#' representation.
#'
#' Sampling assumes the fit_bundle contains exactly one fitted model
#' (length(fit_bundle$fits) == 1). This is the intended usage after model
#' selection, for example via select_fit_tune() or fit_one().
#'
#' IMPORTANT: The sample() method expects newdata in long hazard format.
#' Expansion of wide W to long form (repeating rows across all bins and
#' attaching bin_lower and bin_upper) is handled upstream by
#' sample.dsldensify(). The runner itself never constructs hazard grids.
#'
#' For each subject, sampling proceeds by:
#'   - predicting hazards h_j for all bins,
#'   - computing implied bin masses
#'       p_j = h_j * prod_{l < j} (1 - h_l),
#'   - normalizing the masses to sum to one,
#'   - sampling a bin index according to p_j,
#'   - sampling uniformly within the selected bin.
#'
#' If the total mass is non-finite or non-positive for any subject, a single
#' warning is issued and sampling for those subjects falls back to uniform
#' sampling over bins.
#'
#' Lightweight fit objects
#'
#' When strip_fit = TRUE, each fitted model is reduced to a minimal
#' representation sufficient for prediction and sampling:
#'   - a numeric coefficient vector aligned to frozen design columns,
#'   - the inverse link function (linkinv),
#'   - the training column names.
#'
#' When strip_fit = FALSE, full glm objects are stored; however, prediction
#' and sampling still use frozen design matrices and extracted coefficients
#' to preserve spline consistency.
#'
#' @param rhs_list A list of RHS specifications, either as one-sided formulas
#'   (for example, ~ W1 + ns(bin_id, df = 5)) or as character strings
#'   (for example, "W1 + splines::ns(bin_id, df = 5)").
#'
#' @param use_weights_col Logical. If TRUE and the training data contain a
#'   column named wts, it is passed as case weights to stats::glm.fit().
#'
#' @param strip_fit Logical. If TRUE (default), store a lightweight
#'   coefficient-based representation of each fitted model. If FALSE, store
#'   full glm objects.
#'
#' @param ... Additional arguments forwarded to stats::glm.fit() when
#'   strip_fit = TRUE and to stats::glm() when strip_fit = FALSE.
#'
#' @return A named list (runner) with the following elements:
#'   method: Character string "glm".
#'   tune_grid: Data frame describing the tuning grid, including .tune and rhs.
#'   fit: Function fit(train_set, ...) returning a fit bundle.
#'   predict: Function predict(fit_bundle, newdata, ...) returning an
#'     n_long x K matrix of predicted hazards.
#'   fit_one: Function fit_one(train_set, tune, ...) fitting only the selected
#'     tuning index.
#'   sample: Function sample(fit_bundle, newdata, n_samp, ...) drawing samples
#'     from the implied conditional density (assumes K = 1).
#'
#' Data requirements
#'
#' The runner expects train_set and newdata in long hazard format, including:
#'   - a binary outcome column in_bin,
#'   - a time-bin column bin_id,
#'   - a subject identifier column obs_id,
#'   - covariates referenced in rhs_list,
#'   - an optional weight column wts.
#'
#' newdata passed to sample() must additionally include bin_lower and bin_upper.
#'
#' @examples
#' rhs_list <- list(
#'   ~ W1 + W2 + splines::ns(bin_id, df = 4),
#'   ~ (W1 + W2) * splines::ns(bin_id, df = 4)
#' )
#'
#' runner <- make_glm_runner(
#'   rhs_list = rhs_list,
#'   use_weights_col = TRUE,
#'   strip_fit = TRUE
#' )
#'
#' @export

make_glm_hazard_runner <- function(
  rhs_list,
  use_weights_col = TRUE,
  strip_fit = TRUE,
  ...
) {
  stopifnot(requireNamespace("splines", quietly = TRUE))
  stopifnot(requireNamespace("data.table", quietly = TRUE))

  # Accept RHS formulas (~ ...) or RHS strings
  if (is.list(rhs_list) && all(vapply(rhs_list, inherits, logical(1), "formula"))) {
    rhs_chr <- vapply(rhs_list, function(f) paste(deparse(f[[2]]), collapse = ""), character(1))
  } else {
    rhs_chr <- as.character(rhs_list)
  }
  if (length(rhs_chr) < 1L) stop("rhs_list must have length >= 1")

  tune_grid <- data.frame(
    .tune = seq_along(rhs_chr),
    rhs = rhs_chr,
    stringsAsFactors = FALSE
  )

  # ---- hazard conventions (match dsldensify + xgboost runner) -------------
  id_col <- "obs_id"
  bin_var <- "bin_id"
  eps <- 1e-15
  clip01 <- function(p) pmin(pmax(p, eps), 1 - eps)

  # --- helpers (spline freezing) ------------------------------------------

  normalize_rhs <- function(rhs) gsub("splines::ns", "ns", rhs, fixed = TRUE)

  find_ns_calls <- function(expr) {
    out <- list()
    rec <- function(e) {
      if (is.call(e)) {
        fn <- e[[1L]]
        if (is.symbol(fn) && as.character(fn) == "ns") out[[length(out) + 1L]] <<- e
        for (j in seq_along(e)[-1L]) rec(e[[j]])
      }
    }
    rec(expr)
    out
  }

  compute_spline_specs <- function(rhs, train_set) {
    
    f <- stats::as.formula(paste0("in_bin ~ ", rhs))
    rhs_expr <- f[[3L]]
    calls <- find_ns_calls(rhs_expr)

    specs <- list()

    for (cl in calls) {
      if (length(cl) < 2L || !is.symbol(cl[[2L]])) {
        stop("Only ns(<variable>, df=...) supported. Found: ", paste(deparse(cl), collapse = ""))
      }
      xvar <- as.character(cl[[2L]])
      if (!(xvar %in% names(train_set))) stop("Spline variable '", xvar, "' not found in data for RHS: ", rhs)
      if (!is.numeric(train_set[[xvar]])) stop("Spline variable '", xvar, "' must be numeric.")

      argn <- names(as.list(cl))
      if (!("df" %in% argn)) stop("ns(", xvar, ", ...) must specify df= for RHS: ", rhs)
      df <- eval(cl[["df"]], envir = train_set, enclos = parent.frame())

      intercept_flag <- FALSE
      if ("intercept" %in% argn) {
        intercept_flag <- isTRUE(eval(cl[["intercept"]], envir = train_set, enclos = parent.frame()))
      }

      key <- paste(xvar, df, as.integer(intercept_flag), sep = "|")
      if (is.null(specs[[key]])) {
        B <- splines::ns(train_set[[xvar]], df = df, intercept = intercept_flag)
        specs[[key]] <- list(
          xvar = xvar,
          df = df,
          intercept = intercept_flag,
          knots = attr(B, "knots"),
          Boundary.knots = attr(B, "Boundary.knots")
        )
      }
    }

    specs
  }

  make_ns_env <- function(specs, parent_env) {
    env <- new.env(parent = parent_env)

    env$ns <- function(x, df = NULL, intercept = FALSE, ...) {
      x_name <- deparse(substitute(x))
      key <- paste(x_name, df, as.integer(isTRUE(intercept)), sep = "|")
      sp <- specs[[key]]
      if (is.null(sp)) {
        stop("No frozen spline spec found for ns(", x_name, ", df=", df,
             ", intercept=", intercept, ").")
      }
      splines::ns(
        x,
        df = sp$df,
        knots = sp$knots,
        Boundary.knots = sp$Boundary.knots,
        intercept = sp$intercept,
        ...
      )
    }

    env
  }

  build_design_train <- function(rhs_raw, train_set) {
    rhs <- normalize_rhs(rhs_raw)
    
    k <- get_spline_df(rhs_raw)
    probs <- seq_len(k - 1) / k
    qs <- quantile(train_set$bin_id, probs = probs)
    
    if(length(unique(qs)) < (k-1)) {
      # Edge case where requesting more knots in spline than possible with unique bins. Skip compute_spline_specs
      unique_bins <- unique(train_set$bin_id)
      f <- stats::as.formula(paste0("in_bin ~ -1 + ", paste0(lapply(unique_bins, function(x) paste0("I(bin_id == ", x, ")")), collapse = "+" ))) 
      
      specs <- NULL
      skip_bin_spline <- TRUE
    } else{
      # Regular case
      f <- stats::as.formula(paste0("in_bin ~ ", rhs))
      specs <- compute_spline_specs(rhs, train_set)
      env <- make_ns_env(specs, parent_env = environment(f))
      environment(f) <- env
      
      skip_bin_spline <- FALSE
    }
  
    tt_full <- stats::terms(f, data = train_set)
    tt_x    <- stats::delete.response(tt_full)

    # training X should also be built from tt_x (same columns, avoids carrying response forward)
    X <- stats::model.matrix(tt_x, data = train_set)

    list(
      X = X,
      terms = tt_x,
      design_spec = list(rhs = rhs, specs = specs, x_cols = colnames(X), skip_bin_spline = skip_bin_spline)
    )
  }

  build_design_new <- function(design_spec, newdata) {
    f <- stats::as.formula(paste0("~ ", design_spec$rhs))

    # run for most cases, only skips when edge case single bin_id in training fold
    if(!design_spec$skip_bin_spline){
      env <- make_ns_env(design_spec$specs, parent_env = environment(f))
      environment(f) <- env
    }
    
    tt <- stats::terms(f, data = newdata)
    Xn <- stats::model.matrix(tt, data = newdata)

    x_cols <- design_spec$x_cols
    missing <- setdiff(x_cols, colnames(Xn))
    if (length(missing)) {
      Xn <- cbind(Xn, matrix(0, nrow(Xn), length(missing), dimnames = list(NULL, missing)))
    }
    Xn <- Xn[, x_cols, drop = FALSE]
    Xn
  }


  # --- fit storage + prediction helpers -----------------------------------

  strip_glm_coef <- function(glm_fit, x_cols) {
    coefs <- stats::coef(glm_fit)
    coefs[is.na(coefs)] <- 0

    miss <- setdiff(x_cols, names(coefs))
    if (length(miss)) {
      coefs <- c(coefs, stats::setNames(rep(0, length(miss)), miss))
    }
    coefs <- coefs[x_cols]

    out <- list(
      coefficients = coefs,
      linkinv = glm_fit$family$linkinv,
      x_cols = x_cols
    )
    class(out) <- "glm_stripped"
    out
  }

  coef_from_glm <- function(glm_obj, x_cols) {
    coefs <- stats::coef(glm_obj)
    coefs[is.na(coefs)] <- 0

    miss <- setdiff(x_cols, names(coefs))
    if (length(miss)) {
      coefs <- c(coefs, stats::setNames(rep(0, length(miss)), miss))
    }
    coefs[x_cols]
  }

  predict_from_coef <- function(coefficients, linkinv, Xnew) {
    if (is.null(names(coefficients))) {
      stop("coefficients must be a named numeric vector (names should match training X colnames).")
    }

    # Treat rank-deficient coefficients as 0 contribution
    coefs <- coefficients
    coefs[is.na(coefs)] <- 0

    Xnew <- as.matrix(Xnew)
    cn <- colnames(Xnew)
    if (is.null(cn)) stop("Xnew must have colnames to align with coefficients.")

    # Ensure Xnew has exactly the coef columns, in the same order
    need <- names(coefs)

    # Add missing columns in Xnew (should be rare, but safe)
    miss <- setdiff(need, cn)
    if (length(miss) > 0L) {
      Z <- matrix(0, nrow = nrow(Xnew), ncol = length(miss))
      colnames(Z) <- miss
      Xnew <- cbind(Xnew, Z)
      cn <- colnames(Xnew)
    }

    # Drop any extra columns not in coefficients
    keep <- intersect(cn, need)
    Xnew <- Xnew[, need, drop = FALSE]  # reorder + subset to coef names

    eta <- as.numeric(Xnew %*% as.numeric(coefs))
    linkinv(eta)
  }


  # predict hazards for K fits (like xgboost::predict_hazards)
  predict_hazards <- function(fit_bundle, newdata, ...) {
    if (!data.table::is.data.table(newdata)) stop("newdata must be a data.table.")
    fits <- fit_bundle$fits
    if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")

    n <- nrow(newdata)
    out <- matrix(NA_real_, nrow = n, ncol = length(fits))

    # resolve RHS keys for each fit (prefer tune indices if present)
    rhs_keys <- NULL
    if (!is.null(fit_bundle$tune) && length(fit_bundle$tune) == length(fits)) {
      rhs_keys <- tune_grid$rhs[as.integer(fit_bundle$tune)]
    } else if (!is.null(fit_bundle$rhs_chr) && length(fit_bundle$rhs_chr) == length(fits)) {
      rhs_keys <- fit_bundle$rhs_chr
    } else {
      # fallback: assume ordering matches tune_grid
      rhs_keys <- tune_grid$rhs[seq_along(fits)]
    }

    for (k in seq_along(fits)) {
      rhs_raw <- rhs_keys[[k]]
      ds <- fit_bundle$design_specs[[rhs_raw]]
      if (is.null(ds)) stop("Missing design_specs for RHS: ", rhs_raw)

      Xn <- build_design_new(ds, as.data.frame(newdata))
      fit_k <- fits[[k]]

      p <- if (inherits(fit_k, "glm_stripped")) {
        predict_from_coef(fit_k$coefficients, fit_k$linkinv, Xn)
      } else {
        # full glm object: still predict via frozen X to preserve frozen spline basis
        coefs <- coef_from_glm(fit_k, ds$x_cols)
        predict_from_coef(coefs, fit_k$family$linkinv, Xn)
      }

      out[, k] <- clip01(p)
    }

    out
  }
  
  # helper to get degrees of freedom for spline from formula
  get_spline_df <- function(rhs_raw) {
    rhs <- normalize_rhs(rhs_raw)
    f <- stats::as.formula(paste0("~", rhs))
    
    # Find ns() call
    ns_call <- NULL
    
    for (term in attr(stats::terms(f), "term.labels")) {
      expr <- str2lang(term)
      if (is.call(expr) && grepl("^ns$", as.character(expr[[1]]))) {
        ns_call <- expr
        break
      }
    }
    
    if (is.null(ns_call)) {
      stop("No ns() term found in rhs")
    }
    
    # Extract df argument
    df_val <- NULL
    
    # Look for named argument df=
    arg_names <- names(ns_call)
    if ("df" %in% arg_names) {
      df_val <- eval(ns_call[["df"]])
    } else {
      stop("ns() call does not contain df argument")
    }
    
    as.integer(df_val)
  }
  

  # --- runner --------------------------------------------------------------

  list(
    method = "glm",
    tune_grid = tune_grid,
    positive_support = TRUE,
    fit = function(train_set, ...) {
      if (!("in_bin" %in% names(train_set))) stop("train_set must contain column 'in_bin'.")

      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      design_specs <- setNames(vector("list", length(rhs_chr)), rhs_chr)
      fits <- vector("list", nrow(tune_grid))

      for (k in seq_len(nrow(tune_grid))) {
        rhs_raw <- tune_grid$rhs[k]
        built <- build_design_train(rhs_raw, train_set)
        X <- built$X
        y <- as.numeric(train_set$in_bin)

        fit_k <- suppressWarnings(stats::glm.fit(
          x = X, y = y,
          weights = wts_vec,
          family = stats::binomial(),
          ...
        ))

        if (strip_fit) {
          fits[[k]] <- strip_glm_coef(fit_k, x_cols = built$design_spec$x_cols)
        } else {
          # store full glm object, but predictions will still use frozen design matrix
          fits[[k]] <- suppressWarnings(stats::glm(
            stats::as.formula(paste0("in_bin ~ ", normalize_rhs(rhs_raw))),
            data = train_set,
            family = stats::binomial(),
            weights = wts_vec,
            ...
          ))
        }

        design_specs[[rhs_raw]] <- built$design_spec
      }

      list(
        fits = fits,
        design_specs = design_specs,
        rhs_chr = rhs_chr,
        stripped = isTRUE(strip_fit)
      )
    },

    predict = function(fit_bundle, newdata, ...) {
      dt <- data.table::as.data.table(newdata)
      # keep consistent with hazard runners: use predict_hazards() for scoring
      predict_hazards(fit_bundle = fit_bundle, newdata = dt, ...)
    },

    fit_one = function(train_set, tune, ...) {
      k <- as.integer(tune)
      if (length(k) != 1L || is.na(k) || k < 1L || k > nrow(tune_grid)) {
        stop("tune must be a single integer in 1..", nrow(tune_grid))
      }

      rhs_raw <- tune_grid$rhs[k]

      if (!("in_bin" %in% names(train_set))) stop("train_set must contain column 'in_bin'.")
      has_wts <- use_weights_col && ("wts" %in% names(train_set))
      wts_vec <- if (has_wts) train_set$wts else NULL

      built <- build_design_train(rhs_raw, train_set)
      X <- built$X
      y <- as.numeric(train_set$in_bin)

      fit_k <- suppressWarnings(stats::glm.fit(
        x = X, y = y,
        weights = wts_vec,
        family = stats::binomial(),
        ...
      ))

      fit_store <- if (strip_fit) strip_glm_coef(fit_k, x_cols = built$design_spec$x_cols) else {
        suppressWarnings(stats::glm(
          stats::as.formula(paste0("in_bin ~ ", normalize_rhs(rhs_raw))),
          data = train_set,
          family = stats::binomial(),
          weights = wts_vec,
          ...
        ))
      }

      list(
        fits = list(fit_store),
        design_specs = setNames(list(built$design_spec), rhs_raw),
        rhs_chr = rhs_raw,
        stripped = isTRUE(strip_fit),
        tune = k
      )
    },

    # hazard-based sampling (match xgboost conventions)
    sample = function(fit_bundle, newdata, n_samp, seed = NULL, ...) {
      if (!data.table::is.data.table(newdata)) stop("newdata must be a data.table.")
      if (length(n_samp) != 1L || is.na(n_samp) || n_samp < 1L) stop("n_samp must be a positive integer.")
      n_samp <- as.integer(n_samp)

      fits <- fit_bundle$fits
      if (is.null(fits) || !length(fits)) stop("fit_bundle does not contain fits.")
      if (length(fits) != 1L) stop("sample() assumes K=1: fit_bundle$fits must have length 1 (selected model).")

      if (!("bin_lower" %in% names(newdata)) || !("bin_upper" %in% names(newdata))) {
        stop("newdata must contain 'bin_lower' and 'bin_upper' columns for sampling.")
      }
      if (!(id_col %in% names(newdata))) {
        stop("newdata must contain id_col='", id_col, "' for sampling.")
      }
      if (!(bin_var %in% names(newdata))) {
        stop("newdata must contain bin_var='", bin_var, "' for sampling.")
      }

      if (!is.null(seed)) set.seed(seed)

      # Predict hazards (n_long x 1) in the original newdata row order
      haz <- predict_hazards(fit_bundle = fit_bundle, newdata = newdata, ...)
      haz <- as.numeric(haz[, 1])

      dt <- data.table::as.data.table(newdata)
      dt[, .row_id__ := .I]
      data.table::setorderv(dt, c(id_col, bin_var))
      haz <- haz[dt$.row_id__]  # reorder hazards to match dt after sorting

      rows_by_id <- split(seq_len(nrow(dt)), dt[[id_col]])
      ids <- names(rows_by_id)

      out <- matrix(NA_real_, nrow = length(ids), ncol = n_samp)
      rownames(out) <- ids
      warned_zero_mass <- FALSE

      for (ii in seq_along(ids)) {
        rr <- rows_by_id[[ii]]
        m <- length(rr)
        if (m < 1L) next

        h <- clip01(haz[rr])

        lower <- dt$bin_lower[rr]
        upper <- dt$bin_upper[rr]
        if (any(!is.finite(lower)) || any(!is.finite(upper)) || any(upper <= lower)) {
          stop("Invalid bin_lower/bin_upper for id='", ids[[ii]], "'.")
        }

        # masses p_j = h_j * prod_{l<j}(1 - h_l)
        if (m == 1L) {
          mass <- h
        } else {
          s_prev <- c(1, cumprod(1 - h)[-m])
          mass <- h * s_prev
        }

        tot <- sum(mass)
        if (!is.finite(tot) || tot <= 0) {

          if (!warned_zero_mass) {
            warning(
              "Hazard-based sampling encountered zero or non-finite total mass for at least one observation.\n",
              "Falling back to uniform sampling over bins for those cases.\n",
              "This can occur if predicted hazards are numerically near zero across all bins\n",
              "or if survival past the grid has probability ~1."
            )
            warned_zero_mass <- TRUE
          }

          j <- sample.int(m, size = n_samp, replace = TRUE)

        } else {
          pmf <- mass / tot
          j <- sample.int(m, size = n_samp, replace = TRUE, prob = pmf)
        }

        out[ii, ] <- stats::runif(n_samp, min = lower[j], max = upper[j])
      }

      out
    }
  )
}
