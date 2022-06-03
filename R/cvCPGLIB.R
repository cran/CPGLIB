#' 
#' @title Competing Proximal Gradients Library for Ensembles of Generalized Linear Models - Cross-Validation
#' 
#' @description \code{cv.cpg} computes and cross-validates the coefficients for ensembles of generalized linear models via competing proximal gradients.
#' 
#' @param x Design matrix.
#' @param y Response vector.
#' @param glm_type Description of the error distribution and link function to be used for the model. Must be one of "Linear" or
#' "Logistic". Default is "Linear".
#' @param G Number of groups in the ensemble.
#' @param full_diversity Argument to determine if the overlap between the models should be zero. Default is FALSE.
#' @param include_intercept Argument to determine whether there is an intercept. Default is TRUE.
#' @param alpha_s Sparsity mixing parmeter. Default is 3/4.
#' @param alpha_d Diversity mixing parameter. Default is 1.
#' @param n_lambda_sparsity Number of candidates for sparsity tuning parameter. Default is 100.
#' @param n_lambda_diversity Number of candidates for diveristy tuning parameter. Default is 100.
#' @param tolerance Convergence criteria for the coefficients. Default is 1e-8.
#' @param max_iter Maximum number of iterations in the algorithm. Default is 1e5.
#' @param n_folds Number of cross-validation folds. Default is 10.
#' @param n_threads Number of threads. Default is a single thread.
#' 
#' @return An object of class \code{cv.cpg}
#' 
#' @export
#' 
#' @author Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
#' 
#' @seealso \code{\link{coef.cv.CPGLIB}}, \code{\link{predict.cv.CPGLIB}}
#' 
#' @examples 
#' \donttest{
#' # Data simulation
#' set.seed(1)
#' n <- 50
#' N <- 2000
#' p <- 300
#' beta.active <- c(abs(runif(p, 0, 1/2))*(-1)^rbinom(p, 1, 0.3))
#' # Parameters
#' p.active <- 150
#' beta <- c(beta.active[1:p.active], rep(0, p-p.active))
#' Sigma <- matrix(0, p, p)
#' Sigma[1:p.active, 1:p.active] <- 0.5
#' diag(Sigma) <- 1
#' 
#' # Train data
#' x.train <- mvnfast::rmvn(n, mu = rep(0, p), sigma = Sigma) 
#' prob.train <- exp(x.train %*% beta)/
#'               (1+exp(x.train %*% beta))
#' y.train <- rbinom(n, 1, prob.train)
#' # Test data
#' x.test <- mvnfast::rmvn(N, mu = rep(0, p), sigma = Sigma)
#' prob.test <- exp(x.test %*% beta)/
#'              (1+exp(x.test %*% beta))
#' y.test <- rbinom(N, 1, prob.test)
#' 
#' # CV CPGLIB - Multiple Groups
#' cpg.out <- cv.cpg(x.train, y.train,
#'                   glm_type = "Logistic",
#'                   G = 5, include_intercept = TRUE,
#'                   alpha_s = 3/4, alpha_d = 1,
#'                   n_lambda_sparsity = 100, n_lambda_diversity = 100,
#'                   tolerance = 1e-5, max_iter = 1e5)
#' 
#' # Predictions
#' cpg.prob <- predict(cpg.out, newx = x.test, type = "prob", 
#'                     groups = 1:cpg.out$G, ensemble_type = "Model-Avg")
#' cpg.class <- predict(cpg.out, newx = x.test, type = "class", 
#'                      groups = 1:cpg.out$G, ensemble_type = "Model-Avg")
#' plot(prob.test, cpg.prob, pch = 20)
#' abline(h = 0.5,v = 0.5)
#' mean((prob.test-cpg.prob)^2)
#' mean(abs(y.test-cpg.class))
#' 
#' }
#' 

cv.cpg <- function(x, y, 
                   glm_type = c("Linear", "Logistic")[1], 
                   G = 5,
                   full_diversity = FALSE,
                   include_intercept=TRUE, 
                   alpha_s = 3/4, alpha_d = 1,
                   n_lambda_sparsity = 100, n_lambda_diversity = 100,
                   tolerance = 1e-8, max_iter = 1e5,
                   n_folds = 10,
                   n_threads = 1){
  
  # Check response data
  y <- Check_Response_CPGLIB(y, glm_type)

  # Check data
  Check_Data_CV_CPGLIB(x, y,
                       glm_type,
                       G,
                       alpha_s, alpha_d,
                       n_lambda_sparsity, n_lambda_diversity,
                       tolerance, max_iter,
                       n_folds,
                       n_threads)
  
  # Shuffling the data
  n <- nrow(x)
  random.permutation <- sample(1:n, n)
  x.permutation <- x[random.permutation, ]
  y.permutation <- y[random.permutation]
  
  # Setting the model type
  type.cpp <- switch(glm_type,
                     "Linear" = 1,
                     "Logistic" = 2)
  
  # Setting to return fully diverse models for CPP computation
  full_diversity.cpp <- sum(full_diversity)
  
  # Setting to include intercept parameter for CPP computation
  include_intercept.cpp <- sum(include_intercept)
  
  # Source code computation
  cpg.out <- CV_CPGLIB_Main(x.permutation, y.permutation, 
                            type.cpp, 
                            G,
                            full_diversity.cpp,
                            include_intercept.cpp, 
                            alpha_s, alpha_d,
                            n_lambda_sparsity, n_lambda_diversity,
                            tolerance, max_iter,
                            n_folds,
                            n_threads)
  
  # Object construction
  cpg.out <- construct.cv.CPGLIB(cpg.out, match.call(), glm_type, G, n_lambda_sparsity, n_lambda_diversity)
  
  # Return source code output
  return(cpg.out)
}


