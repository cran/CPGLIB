#' 
#' @title Generalized Linear Models via Proximal Gradients - Cross-validation
#' 
#' @description \code{cv.ProxGrad} computes and cross-validates the coefficients for generalized linear models using accelerated proximal gradients.
#' 
#' @param x Design matrix.
#' @param y Response vector.
#' @param glm_type Description of the error distribution and link function to be used for the model. Must be one of "Linear", 
#' "Logistic", "Gamma" or "Poisson". Default is "Linear".
#' @param include_intercept Argument to determine whether there is an intercept. Default is TRUE.
#' @param alpha_s Elastic net mixing parmeter. Default is 3/4.
#' @param n_lambda_sparsity Sparsity tuning parameter value. Default is 100.
#' @param acceleration Argument to determine whether a gradient acceleration method is used. Default is FALSE.
#' @param tolerance Convergence criteria for the coefficients. Default is 1e-3.
#' @param max_iter Maximum number of iterations in the algorithm. Default is 1e5.
#' @param n_folds Number of cross-validation folds. Default is 10.
#' @param n_threads Number of threads. Default is a single thread.
#' 
#' @return An object of class cv.ProxGrad
#' 
#' @export
#' 
#' @author Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
#' 
#' @seealso \code{\link{coef.cv.ProxGrad}}, \code{\link{predict.cv.ProxGrad}}
#' 
#' @examples 
#' \donttest{
#' # Data simulation
#' set.seed(1)
#' n <- 50
#' N <- 2000
#' p <- 1000
#' beta.active <- c(abs(runif(p, 0, 1/2))*(-1)^rbinom(p, 1, 0.3))
#' # Parameters
#' p.active <- 100
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
#' # ProxGrad - Single Groups
#' proxgrad.out <- cv.ProxGrad(x.train, y.train,
#'                             glm_type = "Logistic",
#'                             include_intercept = TRUE,
#'                             alpha_s = 3/4, 
#'                             n_lambda_sparsity = 100, 
#'                             acceleration = TRUE,
#'                             tolerance = 1e-5, max_iter = 1e5)
#' 
#' # Predictions
#' proxgrad.prob <- predict(proxgrad.out, newx = x.test, type = "prob")
#' proxgrad.class <- predict(proxgrad.out, newx = x.test, type = "class")
#' plot(prob.test, proxgrad.prob, pch = 20)
#' abline(h = 0.5,v = 0.5)
#' mean((prob.test-proxgrad.prob)^2)
#' mean(abs(y.test-proxgrad.class))
#' 
#' }
#' 

cv.ProxGrad <- function(x, y, 
                        glm_type = c("Linear", "Logistic", "Gamma", "Poisson")[1], 
                        include_intercept=TRUE, 
                        alpha_s = 3/4,
                        n_lambda_sparsity = 100, 
                        acceleration = FALSE,
                        tolerance = 1e-3, max_iter = 1e5,
                        n_folds = 10,
                        n_threads = 1){
  
  # Check response data
  y <- Check_Response(y, glm_type)
  
  # Check data
  Check_Data_CV_ProxGrad(x, y, 
                         glm_type, 
                         alpha_s, 
                         n_lambda_sparsity, 
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
                     "Logistic" = 2,
                     "Gamma" = 3,
                     "Poisson" = 4)
  
  # Setting to include intercept parameter for CPP computation
  include_intercept.cpp <- sum(include_intercept)
  
  # Setting of acceleration for CPP computation
  acceleration.cpp <- sum(acceleration)
  
  # Source code computation
  cv.ProxGrad.out <- CV_ProxGrad_Main(x.permutation, y.permutation, 
                                      type.cpp, 
                                      include_intercept.cpp, 
                                      alpha_s,
                                      acceleration.cpp,
                                      n_lambda_sparsity,
                                      tolerance, max_iter,
                                      n_folds,
                                      n_threads)
  
  # # Object construction
  cv.ProxGrad.out <- construct.cv.ProxGrad(cv.ProxGrad.out, match.call(), glm_type, n_lambda_sparsity)
  
  # Return source code output
  return(cv.ProxGrad.out)
}


