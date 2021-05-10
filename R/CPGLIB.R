#' 
#' @title Competing Proximal Gradients Library for Ensembles of Generalized Linear Models
#' 
#' @description \code{cpg} computes the coefficients for ensembles of generalized linear models via competing proximal gradients.
#' 
#' @param x Design matrix.
#' @param y Response vector.
#' @param glm_type Description of the error distribution and link function to be used for the model. Must be one of "Linear", 
#' "Logistic", "Gamma" or "Poisson". Default is "Linear".
#' @param G Number of groups in the ensemble.
#' @param include_intercept Argument to determine whether there is an intercept. Default is TRUE.
#' @param alpha_s Sparsity mixing parmeter. Default is 3/4.
#' @param alpha_d Diversity mixing parameter. Default is 1.
#' @param lambda_sparsity Sparsity tuning parameter value.
#' @param lambda_diversity Diversity tuning parameter value.
#' @param balanced_cycling Argument to determine the cycling strategy for the optimal solution search. Default is TRUE.
#' @param permutate_search Argument to determine whether permutations are used to search for the optimal solution. Default is FALSE.
#' @param acceleration Argument to determine whether a gradient acceleration method is used. Default is FALSE.
#' @param tolerance Convergence criteria for the coefficients. Default is 1e-3.
#' @param max_iter Maximum number of iterations in the algorithm. Default is 1e5.
#' 
#' @return An object of class \code{cpg}
#' 
#' @export
#' 
#' @author Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
#' 
#' @seealso \code{\link{coef.CPGLIB}}, \code{\link{predict.CPGLIB}}
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
#' x.train <- mvnfast::rmvn(n, mu = rep(0, p), sigma  =  Sigma) 
#' prob.train <- exp(x.train %*% beta)/
#'               (1+exp(x.train %*% beta))
#' y.train <- rbinom(n, 1, prob.train)
#' # Test data
#' x.test <- mvnfast::rmvn(N, mu = rep(0, p), sigma  =  Sigma)
#' prob.test <- exp(x.test %*% beta)/
#'              (1+exp(x.test %*% beta))
#' y.test <- rbinom(N, 1, prob.test)
#' 
#' # CPGLIB - Multiple Groups
#' cpg.out <- cpg(x.train, y.train,
#'                glm_type = "Logistic",
#'                G = 5, include_intercept = TRUE,
#'                alpha_s = 3/4, alpha_d = 1,
#'                lambda_sparsity = 0.01, lambda_diversity = 1,
#'                balanced_cycling = TRUE,
#'                tolerance = 1e-5, max_iter = 1e5)
#' 
#' # Predictions
#' cpg.prob <- predict(cpg.out, newx = x.test, type = "prob", 
#'                     groups = 1:cpg.out$G, ensemble_type = "Model-Avg")
#' cpg.class <- predict(cpg.out, newx = x.test, type = "prob", 
#'                      groups = 1:cpg.out$G, ensemble_type = "Model-Avg")
#' plot(prob.test, cpg.prob, pch = 20)
#' abline(h = 0.5,v = 0.5)
#' mean((prob.test-cpg.prob)^2)
#' mean(abs(y.test-cpg.class))
#' 
#' }
#' 

cpg <- function(x, y, 
                glm_type = c("Linear", "Logistic", "Gamma", "Poisson")[1], 
                G = 5,
                include_intercept = TRUE, 
                alpha_s = 3/4, alpha_d = 1,
                lambda_sparsity, lambda_diversity,
                balanced_cycling = TRUE,
                permutate_search = FALSE,
                acceleration = FALSE,
                tolerance = 1e-5, max_iter = 1e5){
  
  # Check response data
  y <- Check_Response_CPGLIB(y, glm_type)

  # Check data
  Check_Data_ProxGrad(x, y,
                      glm_type,
                      alpha_s,
                      lambda_sparsity,
                      tolerance, max_iter)
  
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
  
  # Setting of permutation search for CPP computation
  permutate_search.cpp <- sum(permutate_search)
  
  # Setting of cycling search for CPP computation
  balanced_cycling.cpp <- sum(balanced_cycling)
  
  # Source code computation
  cpg.out <- CPGLIB_Main(x.permutation, y.permutation, 
                         type.cpp, 
                         G,
                         include_intercept.cpp, 
                         alpha_s, alpha_d,
                         lambda_sparsity, lambda_diversity,
                         balanced_cycling.cpp,
                         acceleration.cpp,
                         permutate_search.cpp,
                         tolerance, max_iter)
  
  # Object construction
  cpg.out <- construct.CPGLIB(cpg.out, match.call(), glm_type, G, lambda_sparsity, lambda_diversity)
  
  # Return source code output
  return(cpg.out)
}


