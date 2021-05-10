#' 
#' @title Coefficients for ProxGrad Object
#' 
#' @description \code{coef.ProxGrad} returns the coefficients for a ProxGrad object.
#' 
#' @method coef ProxGrad
#'
#' @param object An object of class ProxGrad.
#' @param ... Additional arguments for compatibility.
#' 
#' @return The coefficients for the ProxGrad object.
#' 
#' @export
#' 
#' @author Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
#' 
#' @seealso \code{\link{ProxGrad}}
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
#' # ProxGrad - Single Group
#' proxgrad.out <- ProxGrad(x.train, y.train,
#'                          glm_type = "Logistic",
#'                          include_intercept = TRUE,
#'                          alpha_s = 3/4,
#'                          lambda_sparsity = 0.01, 
#'                          acceleration = TRUE,
#'                          tolerance = 1e-5, max_iter = 1e5)
#' 
#' # Coefficients
#' coef(proxgrad.out)
#' 
#' }
#' 
coef.ProxGrad <- function(object, ...){
  
  # Check input data
  if(!any(class(object) %in% "ProxGrad"))
    stop("The object should be of class \"ProxGrad\"")
  
  return(object$coef)
}

#' 
#' @title Coefficients for cv.ProxGrad Object
#' 
#' @method coef cv.ProxGrad
#' 
#' @description \code{coef.cv.ProxGrad} returns the coefficients for a cv.ProxGrad object.
#'
#' @param object An object of class cv.ProxGrad.
#' @param ... Additional arguments for compatibility.
#' 
#' @return The coefficients for the cv.ProxGrad object.
#' 
#' @export
#' 
#' @author Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
#' 
#' @seealso \code{\link{cv.ProxGrad}}
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
#' # CV ProxGrad - Single Group
#' proxgrad.out <- cv.ProxGrad(x.train, y.train,
#'                             glm_type = "Logistic",
#'                             include_intercept = TRUE,
#'                             alpha_s = 3/4,
#'                             n_lambda_sparsity = 100, 
#'                             acceleration = TRUE,
#'                             tolerance = 1e-5, max_iter = 1e5)
#' 
#' # Coefficients
#' coef(proxgrad.out)
#' 
#' }
#' 
coef.cv.ProxGrad <- function(object, ...){
  
  # Check input data
  if(!any(class(object) %in% "cv.ProxGrad"))
    stop("The object should be of class \"cv.ProxGrad\"")
  
  return(object$coef)
}