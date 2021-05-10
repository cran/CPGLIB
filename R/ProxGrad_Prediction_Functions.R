#' 
#' @title Predictions for ProxGrad Object
#' 
#' @description \code{predict.ProxGrad} returns the predictions for a ProxGrad object.
#' 
#' @method predict ProxGrad
#'
#' @param object An object of class ProxGrad
#' @param newx New data for predictions.
#' @param type The type of predictions for binary response. Options are "prob" (default) and "class".
#' @param ... Additional arguments for compatibility.
#' 
#' @return The predictions for the ProxGrad object.
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
predict.ProxGrad <- function(object, newx, type = c("prob", "class")[1], ...){
  
  # Check input data
  if(!any(class(object) %in% "ProxGrad"))
    stop("The object should be of class \"ProxGrad\"")
  
  split.coef <- coef(object)
  
  if(object$glm_type=="Linear"){
    
    return(split.coef[1] + newx %*% split.coef[-1])
    
  } else if(object$glm_type=="Logistic"){
    
    if(!(type %in% c("prob", "class")))
      stop("The variable \"type\" must be one of: \"prob\", or \"class\".")
    
    logistic.prob <- exp(split.coef[1] + newx %*% split.coef[-1])/(1+exp(split.coef[1] + newx %*% split.coef[-1]))
    
    if(type=="prob")
      return(logistic.prob) else if(type=="class")
        return(round(logistic.prob, 0))
    
  } else if(object$glm_type=="Gamma"){
    
    return(-1/(split.coef[1] + newx %*% split.coef[-1]))
    
  } else if(object$glm_type=="Poisson"){
    
    return(exp(split.coef[1] + newx %*% split.coef[-1]))
    
  }
  
}
#'
#' @title Predictions for cv.ProxGrad Object
#' 
#' @description \code{predict.cv.ProxGrad} returns the predictions for a ProxGrad object.
#' 
#' @method predict cv.ProxGrad
#'
#' @param object An object of class cv.ProxGrad.
#' @param newx New data for predictions.
#' @param type The type of predictions for binary response. Options are "prob" (default) and "class".
#' @param ... Additional arguments for compatibility.
#' 
#' @return The predictions for the cv.ProxGrad object.
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
predict.cv.ProxGrad <- function(object, newx, type = c("prob", "class")[1], ...){
  
  # Check input data
  if(!any(class(object) %in% "cv.ProxGrad"))
    stop("The object should be of class \"cv.ProxGrad\".")
  
  split.coef <- coef(object)
  
  if(object$glm_type=="Linear"){
    
    return(split.coef[1] + newx %*% split.coef[-1])
    
  } else if(object$glm_type=="Logistic"){
    
    if(!(type %in% c("prob", "class")))
      stop("The variable \"type\" must be one of: \"prob\", or \"class\".")
    
    logistic.prob <- exp(split.coef[1] + newx %*% split.coef[-1])/(1+exp(split.coef[1] + newx %*% split.coef[-1]))
    
    if(type=="prob")
      return(logistic.prob) else if(type=="class")
        return(round(logistic.prob, 0))
    
  } else if(object$glm_type=="Gamma"){
    
    return(-1/(split.coef[1] + newx %*% split.coef[-1]))
    
  } else if(object$glm_type=="Poisson"){
    
    return(exp(split.coef[1] + newx %*% split.coef[-1]))
    
  }
  
}