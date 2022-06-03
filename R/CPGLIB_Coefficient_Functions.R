#' 
#' @title Coefficients for CPGLIB Object
#' 
#' @description \code{coef.CPGLIB} returns the coefficients for a CPGLIB object.
#' 
#' @method coef CPGLIB
#'
#' @param object An object of class CPGLIB.
#' @param groups The groups in the ensemble for the coefficients. Default is all of the groups in the ensemble.
#' @param ensemble_average Option to return the average of the coefficients over all the groups in the ensemble. Default is FALSE.
#' @param ... Additional arguments for compatibility.
#' 
#' @return The coefficients for the CPGLIB object.
#' 
#' @export
#' 
#' @author Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
#' 
#' @seealso \code{\link{cpg}}
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
#' # CPGLIB - Multiple Groups
#' cpg.out <- cpg(x.train, y.train,
#'                glm_type="Logistic",
#'                G=5, include_intercept=TRUE,
#'                alpha_s=3/4, alpha_d=1,
#'                lambda_sparsity=0.01, lambda_diversity=1,
#'                tolerance=1e-5, max_iter=1e5)
#'                
#' # Coefficients for each group                
#' cpg.coef <- coef(cpg.out, ensemble_average = FALSE)
#' }
#' 
coef.CPGLIB <- function(object, groups = NULL, ensemble_average = FALSE, ...){
  
  # Check input data
  if(!any(class(object) %in% "CPGLIB"))
    stop("The object should be of class \"CPGLIB\"")
  
  # Checking groups
  if(is.null(groups))
    groups <- 1:object$G else if(!is.null(groups) && !all(groups %in% (1:object$G)))
      stop("The groups specified are not valid.")
  
  # Return of coefficients
  if(!ensemble_average)
    return(rbind(t(object$Intercept[groups,]), object$Betas[,groups, drop=FALSE])) else
      return(apply(rbind(t(object$Intercept[groups,]), object$Betas[,groups, drop=FALSE]), 1, mean))
}

#' 
#' @title Coefficients for cv.CPGLIB Object
#' 
#' @method coef cv.CPGLIB
#' 
#' @description \code{coef.cv.CPGLIB} returns the coefficients for a cv.CPGLIB object.
#'
#' @param object An object of class cv.CPGLIB.
#' @param groups The groups in the ensemble for the coefficients. Default is all of the groups in the ensemble.
#' @param ensemble_average Option to return the average of the coefficients over all the groups in the ensemble. Default is FALSE.
#' @param ... Additional arguments for compatibility.
#' 
#' @return The coefficients for the cv.CPGLIB object. Default is FALSE.
#' 
#' @export
#' 
#' @author Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
#' 
#' @seealso \code{\link{cv.cpg}}
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
#' mean(y.test)
#' 
#' # CV CPGLIB - Multiple Groups
#' cpg.out <- cv.cpg(x.train, y.train,
#'                   glm_type = "Logistic",
#'                   G = 5, include_intercept = TRUE,
#'                   alpha_s = 3/4, alpha_d = 1,
#'                   n_lambda_sparsity = 100, n_lambda_diversity = 100,
#'                   tolerance = 1e-5, max_iter = 1e5)
#' cpg.coef <- coef(cpg.out)
#' 
#' # Coefficients for each group                
#' cpg.coef <- coef(cpg.out, ensemble_average = FALSE)
#' 
#' }
#' 
#' 
coef.cv.CPGLIB <- function(object, groups = NULL, ensemble_average = FALSE, ...){
  
  # Check input data
  if(!any(class(object) %in% "cv.CPGLIB"))
    stop("The object should be of class \"cv.CPGLIB\"")
  
  # Checking groups
  if(is.null(groups))
    groups <- 1:object$G else if(!is.null(groups) && !all(groups %in% (1:object$G)))
      stop("The groups specified are not valid.")
  
  # Extracting coefficients 
  if(length(groups)==1)
    extracted.coef <- as.matrix(c(object$Intercept[groups, object$Optimal_Index, drop=TRUE], 
                        object$Betas[, groups, object$Optimal_Index, drop=TRUE]),
                        ncol=1) else
      extracted.coef <- rbind(t(object$Intercept[groups, object$Optimal_Index, drop=TRUE]), 
                              object$Betas[, groups, object$Optimal_Index, drop=TRUE])
  
  # Return of coefficients
  if(!ensemble_average)
    return(extracted.coef) else
      return(apply(extracted.coef, 1, mean))
}