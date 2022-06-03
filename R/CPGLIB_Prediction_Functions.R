#' 
#' @title Predictions for CPGLIB Object
#' 
#' @description \code{predict.CPGLIB} returns the predictions for a CPGLIB object.
#' 
#' @method predict CPGLIB
#'
#' @param object An object of class CPGLIB.
#' @param newx New data for predictions.
#' @param groups The groups in the ensemble for the predictions. Default is all of the groups in the ensemble.
#' @param ensemble_type The type of ensembling function for the models. Options are "Model-Avg", "Coef-Avg" or "Weighted-Prob" for 
#' classifications predictions. Default is "Model-Avg".
#' @param class_type The type of predictions for classification. Options are "prob" and "class". Default is "prob".
#' @param ... Additional arguments for compatibility.
#' 
#' @return The predictions for the CPGLIB object.
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
#'                glm_type = "Logistic",
#'                G = 5, include_intercept = TRUE,
#'                alpha_s = 3/4, alpha_d = 1,
#'                lambda_sparsity = 0.01, lambda_diversity = 1,
#'                tolerance = 1e-5, max_iter = 1e5)
#' 
#' # Predictions
#' cpg.prob <- predict(cpg.out, newx = x.test, type = "prob", 
#'                     groups = 1:cpg.out$G, ensemble_type = "Model-Avg")
#' cpg.class <- predict(cpg.out, newx = x.test, type = "prob", 
#'                      groups = 1:cpg.out$G, ensemble_type = "Model-Avg")
#' plot(prob.test, cpg.prob, pch=20)
#' abline(h=0.5,v=0.5)
#' mean((prob.test-cpg.prob)^2)
#' mean(abs(y.test-cpg.class))
#' 
#' }
#' 
predict.CPGLIB <- function(object, newx, 
                           groups = NULL,
                           ensemble_type = c("Model-Avg", "Coef-Avg", "Weighted-Prob", "Majority-Vote")[1], 
                           class_type = c("prob", "class")[1], 
                           ...){
  
  # Check input data
  if(!any(class(object) %in% "CPGLIB"))
    stop("The object should be of class \"CPGLIB\"")
  
  # Checking groups
  if(is.null(groups))
    groups <- 1:object$G else if(!is.null(groups) && !all(groups %in% (1:object$G)))
      stop("The groups specified are not valid.")
  
  # Check ensemble function
  if(!any(ensemble_type %in% c("Model-Avg", "Coef-Avg", "Weighted-Prob", "Majority-Vote")))
    stop("The argument \"ensemble_type\" must be one of \"Model-Avg\", \"Coef-Avg\", \"Weighted-Prob\" or \"Majority-Vote\".")
  
  # Argument compability
  if(object$glm_type!="Logistic" && any(ensemble_type %in% c("Weighted-Prob", "Majority-Vote")))
    stop("The \"ensemble_type\" argument is incompatible with the GLM type.") else{
      if((ensemble_type %in% c("Weighted-Prob", "Majority-Vote")) && class_type=="prob")
        stop("The options \"Weighted-Prob\" or \"Majority-Vote\" must have the argument \"class_type\" set to \"class\".")
    }
  
  if(object$glm_type=="Linear"){ # LINEAR MODEL
    
    cpg.coef <- coef(object, groups=groups, ensemble_average=TRUE)
    return(cpg.coef[1] + newx %*% cpg.coef[-1])
    
  } else if(object$glm_type=="Logistic"){ # LOGISTIC MODEL
    
    if(!(class_type %in% c("prob", "class")))
      stop("The variable \"type\" must be one of: \"prob\", or \"class\".")
    
    if(ensemble_type=="Model-Avg"){
      
      cpg.coef <- coef(object)
      
      logistic.prob <- sapply(groups, function(cpg.coef, x) 
        return(exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x])/(1+exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x]))),
        cpg.coef=cpg.coef)
      
      logistic.prob <- apply(logistic.prob, 1, mean)
      
      if(class_type=="prob")
        return(logistic.prob) else if(class_type=="class")
          return(round(logistic.prob, 0))
      
    } else if(ensemble_type=="Coef-Avg"){
      
      cpg.coef <- coef(object, ensemble_average=TRUE)
      
      logistic.prob <- exp(cpg.coef[1] + newx %*% cpg.coef[-1])/(1+exp(cpg.coef[1] + newx %*% cpg.coef[-1]))
      
      if(class_type=="prob")
        return(logistic.prob) else if(class_type=="class")
          return(round(logistic.prob, 0))
      
    } else if(ensemble_type=="Weighted-Prob"){
      
      cpg.coef <- coef(object)
      
      logistic.prob <- sapply(groups, function(cpg.coef, x) 
        return(exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x])/(1+exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x]))),
        cpg.coef=cpg.coef)
      
      return(as.numeric(apply(logistic.prob, 1, function(x) return(prod(x)>prod(1-x)))))
      
    } else if(ensemble_type=="Majority-Vote"){
      
      cpg.coef <- coef(object)
      
      logistic.prob <- sapply(groups, function(cpg.coef, x) 
        return(exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x])/(1+exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x]))),
        cpg.coef=cpg.coef)
      
      return(as.numeric(apply(2*round(logistic.prob, 0), 1, mean)>=1))
    }
    
  } else if(object$glm_type=="Gamma"){ # GAMMA MODEL
    
    if(ensemble_type=="Model-Avg"){
      
      cpg.coef <- coef(object)
      
      gamma.predictions <- sapply(groups, function(x, cpg.coef)
        exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x]),
        cpg.coef=cpg.coef)
      
      return(apply(gamma.predictions, 1, mean))
      
    } else if(ensemble_type=="Coef-Avg"){
      
      cpg.coef <- coef(object, groups=groups, ensemble_average=TRUE)
      return(exp(cpg.coef[1] + newx %*% cpg.coef[-1]))
    }

  } else if(object$glm_type=="Poisson"){ # POISSON MODEL
    
    if(ensemble_type=="Model-Avg"){
      
      cpg.coef <- coef(object)
      
      poisson.predictions <- sapply(groups, function(x, cpg.coef)
        exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x]),
        cpg.coef=cpg.coef)
      
      return(apply(poisson.predictions, 1, mean))
      
    } else if(ensemble_type=="Coef-Avg"){
      
      cpg.coef <- coef(object, groups=groups, ensemble_average=TRUE)
      return(exp(cpg.coef[1] + newx %*% cpg.coef[-1]))
    }
  }
}
#'
#' @title Predictions for cv.ProxGrad Object
#' 
#' @description \code{predict.cv.CPGLIB} returns the predictions for a ProxGrad object.
#' 
#' @method predict cv.CPGLIB
#'
#' @param object An object of class cv.CPGLIB.
#' @param newx New data for predictions.
#' @param groups The groups in the ensemble for the predictions. Default is all of the groups in the ensemble.
#' @param ensemble_type The type of ensembling function for the models. Options are "Model-Avg", "Coef-Avg" or "Weighted-Prob" for 
#' classifications predictions. Default is "Model-Avg".
#' @param class_type The type of predictions for classification. Options are "prob" and "class". Default is "prob".
#' @param ... Additional arguments for compatibility.
#' 
#' @return The predictions for the cv.CPGLIB object.
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
#' 
predict.cv.CPGLIB <- function(object, newx, 
                              groups = NULL,
                              ensemble_type = c("Model-Avg", "Coef-Avg", "Weighted-Prob", "Majority-Vote")[1], 
                              class_type = c("prob", "class")[1], 
                              ...){
  
  # Check input data
  if(!any(class(object) %in% "cv.CPGLIB"))
    stop("The object should be of class \"cv.CPGLIB\"")
  
  # Checking groups
  if(is.null(groups))
    groups <- 1:object$G else if(!is.null(groups) && !all(groups %in% (1:object$G)))
      stop("The groups specified are not valid.")
  
  # Check ensemble function
  if(!any(ensemble_type %in% c("Model-Avg", "Coef-Avg", "Weighted-Prob", "Majority-Vote")))
    stop("The argument \"ensemble_type\" must be one of \"Model-Avg\", \"Coef-Avg\", \"Weighted-Prob\" or \"Majority-Vote\".")
  
  # Argument compability
  if(object$glm_type!="Logistic" && any(ensemble_type %in% c("Weighted-Prob", "Majority-Vote")))
    stop("The \"ensemble_type\" argument is incompatible with the GLM type.") else{
      if((ensemble_type %in% c("Weighted-Prob", "Majority-Vote")) && class_type=="prob")
        stop("The options \"Weighted-Prob\" or \"Majority-Vote\" must have the argument \"class_type\" set to \"class\".")
    }
  
  if(object$glm_type=="Linear"){ # LINEAR MODEL
    
    cpg.coef <- coef(object, groups=groups, ensemble_average=TRUE)
    return(cpg.coef[1] + newx %*% cpg.coef[-1])
    
  } else if(object$glm_type=="Logistic"){ # LOGISTIC MODEL
    
    if(!(class_type %in% c("prob", "class")))
      stop("The variable \"type\" must be one of: \"prob\", or \"class\".")
    
    if(ensemble_type=="Model-Avg"){
      
      cpg.coef <- coef(object)
      
      logistic.prob <- sapply(groups, function(cpg.coef, x) 
        return(exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x])/(1+exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x]))),
        cpg.coef=cpg.coef)
      
      logistic.prob <- apply(logistic.prob, 1, mean)
      
      if(class_type=="prob")
        return(logistic.prob) else if(class_type=="class")
          return(round(logistic.prob, 0))
      
    } else if(ensemble_type=="Coef-Avg"){
      
      cpg.coef <- coef(object, groups=groups, ensemble_average=TRUE)
      
      logistic.prob <- exp(cpg.coef[1] + newx %*% cpg.coef[-1])/(1+exp(cpg.coef[1] + newx %*% cpg.coef[-1]))
      
      if(class_type=="prob")
        return(logistic.prob) else if(class_type=="class")
          return(round(logistic.prob, 0))
      
    } else if(ensemble_type=="Weighted-Prob"){
      
      cpg.coef <- coef(object)
      
      logistic.prob <- sapply(groups, function(cpg.coef, x) 
        return(exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x])/(1+exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x]))),
        cpg.coef=cpg.coef)
      
      return(as.numeric(apply(logistic.prob, 1, function(x) return(prod(x)>prod(1-x)))))
      
    } else if(ensemble_type=="Majority-Vote"){
      
      cpg.coef <- coef(object)
      
      logistic.prob <- sapply(groups, function(cpg.coef, x) 
        return(exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x])/(1+exp(cpg.coef[1,x] + newx %*% cpg.coef[-1,x]))),
        cpg.coef=cpg.coef)
      
      return(as.numeric(apply(2*round(logistic.prob, 0), 1, mean)>=1))
    }
    
  } 
}