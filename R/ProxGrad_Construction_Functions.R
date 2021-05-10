# -----------------------------------------------------------------------
# Object Construction for ProxGrad object
#
# object: the ProxGrad object
# fn_call: the function call
# -----------------------------------------------------------------------
construct.ProxGrad <- function(object, fn_call, glm_type, lambda_sparsity){
  
  class(object) <- append("ProxGrad", class(object))
  object$call <- fn_call
  
  object$glm_type <- glm_type
  
  object$lambda_sparsity <- lambda_sparsity
  
  object$coef <- c(object$Intercept, object$Betas)
  
  return(object)
}

# -----------------------------------------------------------------------
# Object Construction for cv.ProxGrad object
#
# object: the cv.ProxGrad object
# fn_call: the function call
# -----------------------------------------------------------------------
construct.cv.ProxGrad <- function(object, fn_call, glm_type, n_lambda_sparsity){
  
  class(object) <- append("cv.ProxGrad", class(object))
  object$call <- fn_call
  
  object$glm_type <- glm_type
  
  object$n_lambda_sparsity <- n_lambda_sparsity

  object$coef <- c(object$Intercept[object$Optimal_Index], object$Betas[,object$Optimal_Index])
  
  return(object)
}