# -----------------------------------------------------------------------
# Object Construction for ProxGrad object
#
# object: the ProxGrad object
# fn_call: the function call
# -----------------------------------------------------------------------
construct.ProxGrad <- function(object, fn_call, glm_type, lambda_sparsity){
  
  class(object) <- append("ProxGrad", class(object))
  object$call <- fn_call
  
  object$coef <- c(object$Intercept, object$Betas)
  
  object$lambda_sparsity <- lambda_sparsity
  
  return(object)
}

# -----------------------------------------------------------------------
# Object Construction for cv.ProxGrad object
#
# object: the cv.ProxGrad object
# fn_call: the function call
# -----------------------------------------------------------------------
construct.cv.ProxGrad <- function(object, fn_call, glm_type){
  
  class(object) <- append("cv.ProxGrad", class(object))
  object$call <- fn_call
  
  object$glm_type <- glm_type

  object$coef <- c(object$Intercept[object$Optimal_Index], object$Betas[,object$Optimal_Index])
  
  return(object)
}