# -----------------------------------------------------------------------
# Object Construction for CPGLIB object
#
# object: the CPGLIB object
# fn_call: the function call
# -----------------------------------------------------------------------
construct.CPGLIB <- function(object, fn_call, glm_type, G, lambda_sparsity, lambda_diversity){
  
  class(object) <- append("CPGLIB", class(object))
  object$call <- fn_call
  
  object$glm_type <- glm_type
  
  object$G <- G
  
  object$lambda_sparsity <- lambda_sparsity
  object$lambda_diversity <- lambda_diversity
  
  return(object)
}

# -----------------------------------------------------------------------
# Object Construction for cv.CPGLIB object
#
# object: the cv.CPGLIB object
# fn_call: the function call
# -----------------------------------------------------------------------
construct.cv.CPGLIB <- function(object, fn_call, glm_type, G, n_lambda_sparsity, n_lambda_diversity){
  
  class(object) <- append("cv.CPGLIB", class(object))
  object$call <- fn_call
  
  object$glm_type <- glm_type
  
  object$G <- G
  
  object$n_lambda_sparsity <- n_lambda_sparsity
  object$n_lambda_diversity <- n_lambda_diversity

  return(object)
}