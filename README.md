
[![Build Status](https://app.travis-ci.com/AnthonyChristidis/CPGLIB.svg?branch=master)](https://app.travis-ci.com/AnthonyChristidis/CPGLIB)  [![CRAN\_Status\_Badge](http://www.r-pkg.org/badges/version/CPGLIB)](https://cran.r-project.org/package=CPGLIB) [![Downloads](http://cranlogs.r-pkg.org/badges/CPGLIB)](https://cran.r-project.org/package=CPGLIB)

CPGLIB
======

This package provides functions to generate ensembles of generalized linear models using competing proximal gradients.

------------------------------------------------------------------------

### Installation

You can install the **stable** version on [R CRAN](https://cran.r-project.org/package=CPGLIB).

``` r
install.packages("CPGLIB", dependencies = TRUE)
```

You can install the **development** version from [GitHub](https://github.com/AnthonyChristidis/CPGLIB)

``` r
library(devtools)
devtools::install_github("AnthonyChristidis/CPGLIB")
```

### Usage

``` r
# Required Libraries
library(mvnfast)

# Sigmoid function
sigmoid <- function(t){
  return(exp(t)/(1+exp(t)))
}

# Data simulation
set.seed(1)
n <- 50
N <- 2000
p <- 300
beta.active <- c(abs(runif(p, 0, 1/2))*(-1)^rbinom(p, 1, 0.3))
# Parameters
p.active <- 150
beta <- c(beta.active[1:p.active], rep(0, p-p.active))
Sigma <- matrix(0, p, p)
Sigma[1:p.active, 1:p.active] <- 0.5
diag(Sigma) <- 1

# Train data
x.train <- rmvn(n, mu = rep(0, p), sigma = Sigma) 
prob.train <- sigmoid(x.train %*% beta)
y.train <- rbinom(n, 1, prob.train)

# Test data
x.test <- rmvn(N, mu = rep(0, p), sigma = Sigma)
prob.test <- sigmoid(x.test %*% beta + offset)
y.test <- rbinom(N, 1, prob.test)
mean(y.test)
sp.sen.par <- y.test==0

# CPGLIB - CV (Multiple Groups)
cpg.out <- cv.cpg(x.train, y.train,
                  type="Logistic",
                  G=5, include_intercept=TRUE,
                  alpha_s=3/4, alpha_d=4/4,
                  n_lambda_sparsity=100, n_lambda_diversity=100,
                  tolerance=1e-3, max_iter=1e3,
                  n_folds=5,
                  n_threads=1)

# Coefficients
cpg.coef <- coef(cpg.out, ensemble_average=TRUE)

# Plot of predicted probabilities
cpg.prob <- predict(cpg.out, x.test,  groups=1:cpg.out$G, class_type="prob", ensemble_type="Model-Avg")
plot(prob.test, cpg.prob, pch=20)
abline(h=0.5,v=0.5)

# Misclassification rate
cpg.class <- predict(cpg.out, x.test, groups=1:10, class_type="class", ensemble_type="Model-Avg")
mean(abs(y.test-cpg.class))
```

### License

This package is free and open source software, licensed under GPL (&gt;= 2).
