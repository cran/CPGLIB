% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ProxGrad_Prediction_Functions.R
\name{predict.ProxGrad}
\alias{predict.ProxGrad}
\title{Predictions for ProxGrad Object}
\usage{
\method{predict}{ProxGrad}(object, newx, type = c("prob", "class")[1], ...)
}
\arguments{
\item{object}{An object of class ProxGrad}

\item{newx}{New data for predictions.}

\item{type}{The type of predictions for binary response. Options are "prob" (default) and "class".}

\item{...}{Additional arguments for compatibility.}
}
\value{
The predictions for the ProxGrad object.
}
\description{
\code{predict.ProxGrad} returns the predictions for a ProxGrad object.
}
\examples{
\donttest{
# Data simulation
set.seed(1)
n <- 50
N <- 2000
p <- 1000
beta.active <- c(abs(runif(p, 0, 1/2))*(-1)^rbinom(p, 1, 0.3))
# Parameters
p.active <- 100
beta <- c(beta.active[1:p.active], rep(0, p-p.active))
Sigma <- matrix(0, p, p)
Sigma[1:p.active, 1:p.active] <- 0.5
diag(Sigma) <- 1

# Train data
x.train <- mvnfast::rmvn(n, mu = rep(0, p), sigma = Sigma) 
prob.train <- exp(x.train \%*\% beta)/
              (1+exp(x.train \%*\% beta))
y.train <- rbinom(n, 1, prob.train)
# Test data
x.test <- mvnfast::rmvn(N, mu = rep(0, p), sigma = Sigma)
prob.test <- exp(x.test \%*\% beta)/
             (1+exp(x.test \%*\% beta))
y.test <- rbinom(N, 1, prob.test)

# ProxGrad - Single Group
proxgrad.out <- ProxGrad(x.train, y.train,
                         glm_type = "Logistic",
                         include_intercept = TRUE,
                         alpha_s = 3/4,
                         lambda_sparsity = 0.01, 
                         tolerance = 1e-5, max_iter = 1e5)

# Predictions
proxgrad.prob <- predict(proxgrad.out, newx = x.test, type = "prob")
proxgrad.class <- predict(proxgrad.out, newx = x.test, type = "class")
plot(prob.test, proxgrad.prob, pch = 20)
abline(h = 0.5,v = 0.5)
mean((prob.test-proxgrad.prob)^2)
mean(abs(y.test-proxgrad.class))

}

}
\seealso{
\code{\link{ProxGrad}}
}
\author{
Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
}
