library(rbenchmark)
library(gputools)

n <- 10000; p <- 1000
beta <- rnorm(p)

X <- matrix(rnorm(n*p), n, p)
y <- X%*%beta + rnorm(n)*0.1

benchmark( lm(y~X), 
           gpuLm(y~X), 
           columns=c("test", "replications", "elapsed", "relative"),
           order="relative", replications=10)