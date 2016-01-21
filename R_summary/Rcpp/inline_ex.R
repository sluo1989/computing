library(Rcpp)
library(RcppArmadillo)
library(inline)
library(rbenchmark)

src <- '
    Rcpp::NumericMatrix Xr (Xs);
    Rcpp::NumericVector yr (ys);
    int n = Xr.nrow(), p = Xr.ncol();
    arma::mat X(Xr.begin(), n, p, false);
    arma::colvec y(yr.begin(), yr.size(), false);
    int df = n - p; 
    
    arma::colvec coef = arma::solve(X, y);
    arma::colvec res = y - X*coef; 
    
    double s2 = std::inner_product(res.begin(), res.end(),
                                   res.begin(), 0.0)/df;
    
    arma::colvec sderr = arma::sqrt(s2*
      arma::diagvec(arma::pinv(arma::trans(X)*X)));

    return Rcpp::List::create(Rcpp::Named("coefficients") = coef,
                              Rcpp::Named("stderr")       = sderr,
                              Rcpp::Named("df")           = df);
'

flm <- cxxfunction(sig = c(Xs="numeric", ys="numeric"), src, plugin="RcppArmadillo")


## benchmark lm and flm

n <- 20; p <- 5
beta <- c(1,2,-2,5,3)

X <- matrix(rnorm(100), n, p)
y <- X%*%beta + rnorm(n)*0.1

benchmark( lm(y~X), 
           flm(X,y), 
           columns=c("test", "replications", "elapsed", "relative"),
           order="relative", replications=100)

