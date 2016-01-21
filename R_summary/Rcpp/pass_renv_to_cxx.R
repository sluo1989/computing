src <- '
int nr = Rcpp::as<int>(x), nc = Rcpp::as<int>(y);
Rcpp::Environment stats("package:stats");
Rcpp::Function rnorm = stats["rnorm"];
Rcpp::NumericVector vec(rnorm(nr*nr, Rcpp::Named("sd",100.0)));
Rcpp::NumericMatrix mat(nr, nc, vec.begin());
return mat;
'

func <- cxxfunction(sig=c(x="integer", y="integer"), src, plugin="Rcpp")

func(3,4)

