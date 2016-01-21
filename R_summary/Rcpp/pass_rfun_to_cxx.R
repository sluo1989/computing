src <- '
Rcpp::Function sort(rfun);
return sort(y, Named("decreasing", true));
'

fun <- cxxfunction(sig=c(rfun="function", y="ANY"), src, plugin="Rcpp")

fun(sort, sample(1:5, 10, TRUE))