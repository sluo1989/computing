inc <- '
using namespace Rcpp;

double norm ( double x, double y ) {
    return sqrt ( x*x + y*y );
}

RCPP_MODULE(mod) {
    function ("norm", &norm);
}
'

fx <- cxxfunction(sig=c(), plugin="Rcpp", include=inc)
mod <- Module("mod", getDynLib(fx))

mod$norm(3,4)

