library(Rcpp)
library(RcppEigen)


cpptxt <- '
Eigen::VectorXd lmsol(const Eigen::MatrixXd & X, const Eigen::VectorXd & y) {
    return X.colPivHouseholderQr().solve(y);
}
'

fibR <- cppFunction(cpptxt, depends="RcppEigen")