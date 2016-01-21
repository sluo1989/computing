Rcpp::sourceCpp("Kalman.cpp")

pos <- as.matrix(read.table("pos.txt", header=FALSE,
                            col.names=c("x","y")))
res <- KalmanCpp(pos)