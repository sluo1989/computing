library(parallel)

# serial calculation
worker_func <- function(n) { return(sum(rnorm(n))) }
values <- 1:100
res1 <- lapply(values, worker_func)

# mclapply (check ### mcmapply ###)
RNGkind("L'Ecuyer-CMRG")
set.seed(1989)
res2 <- mclapply(values, worker_func, mc.cores=4)


# parLapply (PSOCK) (check ### clusterMap ###)
cl <- makeCluster(getOption ("cl.cores", 4), type="PSOCK")
clusterSetRNGStream(cl, 1989)
res3 <- parLapply(cl, values, worker_func)
stopCluster(cl)

# # parLapply (MPI)
# library(Rmpi)
# cl <- makeCluster(4, type="MPI")
# clusterSetRNGStream(cl, 1989)
# res4 <- parLapply(cl, values, worker_func)
# stopCluster(cl)
# mpi.exit()

# clusterExport, clusterEvalQ, clusterCall, clusterApply
cl <- makeCluster(getOption ("cl.cores", 4), type="PSOCK")
clusterSetRNGStream(cl, 1989)
a <- 3
b <- 1:2
clusterExport(cl, list("a", "b"))
clusterEvalQ(cl, a+5)
clusterCall(cl, sin, b)
clusterApply(cl, c(3,6,9,12), sqrt)
stopCluster(cl)