library(parallel)
library(Rmpi)

# serial calculation
worker_func <- function(n) { return(sum(rnorm(n))) }
values <- 1:100

# parLapply (MPI)
cl <- makeCluster(4, type="MPI")
clusterSetRNGStream(cl, 1989)
res4 <- parLapply(cl, values, worker_func)
stopCluster(cl)
mpi.exit()
print(unlist(res4))
