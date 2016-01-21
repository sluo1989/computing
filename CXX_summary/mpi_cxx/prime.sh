mpic++ prime_mpi.cpp
#
if [ $? -ne 0 ]; then
  echo "Errors compiling prime_mpi.cpp"
  exit
fi
#
#  Rename the executable.
#
mv a.out prime
#
#  Ask MPI to use 8 processes to run your program.
#
mpirun -np 8 ./prime > prime_output.txt
#
#  Clean up.
#
rm prime
#
echo "Program output written to prime_fsu_output.txt"
