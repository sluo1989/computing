#!/bin/bash
#
mpic++ -c matvec_mpi.cpp
if [ $? -ne 0 ]; then
  echo "Errors compiling matvec_mpi.cpp"
  exit
fi
#
mpic++ matvec_mpi.o -lmpi -lm
if [ $? -ne 0 ]; then
  echo "Errors loading matvec_mpi.o"
  exit
fi
rm matvec_mpi.o
#
mv a.out matvec
mpirun -v -np 4 ./matvec > matvec_output.txt
if [ $? -ne 0 ]; then
  echo "Errors running matvec"
  exit
fi
rm matvec
#
echo "The matvec test problem has been executed."
