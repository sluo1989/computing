#!/bin/bash
#
mpic++ -c quadrature_mpi.cpp
if [ $? -ne 0 ]; then
  echo "Errors compiling quadrature_mpi.cpp"
  exit
fi
#
mpic++ quadrature_mpi.o -lmpi
if [ $? -ne 0 ]; then
  echo "Errors loading quadrature_mpi.o"
  exit
fi
rm quadrature_mpi.o
#
mv a.out quadrature
mpirun -v -np 4 ./quadrature > quadrature_output.txt
if [ $? -ne 0 ]; then
  echo "Errors running quadrature"
  exit
fi
rm quadrature
#
echo "The quadrature test problem has been executed."
