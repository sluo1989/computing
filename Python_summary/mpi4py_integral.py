from mpi4py import MPI
from scipy.integrate import quad
from scipy.special import jv

def local_integrate(A, B, f):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nr_processors = comm.Get_size()

    h = float(B - A) / nr_processors
    local_A = A + rank * h
    local_B = local_A + h

    local_result, local_error = quad(f, local_A, local_B)

    print 'Process', rank
    print 'A =', local_A
    print 'B =', local_B
    print 'Integral =', local_result

    if rank == 0:
        results = []
        errors = []

        results.append(local_result)
        errors.append(local_error)

        for i in range(1, nr_processors):
            data = comm.recv(source=i)
            remote_result, remote_error = data
            results.append(remote_result)
            errors.append(remote_error)

        integral = sum(results)
        print integral

    else:
        data = (local_result, local_error)
        comm.send(data, dest=0)


def main():
    A, B = 0, 1000

    def f(x):
        '''Bessel function of order 2.5'''
        return jv(2.5, x)

    local_integrate(A, B, f)

if __name__ == '__main__':
    main()
