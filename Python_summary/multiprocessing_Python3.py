from multiprocessing import Pool
from parutils import distribute
import numpy
import sharedmem


def apply_sqrt(a, imin, imax):
    return numpy.sqrt(a[imin:imax])


if __name__ == '__main__':
    pool = Pool()
    data = sharedmem.empty((100,), numpy.float)
    data[:] = numpy.arange(len(data))
    slices = distribute(len(data))
    results = [pool.apply_async(apply_sqrt, (data, imin, imax))
               for (imin, imax) in slices]
    for r, (imin, imax) in zip(results, slices):
        data[imin:imax] = r.get()
    print data