'''
launching different tasks in parallel
launching tasks with more than one argument 
better control of task distribution
'''
from multiprocessing import Pool
import numpy

def sqrt(x):
    return numpy.sqrt(x)

if __name__ == '__main__':
    pool = Pool()
    results = [pool.apply_async(sqrt, (x,))
               for x in range(100)]
    roots = [r.get() for r in results]
    print roots