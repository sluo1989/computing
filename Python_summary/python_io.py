'''
Created on Dec 9, 2015

@author: Shikai
'''
#############################################################
'''
import sys

def test():
    if (len(sys.argv) == 1):
        print 'Not enough arguments'
    else:
        args = sys.argv[1:]
        print 'Hello ' + ' '.join(args)
                
if __name__ == '__main__':
    test()
'''

x = raw_input('what are the first 10 perfect squares? ')
# >>> map(lambda x : x*x, range(10))
# x = eval(x)
x = input('what are the first 10 perfect squares? ')
# >>> map(lambda x : x*x, range(10))
# input() will attempt to evaluate it as if it were a python program
# input() = raw_input() + eval()

print '{} and {}'.format('spam', 'eggs')
print '{0} and {1}'.format('spam', 'eggs')
print '{1} and {0}'.format('spam', 'eggs')
print 'This {food} is {adjective}.'.format(
        food='spam', adjective='absolutely horrible')

import math
# !s (str()), !r (repr()) 
print 'The value of PI is approximately {!r}.'.format(math.pi)
print 'The value of PI is approximately {0:.3f}.'.format(math.pi)

table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
for name, phone in table.items():
    print '{0:10} ==> {1:10d}'.format(name, phone)

#############################################################
import os
os.chdir('/Users/Shikai/Documents/workspace/python_io/src')
f = open('sample.txt','r')
for line in f:
    print line,
    # Note the trailing comma on previous line,
    # it is to avoid extra empty line
f.close()
lines = list(open('sample.txt'))

# better use 'with' to deal with file objects
with open('sample.txt', 'r') as f:
    lines = f.readlines()

'''
import fileinput
import glob # unix style pathname pattern expansion 
import string, sys

# fileinput.input(sys.argv[1:])
for line in fileinput.input(glob.glob("samples/*.txt")):
    if fileinput.isfirstline(): # first in a file?
        sys.stderr.write("-- reading %s --\n" % fileinput.filename())
    sys.stdout.write(str(fileinput.lineno()) + " " + string.upper(line))

https://docs.python.org/2/library/fileinput.html#module-fileinput
'''

''' 
numpy input and output

# numpy binary files
load(file, ...) # load arrays from .npy or .npz files
save(file, ...) # save one array to a binary file .npy format
savez(file, ...) # save several arrays into a single file in uncompressed .npz format
savez_compressed(file, ...) # in compressed .npz format

# text files
loadtxt(fname, ...)
savetxt(fname, ...)
genfromtxt(file_name, # local file or url or gz or bz2
         dtype, # 1) a single type, 
                # 2) a sequence of types corresponding to each column
                # 3) a sequence of tuples (name, type)
         delimiter, # 1) default: whitespaces
                    # 2) str, ',', ';'
                    # 3) int, width of each field
                    # 4) a sequence
         skip_header, # number of lines to skip at the beginning of the file
         skip_footer, # number of lines to skip at the end of the file
         missing_values, # 1) a string or a comma-separated string
                         # 2) a sequence of strings
                         # 3) a dictionary 
         filling_values, # 1) a single value for all columns
                         # 2) a sequence of values, for corresponding column
                         # 3) a dictionary
         comments, # str, '#'
         usecols, # sequence of ints or names
         autostrip, # strip white spaces from the variables
         max_rows) # maximum number of rows to read
         
Note: refer to help(numpy.genfromtxt) for more details
http://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html 
http://docs.scipy.org/doc/numpy-1.10.0/reference/routines.io.html                 
'''
'''
refer to pandas IO tools, fancy
http://pandas.pydata.org/pandas-docs/stable/io.html

### appending to a csv
with open(filename, 'a') as f:
    df.to_csv(f, header=False)
    # header=False so as not to append column names

### read in multiple files together
df = []
for file in files:
    df.append(process_your_file(file))
df = pd.concat(df)

### read in a large file chunk by chunk to avoid memory issue
df = pd.read_csv(filename, iterator=True, chunksize=1000)
df = pd.concat(df, ignore_index=True)

### inferring dtypes from a file
chunker = pd.read_csv(filename, chunksize=1000, ...)
dtypes = pd.DataFrame([chunk.dtypes for chunk in chunker])
types = dtypes.max().to_dic()
chunker = pd.read_csv(filename, chunksize=1000, dtype=types, ...)

### Creating a store chunk-by-chunk from a large csv file
store = pd.HDFStore('store.h5',mode='w')
for chunk in pd.read_csv('file.csv', chunksize=1000):
    store.append('df', chunk) 
store.close()

### 


'''