import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# create numpy array
array1 = np.arange(5)
array2 = np.array([[0,1,2],[3,4,5]], dtype=np.int8)
array3 = np.zeros((2,3)) # np.ones((2,3))
array4 = np.zeros_like(array2) # np.ones_like(array2)
array5 = np.eye(3) # np.eye(3,4), np.identity(3)
array6 = np.random.randn(3,4) # similar to Matlab
array7 = np.arange(16).reshape((2,2,4)) 

# change dtype
array2.astype(np.float64) # always create a new array

# array indexing and slicing (view on array, no copies)
array2_view = array2[:2, 1:] 
array2_view[0,0] = 5
array2_copy = array2[:2, 1:].copy() # create a copy

# boolean indexing
array2[array2>3] = 4

# fancy indexing - always copies data into a new array 
array2[1] = 0 # set second row to zeros
array2[[1,0]] # reverse two rows
array6[[0,1,2],[0,2,3]] # take out array6[0,0], array6[1,2], array6[2,3]
array6[[2,0]][:,[0,3]] # select sub-array corresponding to rows [2,0] and columns [0,3]
array6[np.ix_([2,0],[0,3])] # same as above

# take a transpose of an array
array6.T
array7.transpose((1,0,2)) # check the document for details
array7.swapaxes(1,2)

# fast element-wise array functions
# unary functions
# sin, cos, square, abs, log, floor, ceil, isnan
np.exp(array7) 
np.sqrt(array7) 

# binary functions
# add, subtract, multiply, divide, power, maximum, mimumum, mod, 
# greater, greater_equal, less, less_equal, equal, not_equal, 
# logical_and, logical_or, logical_xor
x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x, y)

# expressing conditional logic as array operations
array8 = np.random.randn(3,5)
np.where(array8 > 2, 2, array8) # set values greater than 2 to be 2

# mean, std, var, sum, max, min, argmax, argmin, cumsum, cumprod
array8.mean(axis=1) # (3,)
array8.mean(axis=0) # (5,)

# boolean array
array9 = array8 > 0
array9.sum() # number of positive items
array9.any() # any positive item ?
array9.all() # all positive item ?

# sort 
array8.sort(axis=1) # sort along dimension 1
values = np.array([5,0,1,3,2])
indexer = values.argsort()
arr = np.random.randn(3,5)
arr[0] = values
arr[:, arr[0].argsort()] # sort rows in concordance to order of first row


# set logic: unique, in1d, interset1d, union1d, setdiff1d, setxor1d
array10 = np.array([3,3,3,2,2,1,1,4])
np.unique(array10) # get unique items - sorted(set(array10))
array11 = np.in1d(array10, [2,3,6]) # test membership of values in array10 in [2,3,6]

# save and load array
np.save('my_array.npy', array10) 
np.load('my_array.npy')

np.savez('my_arrays.npz', array9, array10)
arch = np.load('my_arrays.npz')

# save and load txt 
# np.loadtxt('my_txt.txt', delimiter=','), np.savetxt,

# linear algebra
# np.diag, dot, trace
# np.linalg.det, eig, svd, qr, inv, pinv, solve, lstsq
from numpy.linalg import qr, svd
mat = np.random.randn(3,4)
q, r = qr(mat)
s, v, d = svd(mat)



# numpy advanced topics
# 1) pointer to data
# 2) data type, dtype
# 3) shape
# 4) strides
# created in row major (C) order rather than (F) order

arr = np.ones((3,4,5), dtype=np.float64)
arr.dtype   # float64
arr.ndim    # 3  
arr.shape   # (3,4,5)
arr.strides # (160, 40, 8) 

# reshape arrays
arr = np.arange(8)
arr.reshape((4,2)) # arr.reshape((4,-1))
arr.ravel() # not necessary a copy, high dimension to one dimension
arr.flatten() # copy, high dimension to one dimension

# concatenating and splitting arrays
arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[7,8,9],[10,11,12]])

arr3 = np.concatenate([arr1, arr2], axis=0) 
# np.vstack((arr1, arr2)) or np.vstack([arr1, arr2]) or np.r_(arr1, arr2)
arr4 = np.concatenate([arr1, arr2], axis=1) 
# np.hstack((arr1, arr2)) or np.hstack([arr1, arr2]) or np.c_(arr1, arr2)

arr1, arr2 = np.split(arr3, [2], axis=0) 

# repeat elements, tile and repeat
arr = np.random.randn(2,3)
arr.repeat([2,4], axis=0)
arr.repeat([2,3,4], axis=1)

np.tile(arr, (2,4))

# broadcasting rule
# Two arrays are compatible for broadcasting if for each 'trailing dimension', the axis lengths
# match or if either of length 1.

arr = np.random.randn(4,3)
demeaned = arr - arr.mean(0) # (4,3) - (3,) 

arr = np.random.randn(3,4,5)
depth_means = arr.mean(2)
demeaned = arr - depth_means[:,:,np.newaxis] # add newaxis to make broadcasting work

arr = np.zeros((4,3))
col = np.random.randn(4) # (4,)
arr[:] = col[:, np.newaxis] # arr[:,:2] = col[:, np.newaxis]

# ufunc instance methods
arr = np.random.randn(5,6)
arr[::2].sort(1) # sort a few rows
np.logical_and.reduce(arr[:,:-1] < arr[:,1:], axis=1)
ret = np.subtract.outer(np.random.randn(3,4), np.random.randn(5)) # (3,4,5)

arr = np.multiply.outer(np.arange(4), np.arange(5)) 
np.add.reduceat(arr, [0,2,4], axis=1)


# create Series
obj = Series([4,7,-5,3], index=['d','b','a','c']) # sequence-like 
dict_data = {'Ohio':35000, 'Texas':71000, 'Oregon':16000, 'Utah':5000} # dictionary
obj2 = Series(dict_data)
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj3 = Series(dict_data, index=states)

obj.values # return underlying numpy array 
obj.index # return the index 
obj3.isnull() # pd.isnull(obj3), obj3.notnull(), pd.notnull(obj3)

# create DataFrame
data = {'state' : ['Ohio','Ohio','Ohio','Nevada','Nevada'],
        'year' : [2000, 2001, 2002, 2001, 2002],
        'pop' : [1.5, 2.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame2 = DataFrame(data, columns=['year','state','pop','debt'],
                   index=['one','two','three','four','five'])
pop = {'Nevada' : {2001: 2.4, 2002: 2.9},
       'Ohio' : {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3.T
frame3.index.name = 'year'
frame3.columns.name = 'state'


# essential functionality
# reindexing
obj = Series(np.random.randn(4), index=['d','b','a','c'])
obj2 = obj.reindex(list('abcde'))
obj2 = obj.reindex(list('abcde'), fill_value=0)

obj3 = Series(['blue','purple','yellow'], index=[0,2,4]) # ordered data 
obj3.reindex(range(6), method='ffill')

frame = DataFrame(np.arange(9).reshape((3,3)), index=['a','c','d'],
                  columns=['Ohio','Texas','California'])
frame2 = frame.reindex(list('abcd')) 
states = ['Texas','Utah','California']
frame3 = frame.reindex(columns=states)
frame4 = frame.reindex(index=list('abcde'), method='ffill', 
                       columns=states)
frame.ix[['a','b','c','d'], states]

# drop entry from an axis - return a new copy if inplace=False
frame.drop(['a','c']) 
frame.drop('Ohio', axis=1)
frame.drop(['Ohio', 'California'], axis=1, inplace=False)


# indexing, selection, filtering
frame = DataFrame(np.arange(9).reshape((3,3)), index=['a','c','d'],
                  columns=['Ohio','Texas','California'])
frame['Ohio']
frame[['Ohio','California']]
frame[:1] # works only on range indexing 

frame.ix['a',['Ohio', 'Texas']] # frame.ix['a',[0,1]]
frame.ix['a'] # frame.ix[0]

# frame[frame['Ohio'] > 2]
# frame[frame < 5]
# frame.ix[frame.Ohio > 2, :2]

# broadcasting in pandas
series = frame.ix[0]
frame - series
series = frame['Ohio']
frame.sub(series, axis=0)

# function mapping
f = lambda x : x.max() - x.min()
frame.apply(f) # column-wise operation
frame.apply(f, axis=1) # row-wise operation

def f(x):
    return Series([x.min(), x.max()], index=['min','max'])

frame.apply(f)

f = lambda x : '%.2f' % x
frame.applymap(f) # element-wise operation


# sort index and values
obj = Series([4,7,-3,2])
obj.order()
frame = DataFrame(np.random.randn(5,4), index=['three','one','two','five','four'],
                  columns=['d','a','b','c'])
frame.sort_index() 
frame.sort_index(axis=1)
frame.sort_values(by='a') # frame.sort_values(by=['a','b'])

obj.rank()
obj.rank(method='first') # 'average', 'min', 'max', 'first'

frame.rank(axis=1)

# descriptive statistics
df = DataFrame([[1.4,np.nan],[7.1,-4.5],
               [np.nan,np.nan],[0.75,-1.3]],
               index=list('abcd'),
               columns=['one','two'])
df.describe() 
# skipna=True, mean, std, var, sum, 
# max, min, argmax, argmin, idxmax, idxmin,
# cumsum, cumprod, diff, pct_change

# Correlation and Covariance
df = DataFrame(np.random.randn(100,3), columns=list('abc'))
df.corr() 
df.cov()
df.corrwith(df['a'])

# unique values, value counts, membership
obj = Series(['c','a','d','a','a','b','b','c','c'])
uniques = obj.unique()
obj.value_counts()
mask = obj.isin(['b','c'])
obj[mask]

# deal with missing data
df = DataFrame(np.random.randn(7,3))
df.ix[:4,1] = np.nan; df.ix[:2,2] = np.nan

df.dropna(thresh=3)
df.fillna(0)
df.fillna({1:0.5,3:-1})
df.fillna(method='bfill')
df.fillna(method='bfill', limit=2)


# read and write data 
'''
pd.read_csv(file_path, sep=',', header=None, index_col, names, skiprows, na_value, nrows, chunksize)
pd.to_csv(sys.stdout, na_rep='NULL', index=False, header=False)
'''

# manually working with delimited formats
'''
!cat ex.csv
"a", "b", "c"
"1", "2", "3"
"4", "5", "6", "7"
'''

import csv

with open('ex.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(('one','two','three'))
    writer.writerow(('1','2','3'))
    writer.writerow(('4','5','6','7'))
    
lines = list(csv.reader(open('ex.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}

'''
# XML and HTML: Web Scrapping

from lxml.html import parse
from urllib2 import urlopen
'''

# merge
'''
pd.merge(df1, df2, 
         on, # merge on 
         how, # left, right, inner, outer
         left_on, # deal with different names
         right_on, # on df1 and df2 on the merge columns
         suffixes=('_left', '_right'), # deal with overlapping column names 
         left_index, # True or False, merge items are on index ?
         right_index) # True or False, merge items are on index ?
'''

'''
pd.concat([df1, df2, df3],
          axis, # the axis to concatenate along
          join, # 'inner', 'outer', default 'outer'
          join_axes, # specific indexes to use instead of inner/outer set logic
          keys, # construct hierarchical index using the keys as the outermost level
          levels, # specific levels to use for constructing a multiindex
          ignore_index)          
'''
s1 = Series([0,1], index=['a','b'])
s2 = Series([2,3,4], index=['c','d','e'])
s3 = Series([5,6], index=['f','g'])
pd.concat([s1,s2,s3])
pd.concat([s1,s2,s3], axis=1)
pd.concat([s1,s2,s3], axis=1, keys=['one','two','three'])
s4 = pd.concat([s1*5,s3])
pd.concat([s1,s4], axis=1)
pd.concat([s1,s4], axis=1, join='inner')
pd.concat([s1,s4], axis=1, join_axes=[['a','c','b','e']])
pd.concat([s1,s1,s3], keys=['one','two','three'])
pd.concat([s1,s1,s3], axis=1, keys=['one','two','three'])

df1 = DataFrame(np.arange(6).reshape(3,2), index=['a','b','c'],
                columns=['one','two'])
df2 = DataFrame(5+np.arange(4).reshape(2,2), index=['a','c'], columns=['three','four'])

pd.concat([df1,df2], axis=1, keys=['level1', 'level2'])
df3 = pd.concat([df1,df2], axis=1, keys=['level1', 'level2'],
          names=['upper', 'lower'])

# reshape
# stack and unstack
data = DataFrame(np.arange(6).reshape(2,3),
                 index=pd.Index(['Ohio','Colorado'], name='state'),
                 columns=pd.Index(['one','two','three'], name='number'))
result = data.stack()
result.unstack()
result.unstack('state') 

df = DataFrame({'left':result, 'right':result+5},
               columns=pd.Index(['left','right'], name='side'))
df.unstack('state')

# pivot long to wide format
# check help(pd.DataFrame.pivot) and help(pd.DataFrame.pivot_table)

# transform 
# help(pd.Series.map) 

# replace values
data = Series([1.,-999.,2.,-999.,-1000.,3.])
data.replace(-999, np.nan)
data.replace([-999,-1000], np.nan)
data.replace([-999,-1000], [np.nan,0]) # data.replace({-999:np.nan,-1000:0})

# permutation and random sampling
df = DataFrame(np.arange(20).reshape(5,4))
sampler = np.random.permutation(5)
df.take(sampler)
df.take(sampler[:3])
df.take(np.random.permutation(len(df))[:3]) # sample without replacement (slow but works)
sampler = np.random.randint(0, len(df), size=10)
df.take(sampler) # sample with replacement

# get dummy variables for categorical variable
df = DataFrame({'key':['b','b','a','c','a','b'],
                'value':range(6)})
dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['value']].join(dummies) # type(df['value']) and type(df[['value']])

# string manipulation
'''
startwith, endwith, split, strip, count
index, find, rfind, replace, join,  
upper, lower, ljust, rjust, lstrip, rstrip
'''

val = 'a,b,  guido'
val.split(',')
pieces = [x.strip() for x in val.split(',')]
','.join(pieces)
# 'guido' in val # check substring 

# regular expression
'''
import re
re.match, # pattern must occur at start of target
re.search, # pattern can occur anywhere
re.findall, # finds all instances of the pattern
re.split, # break string into pieces at each occurrence of pattern

# meta-characters
'.' matches any character but newline
'*' matches zero or more instances of preceding pattern
'+' matches one or more instances of preceding pattern
'?' matches zero or one instance of preceding pattern
'^' matches the start of the target
'$' matches the end of the target
() to enclose pattern, # (HW)+ means 'HW', 'HWHW', ...
[] to denote unordered set, # [HW] match 'HW', 'WH' 
{m} to match m copies of preceding pattern
{m,n} to match m to n copies of preceding pattern 
| stands for 'or', (HW)|(hw) will match 'HW' or 'hw'
- denotes range, [A-Z] match all capital letters
'\w' matches all alphanumeric character, same as [A-Za-z0-9]
'\s' matches any whitespace character, same as [\t\n\r\f\v]
'''

import re
text = 'foo  bar\t baz   \tqux'
re.split('\s+', text)


# plot, check matplotlib.pyplot
df = DataFrame({'data1':np.random.randn(20),'data2':np.random.randn(20),
                'key1':['a']*10+['b']*20+['a']*10,
                'key2':['one','two','one','two','one']*4})
grouped = df.groupby('key1')
means = grouped.mean()
def sample_without_replacement(df):
    ll = len(df)
    return df.take(np.random.permutation(ll)[:int(0.5*ll)])
result = []
for _ , group in grouped: # _ to denote not_to_use variable
    res = sample_without_replacement(group) # sample in each group
    result.append(res) 
result = pd.concat(result, ignore_index=True)

# sample without replacement in each group and combine together in one line
result2 = grouped.apply(sample_without_replacement).reset_index(drop=True, inplace=True)

# fill na values using group means 
states = ['Ohio','New York', 'Vermont', 'Florida',
          'Oregon','Nevada','California','Idaho']
group_key = ['East']*4 + ['West']*4
data = Series(np.random.randn(8), index=states)
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
fill_mean = lambda g : g.fillna(g.mean())
data_new = data.groupby(group_key).apply(fill_mean)


### generator
def make_change(amount, coins=[1,5,10,25], hand=None):
    hand = [] if hand is None else hand
    if amount == 0:
        yield hand
    for coin in coins:
        if coin > amount or (len(hand) > 0 and hand[-1] < coin):
            continue
        
        for result in make_change(amount-coin, coins=coins, 
                                  hand=hand+[coin]):
            yield result

for way in make_change(100, coins=[10,25,50]):
    print way

           
### time series data
# data and time
from datetime import datetime, timedelta
now = datetime.now() # check now.date(), now.time(), now.year, now.hour
delta = datetime(2011,1,7) - datetime(2008,6,24,8,15)
start = datetime(2011,1,7)
end = start + timedelta(12)

# convert btw string and datetime
stamp = datetime(2011,1,7)
str(stamp) # stamp.strftime('%Y-%m-%d')
value = '2011-01-03'
datetime.strptime(value,'%Y-%m-%d')
datestrs = ['7/6/2011','8/6/2011']
[datetime.strptime(x) for x in datestrs] # or 
pd.to_datetime(datestrs)

from dateutil.parser import parse
parse('2011-01-03')
parse('Jan 31, 1997 10:45 PM')

ts = Series(np.random.randn(1000),
            index=pd.date_range('1/1/2000', periods=1000))
ts['2001']
ts['1/1/2000':'1/31/2000']
ts.truncate(after='12/31/2001')
# check ts.resample('M')

# shifting data
ts = Series(np.random.randn(4),
            index=pd.date_range('1/1/2000', periods=4, freq='M'))
ts/ts.shift(1) - 1 # pct_change