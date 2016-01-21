'''
Created on Dec 11, 2015

@author: Shikai
'''

def dfs(nums, s, res):
    n = len(nums)
    if s >= n:
        res.append(nums[:])
    for i in xrange(s, n):
        nums[i], nums[s] = nums[s], nums[i]
        dfs(nums, s+1, res)
        nums[i], nums[s] = nums[s], nums[i]
    
nums = [1,2]
res = []
dfs(nums, 0, res)

        