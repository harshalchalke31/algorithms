# Find Minimum in Rotated Sorted Array
from typing import List

class Solution:
    def findMin(self, nums: List[int]) -> int:
        left,right = 0, len(nums)-1
        ans = 5001

        while left<=right:
            mid = (left+right)//2
            if nums[mid]<=ans: ans=nums[mid]
            print(f'Array:{nums} mid:{mid}')

            # left sorted
            if nums[mid]>=nums[left]:
                # eliminated sorted half
                if ans>nums[left]: ans = nums[left]
                left = mid+1 

            # right sorted
            else:
                # check if sorted half can be eliminated
                right = mid-1
        return ans
    

if __name__=='__main__':
    sol = Solution()
    inp1 = [3,4,5,1,2]
    inp2 = [4,5,6,7,0,1,2]
    inp3 = [11,13,15,17]

    inputs = [inp1, inp2, inp3]
    
    for nums in inputs:
        ans = sol.findMin(nums=nums)
        print(ans)