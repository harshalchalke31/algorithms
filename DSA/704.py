# binary search

from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while left <= right:
            mid = (right+left)//2
            print(f'Array:{nums} mid:{mid}')
            if nums[mid]==target: return mid

            #target lies in left half
            if nums[mid]>target:
                right = mid-1
            #target lies in right half
            else:
                left = mid+1
        return -1
    
if __name__=='__main__':
    sol = Solution()
    inp1 = [[1,0,3,5,9,12],9]
    inp2 = [[1,0,3,5,9,12],2]
    inp3 = [[5],5]

    inputs = [inp1, inp2, inp3]
    
    for nums, target in inputs:
        ans = sol.search(nums=nums, target=target)
        print(ans)