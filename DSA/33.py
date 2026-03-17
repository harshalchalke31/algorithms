# Search in rotated sorted array

from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while left<=right:
            mid = (left+right)//2
            if nums[mid]==target: return mid
            print(f'Array:{nums} mid:{mid}')

            # left sorted
            if nums[left]<=nums[mid]:
                # check if sorted part can be eliminated
                if nums[mid]<target or nums[left]>target:
                    left = mid+1
                else:
                    right = mid-1
            # right sorted
            else:
                # check if sorted part can be eliminated
                if nums[mid]>target or nums[right]<target:
                    right = mid-1
                else:
                    left=mid+1
        return -1
    
if __name__=='__main__':
    sol = Solution()
    inp1 = [[4,5,6,7,0,1,2],0]
    inp2 = [[4,5,6,7,0,1,2],3]
    inp3 = [[3,1],1]

    inputs = [inp1, inp2, inp3]
    
    for nums, target in inputs:
        ans = sol.search(nums=nums, target=target)
        print(ans)