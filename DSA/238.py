# 238. Product of Array Except Self
from typing import List
class Solution:
    #Brute force
    # def productExceptSelf(self,nums:List[int])->List[int]:
    #     prod, ans = 1,[]
    #     for i in range(len(nums)):
    #         prod=1
    #         for j in range(len(nums)):
    #             if i!=j:
    #                 prod*=nums[j]
    #         ans.append(prod)
    #     return ans
    
    #Better Soln
    # def productExceptSelf(self, nums: List[int]) -> List[int]:
    #     prefix,postfix = [],[]
    #     ans,mul,mul2 = [],1,1
    #     for idx in range(len(nums)):
    #         mul*=nums[idx] 
    #         mul2*=nums[len(nums)-idx-1]
    #         prefix.append(mul)
    #         postfix.append(mul2)
    #     postfix.reverse()

    #     for idx in range(len(nums)):
    #         if idx==0:
    #             ans.append(postfix[idx+1])
    #         elif idx==len(nums)-1:
    #             ans.append(prefix[idx-1])
    #         else:
    #             ans.append(prefix[idx-1]*postfix[idx+1])
                
    #     return ans

    #Optimal Soln
    def productExceptSelf(self,nums:List[int])->List[int]:
        ans = [1]*len(nums)
        prefix,postfix=1,1
        for idx in range(len(nums)):
            print(f'Prefix before:[{idx}]={prefix}')
            ans[idx]=prefix
            prefix*=nums[idx]
            print(f'Prefix after:[{idx}]={prefix}')
        print(f'Ans after prefix: {ans}')
        for idx in range(len(nums)-1,-1,-1):
            print(f'Postfix before:[{idx}]={postfix}')
            ans[idx]*=postfix
            postfix*=nums[idx]
            print(f'Postfix after:[{idx}]={postfix}')
        return ans


if __name__=='__main__':
    sol = Solution()
    inp1 = [1,2,3,4]
    inp2 = [-1,1,0,-3,3]

    inputs = [inp1, inp2]
    
    for nums in inputs:
        ans = sol.productExceptSelf(nums)
        print(ans)
