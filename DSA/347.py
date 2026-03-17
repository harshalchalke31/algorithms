# 347. Top K Frequent Elements
from typing import List, DefaultDict
from collections import defaultdict
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        hashmap=defaultdict(int)
        for num in nums:
            hashmap[num]+=1
        result = sorted(hashmap,
                        key=hashmap.get, reverse=True)
        return result[:k]

if __name__=='__main__':
    sol = Solution()

    inp1 = [[1,1,1,2,2,3],2]
    inp2 = [[1],1]
    inp3 = [[1,2,1,3,3,3,2,1,2,3,1,3,2],2]

    inputs = [inp1, inp2, inp3]
    
    for nums, k in inputs:
        ans = sol.topKFrequent(nums=nums, k=k)
        print(ans)

