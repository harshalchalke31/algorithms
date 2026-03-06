# 49. Group Anagrams

# Inputs
from typing import List, DefaultDict


inp1 = ["eat","tea","tan","ate","nat","bat"]
inp2 = [""]
inp3 = ["a"]
inp4 = []

inputs = [inp1, inp2, inp3,inp4]

## Solution
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        result = DefaultDict(list)
        for word in strs:
            count = [0]*26
            for character in word:
                count[ord(character)-ord('a')]+=1
            result[tuple(count)].append(word)
        return list(result.values())


if __name__=='__main__':
    sol = Solution()
    for i in inputs:
        ans = sol.groupAnagrams(i)
        print(ans)