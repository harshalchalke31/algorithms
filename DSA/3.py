# Longest Substring Without Repeating Characters
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        hmap = dict()
        left, right, ans = 0,0,0
        # Initailize hashmap
        for i in range(26):
            hmap[i] = -1

        while right<len(s):
            if hmap[ord(s[right])-ord('a')]==-1:
               ans = max(ans, right-left+1)
               hmap[ord(s[right])-ord('a')] = right
               right+=1
            else:
               left = hmap[ord(s[right])-ord('a')]+1
               hmap[ord(s[right])-ord('a')] = right
               right+=1
               
        return ans


if __name__=='__main__':
    sol = Solution()
    inp1 = "abcabcbb"
    inp2 = "bbbbb"
    inp3 = "pwwkew"
    inp4 = "abcdefghii"

    inputs = [inp1, inp2, inp3,inp4]
    
    for s in inputs:
        ans = sol.lengthOfLongestSubstring(s)
        print(ans)
