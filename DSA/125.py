# 125. Valid Palindrome

inp1 =  "A man, a plan, a canal: Panama"
inp2 = "race a car"
inp3 = " "

inputs = [inp1, inp2, inp3]

class Solution:
    def validPalindrome(self,s:str)->bool:
        left, right = 0, len(s)-1
        while left<right:
            while left<right and not s[left].isalnum():
                left+=1
            while left<right and not s[right].isalnum():
                right-=1
            if s[left].lower()!=s[right].lower(): return False
            left+=1
            right-=1
        return True

if __name__=='__main__':
    sol = Solution()
    for i in inputs:
        ans = sol.validPalindrome(i)
        print(ans)
