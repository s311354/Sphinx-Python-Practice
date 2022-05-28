"""
Use the SolutionCase class to represent unit testing framework
"""

import unittest.mock
try:
    from leetcode.impl import solution
except ImportError:
    from impl import solution


class SolutionCase(unittest.TestCase):
    """
    A :class:`~leetcode.mocktest.SolutionCase` object is the unittest module.

    This module can construct and run tests by using assertEqual(a, b) assert methods to check for and report failures.

    If the unittest unit testing framework is correct, then the script produces an output that looks like this::

        ----------------------------------------------------------------------
        Ran 2 tests in 1.314s
    """

    def test_twoSum(self):
        """1. Two Sum"""
        sol = solution.Solution()

        nums = [2, 7, 11, 15]
        target = 9

        expected_output = [0, 1]
        self.assertEqual(sol.twoSum(nums, target), expected_output)

        nums = [3, 2, 4]
        target = 6

        expected_output = [1, 2]
        self.assertEqual(sol.twoSum(nums, target), expected_output)

    def test_minNumberOfSemesters(self):
        """ 1494. Parallel Courses II """
        sol = solution.Solution()
        n = 4
        dependencies = [[2, 1], [3, 1], [1, 4]]
        k = 2

        expected_output = 3
        self.assertEqual(sol.minNumberOfSemesters(n, dependencies, k), expected_output)

        n = 12
        dependencies = [[1, 2], [1, 3], [7, 5], [7, 6], [4, 8], [8, 9], [9, 10], [10, 11], [11, 12]]
        k = 2

        expected_output = 6
        self.assertEqual(sol.minNumberOfSemesters(n, dependencies, k), expected_output)

        n = 13
        dependencies = [[12,8],[2,4],[3,7],[6,8],[11,8],[9,4],[9,7],[12,4],[11,4],[6,4],[1,4],[10,7],[10,4],[1,7],[1,8],[2,7],[8,4],[10,8],[12,7],[5,4],[3,4],[11,7],[7,4],[13,4],[9,8],[13,8]]
        k = 9
        expected_output = 3
        self.assertEqual(sol.minNumberOfSemesters(n, dependencies, k), expected_output)

        n = 14
        dependencies = [[11,7]]
        k = 2
        expected_output = 7
        self.assertEqual(sol.minNumberOfSemesters(n, dependencies, k), expected_output)

    def test_convertToTitle(self):
        """ 168. Excel Sheet Column Title """
        sol = solution.Solution()

        columnNumber = 1
        expected_output = "A"
        self.assertEqual(sol.convertToTitle(columnNumber), expected_output)

        columnNumber = 28
        expected_output = "AB"
        self.assertEqual(sol.convertToTitle(columnNumber), expected_output)

    def test_isMatch(self):
        """ 10. Regular Expression Matching """
        sol = solution.Solution()

        s = "aa"
        p = "a*"
        expected_output = True
        self.assertEqual(sol.isMatch(s, p), expected_output)

    def test_romanToInt(self):
        """ 13. Roman to Integer """
        sol = solution.Solution()

        s = "III"
        expected_output = 3
        self.assertEqual(sol.romanToInt(s), expected_output)

        s = "LVIII"
        expected_output = 58
        self.assertEqual(sol.romanToInt(s), expected_output)

        s = "LV"
        expected_output = 55
        self.assertEqual(sol.romanToInt(s), expected_output)

        s = "MCMXCIV"
        expected_output = 1994
        self.assertEqual(sol.romanToInt(s), expected_output)

    def test_maxLength(self):
        """ 1239. Maximum Length of a Concatenated String with Unique Characters """
        sol = solution.Solution()

        arr = ["un", "iq", "ue"]
        expected_output = 4
        self.assertEqual(sol.maxLength(arr), expected_output)

        arr = ["cha", "r", "act", "ers"]
        expected_output = 6
        self.assertEqual(sol.maxLength(arr), expected_output)

    def test_criticalConnections(self):
        """ 1192. Critical Connections in a Network """
        sol = solution.Solution()

        n = 4
        connections = [[0,1],[1,2],[2,0],[1,3]]
        expected_output = [[1,3]]
        self.assertEqual(sol.criticalConnections(n, connections), expected_output)

        n = 2
        connections = [[0,1]]
        expected_output = [[0,1]]
#         self.assertEqual(sol.criticalConnections(n, connections), expected_output)

        n = 6
        connections = [[0,1],[1,2],[2,0],[1,3],[3,4],[4,5],[5,3]]
        expected_output = [[1,3]]
#         self.assertEqual(sol.criticalConnections(n, connections), expected_output)

    def test_arrayNesting(self):
        """565. Array Nesting docstring for arrayNesting"""
        sol = solution.Solution()

        nums = [0,1,2]
        expected_output = 1
        self.assertEqual(sol.arrayNesting(nums), expected_output)

        nums = [5,4,0,3,1,6,2]
        expected_output = 4
        self.assertEqual(sol.arrayNesting(nums), expected_output)

    def test_findPeakElement(self):
        """162. Find Peak Element docstring for findPeakElement"""
        sol = solution.Solution()

        nums = [1,2,3,1]
        expected_output = 2
        self.assertEqual(sol.findPeakElement(nums), expected_output)

        nums = [1,2,1,3,5,6,4]
        expected_output = 5
        self.assertEqual(sol.findPeakElement(nums), expected_output)

    def test_judgeCircle(self):
        """657. Robot Return to Origin docstring for judgeCircle"""
        sol = solution.Solution()

        moves = "UD"
        expected_output = True
        self.assertEqual(sol.judgeCircle(moves), expected_output)

        moves = "LL"
        expected_output = False
        self.assertEqual(sol.judgeCircle(moves), expected_output)

    def test_longestStrChain(self):
        """1048. Longest String Chain docstring for longestStrChain"""
        sol = solution.Solution()

        words = ["a","b","ba","bca","bda","bdca"]
        expected_output = 4
        self.assertEqual(sol.longestStrChain(words), expected_output)

        words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
        expected_output = 5
        self.assertEqual(sol.longestStrChain(words), expected_output)

    def test_lengthOfLongestSubstring(self):
        """3. Longest Substring Without Repeating Characters docstring for lengthOfLongestSubstring"""
        sol = solution.Solution()

        s = "abcabcbb"
        expected_output = 3
        self.assertEqual(sol.lengthOfLongestSubstring(s), expected_output)

        s = "bbbbb"
        expected_output = 1
        self.assertEqual(sol.lengthOfLongestSubstring(s), expected_output)

        s = "pwwkew"
        expected_output = 3
        self.assertEqual(sol.lengthOfLongestSubstring(s), expected_output)

    def test_minimumCardPickup(self):
        """2260. Minimum Consecutive Cards to Pick Up docstring for minimumCardPickup"""
        sol = solution.Solution()

        cards = [3,4,2,3,4,7]
        expected_output = 4
        self.assertEqual(sol.minimumCardPickup(cards), expected_output)

        cards = [1, 0, 5, 3]
        expected_output = -1
        self.assertEqual(sol.minimumCardPickup(cards), expected_output)

    def test_findCircleNum(self):
        """547. Number of Provinces docstring for findCircleNum"""
        sol = solution.Solution()

        isConnected = [[1,1,0],[1,1,0],[0,0,1]]
        expected_output = 2
        self.assertEqual(sol.findCircleNum(isConnected), expected_output)

        isConnected = [[1,0,0],[0,1,0],[0,0,1]]
        expected_output = 3
        self.assertEqual(sol.findCircleNum(isConnected), expected_output)

    def test_canFinish(self):
        """207. Course Schedule docstring for canFinish"""
        sol = solution.Solution()

        numCourses = 2
        prerequisites = [[1,0],[0,1]]
        expected_output = False
        self.assertEqual(sol.canFinish(numCourses, prerequisites), expected_output)

        numCourses = 2
        prerequisites = [[1,0]]
        expected_output = True
        self.assertEqual(sol.canFinish(numCourses, prerequisites), expected_output)

    def test_lengthOfLIS(self):
        """300. Longest Increasing Subsequence docstring for lengthOfLIS"""
        sol = solution.Solution()

        nums = [10,9,2,5,3,7,101,18]
        expected_output = 4
        self.assertEqual(sol.lengthOfLIS(nums), expected_output)

        nums = [0,1,0,3,2,3]
        expected_output = 4
#         self.assertEqual(sol.lengthOfLIS(nums), expected_output)

        nums = [7,7,7,7,7,7,7]
        expected_output = 1
        self.assertEqual(sol.lengthOfLIS(nums), expected_output)

    def test_originalDigits(self):
        """423. Reconstruct Original Digits from English docstring for originalDigits"""
        sol = solution.Solution()

        s = "owoztneoer"
        expected_output = "012"
        self.assertEqual(sol.originalDigits(s), expected_output)

        s = "fviefuro"
        expected_output = "45"
        self.assertEqual(sol.originalDigits(s), expected_output)

    def test_strongPasswordChecker(self):
        """420. Strong Password Checker docstring for strongPasswordChecker"""
        sol = solution.Solution()

        password = "1337C0d3"
        expected_output = 0
        self.assertEqual(sol.strongPasswordChecker(password), expected_output)

        password = "aA1"
        expected_output = 3
        self.assertEqual(sol.strongPasswordChecker(password), expected_output)

    def test_maxProfit(self):
        """188. Best Time to Buy and Sell Stock IV docstring for maxProfit"""
        sol = solution.Solution()

        k = 2
        prices = [2,4,1]
        expected_output = 2
        self.assertEqual(sol.maxProfit(k, prices), expected_output)

        k = 2
        prices = [3,2,6,5,0,3]
        expected_output = 7
        self.assertEqual(sol.maxProfit(k, prices), expected_output)

    def test_searchRange(self):
        """34. Find First and Last Position of Element in Sorted Array docstring for searchRange"""
        sol = solution.Solution()

        nums = [5,7,7,8,8,10]
        target = 8
        expected_output = [3, 4]
        self.assertEqual(sol.searchRange(nums, target), expected_output)

        nums = [5,7,7,8,8,10]
        target = 6
        expected_output = [-1, -1]
        self.assertEqual(sol.searchRange(nums, target), expected_output)

    def test_maxSubArray(self):
        """53. Maximum Subarray docstring for maxSubArray"""
        sol = solution.Solution()

        nums = [-2,1,-3,4,-1,2,1,-5,4]
        expected_output = 6
        self.assertEqual(sol.maxSubArray(nums), expected_output)

        nums = [5,4,-1,7,8]
        expected_output = 23
        self.assertEqual(sol.maxSubArray(nums), expected_output)

        nums = [1]
        expected_output = 1
        self.assertEqual(sol.maxSubArray(nums), expected_output)

    def test_minPathSum(self):
        """64. Minimum Path Sum docstring for minPathSum"""
        sol = solution.Solution()

        grid = [[1,3,1],[1,5,1],[4,2,1]]
        expected_output = 7
        self.assertEqual(sol.minPathSum(grid), expected_output)
        grid = [[1,2,3],[4,5,6]]
        expected_output = 12
        self.assertEqual(sol.minPathSum(grid), expected_output)

    def test_simplifyPath(self):
        """71. Simplify Path docstring for simplifyPath"""
        sol = solution.Solution()

        path = "/home/"
        expected_output = "/home"
        self.assertEqual(sol.simplifyPath(path), expected_output)

        path = "/../"
        expected_output = "/"
        self.assertEqual(sol.simplifyPath(path), expected_output)

        path = "/home//foo/"
        expected_output = "/home/foo"
        self.assertEqual(sol.simplifyPath(path), expected_output)

    def test_subsets(self):
        """78. Subsets docstring for subsets"""
        sol = solution.Solution()

        nums = [1,2,3]
        expected_output = [[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]
        self.assertEqual(sol.subsets(nums), expected_output)


        nums = [0]
        expected_output = [[],[0]]
        self.assertEqual(sol.subsets(nums), expected_output)

    def test_numDecodings(self):
        """91. Decode Ways docstring for numDecodings"""
        sol = solution.Solution()

        s = "12"
        expected_output = 2
        self.assertEqual(sol.numDecodings(s), expected_output)

        s = "226"
        expected_output = 3
        self.assertEqual(sol.numDecodings(s), expected_output)

        s = "06"
        expected_output = 0
        self.assertEqual(sol.numDecodings(s), expected_output)

    def test_getMinimumDays(self):
        """Interview docstring for getMinimumDays"""
        sol = solution.Solution()

        parcels = [4, 2, 3, 4]
        expected_output = 3
        self.assertEqual(sol.getMinimumDays(parcels), expected_output)

        parcels = [5, 2, 2, 5, 1]
        expected_output = 3
        self.assertEqual(sol.getMinimumDays(parcels), expected_output)

        parcels = [5, 2, 2, 5, 1, 3]
        expected_output = 4
        self.assertEqual(sol.getMinimumDays(parcels), expected_output)

    def test_longestNiceSubstring(self):
        """1763. Longest Nice Substring docstring for longestNiceSubstring"""
        sol = solution.Solution()

        s = "YazaAay"
        expected_output = "aAa"
        self.assertEqual(sol.longestNiceSubstring(s), expected_output)

        s = "abABB"
        expected_output = "abABB"
        self.assertEqual(sol.longestNiceSubstring(s), expected_output)





if __name__ == '__main__':
    unittest.main()
