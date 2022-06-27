"""
Use the SolutionCase class to represent unit testing framework
"""

import unittest.mock
try:
    from leetcode.impl import solution
    from leetcode.impl import solution_design
except ImportError:
    from impl import solution
    from impl import solution_design


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

        s = "aab"
        p = "c*a*b"
        expected_output = True
        self.assertEqual(sol.isMatch(s, p), expected_output)

        s = "mississippi"
        p = "mis*is*ip*."
        expected_output = True
        self.assertEqual(sol.isMatch(s, p), expected_output)

        s = "aaa"
        p = "ab*a*c*a"
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

        arr = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]
        expected_output = 16
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
#         self.assertEqual(sol.findCircleNum(isConnected), expected_output)

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

    def testmaxProfitIV(self):
        """188. Best Time to Buy and Sell Stock IV docstring for maxProfit"""
        sol = solution.Solution()

        k = 2
        prices = [2,4,1]
        expected_output = 2
        self.assertEqual(sol.maxProfitIV(k, prices), expected_output)

        k = 2
        prices = [3,2,6,5,0,3]
        expected_output = 7
        self.assertEqual(sol.maxProfitIV(k, prices), expected_output)

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
        expected_output = [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3], []]
        self.assertEqual(sol.subsets(nums), expected_output)

        nums = [0]
        expected_output = [[0],[]]
        self.assertEqual(sol.subsets(nums), expected_output)

    def test_numDecodings(self):
        """91. Decode Ways docstring for numDecodings"""
        sol = solution.Solution()

        s = "12"
        expected_output = 2
#         self.assertEqual(sol.numDecodings(s), expected_output)

        s = "226"
        expected_output = 3
        self.assertEqual(sol.numDecodings(s), expected_output)

        s = "06"
        expected_output = 0
#         self.assertEqual(sol.numDecodings(s), expected_output)

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
#         self.assertEqual(sol.longestNiceSubstring(s), expected_output)


    def test_containDuplicate(self):
        """217. Contains Duplicate  docstring for containDuplicate"""
        sol = solution.Solution()

        nums = [1,2,3,1]
        expected_output = True
        self.assertEqual(sol.containDuplicate(nums), expected_output)


        nums = [1,1,1,3,3,4,3,2,4,2]
        expected_output = True
        self.assertEqual(sol.containDuplicate(nums), expected_output)

    def test_moveZeroes(self):
        """283. Move Zeroes  docstring for moveZeroes"""
        sol = solution.Solution()

        nums = [0,1,0,3,12]
        expected_output = [1,3,12,0,0]
        self.assertEqual(sol.moveZeroes(nums), expected_output)

        nums = [0]
        expected_output = [0]
        self.assertEqual(sol.moveZeroes(nums), expected_output)

    def test_isValidSudoku(self):
        """36. Valid Sudoku docstring for isValidSudoku"""
        sol = solution.Solution()

        board = [["5","3",".",".","7",".",".",".","."]
                ,["6",".",".","1","9","5",".",".","."]
                ,[".","9","8",".",".",".",".","6","."]
                ,["8",".",".",".","6",".",".",".","3"]
                ,["4",".",".","8",".","3",".",".","1"]
                ,["7",".",".",".","2",".",".",".","6"]
                ,[".","6",".",".",".",".","2","8","."]
                ,[".",".",".","4","1","9",".",".","5"]
                ,[".",".",".",".","8",".",".","7","9"]]
        expected_output = True
        self.assertEqual(sol.isValidSudoku(board), expected_output)

        board = [["8","3",".",".","7",".",".",".","."]
            ,["6",".",".","1","9","5",".",".","."]
            ,[".","9","8",".",".",".",".","6","."]
            ,["8",".",".",".","6",".",".",".","3"]
            ,["4",".",".","8",".","3",".",".","1"]
            ,["7",".",".",".","2",".",".",".","6"]
            ,[".","6",".",".",".",".","2","8","."]
            ,[".",".",".","4","1","9",".",".","5"]
            ,[".",".",".",".","8",".",".","7","9"]]
        expected_output = False
        self.assertEqual(sol.isValidSudoku(board), expected_output)

    def test_halvesAreAlike(self):
        """1704. Determine if String Halves Are Alike docstring for halvesAreAlike"""
        sol = solution.Solution()

        s = "book"
        expected_output = True
        self.assertEqual(sol.halvesAreAlike(s), expected_output)

        s = "textbook"
        expected_output = False
        self.assertEqual(sol.halvesAreAlike(s), expected_output)

    def test_maxProfitII(self):
        """122. Best Time to Buy and Sell Stock II docstring for maxProfitII"""
        sol = solution.Solution()

        prices = [7,1,5,3,6,4]
        expected_output = 7
        self.assertEqual(sol.maxProfitII(prices), expected_output)

        prices = [1,2,3,4,5]
        expected_output = 4
        self.assertEqual(sol.maxProfitII(prices), expected_output)

    def test_maxProfit(self):
        """ 121. Best Time to Buy and Sell Stock """
        sol = solution.Solution()

        prices = [7,1,5,3,6,4]
        expected_output = 5
        self.assertEqual(sol.maxProfit(prices), expected_output)

        prices = [7,6,4,3,1]
        expected_output = 0
        self.assertEqual(sol.maxProfit(prices), expected_output)

    def test_maxProfitwithfee(self):
        """714. Best Time to Buy and Sell Stock with Transaction Fee """
        sol = solution.Solution()

        prices = [1,3,2,8,4,9]
        fee = 2
        expected_output = 8
        self.assertEqual(sol.maxProfitwithfee(prices, fee), expected_output)

        prices = [1,3,7,5,10,3]
        fee = 3
        expected_output = 6
        self.assertEqual(sol.maxProfitwithfee(prices, fee), expected_output)

    def test_minDeletionSize(self):
        """docstring for minDeletionSize"""
        sol = solution.Solution()

        strs = ["cba","daf","ghi"]
        expected_output = 1
        self.assertEqual(sol.minDeletionSize(strs), expected_output)

        strs = ["zyx","wvu","tsr"]
        expected_output = 3
        self.assertEqual(sol.minDeletionSize(strs), expected_output)

    def test_WildcardisMatch(self):
        """ 44. Wildcard Matching """
        sol = solution.Solution()

        s = "aa"
        p = "a"
        expected_output = False
        self.assertEqual(sol.WildcardisMatch(s, p), expected_output)

        s = "aa"
        p = "*"
        expected_output = True
        self.assertEqual(sol.WildcardisMatch(s, p), expected_output)

        s = "cb"
        p = "?a"
        expected_output = False
        self.assertEqual(sol.WildcardisMatch(s, p), expected_output)

    def test_nextGreaterElement(self):
        """docstring for nextGreaterElement"""
        sol = solution.Solution()

        nums1 = [4,1,2]
        nums2 = [1,3,4,2]
        expected_output = [-1,3,-1]
        self.assertEqual(sol.nextGreaterElement(nums1, nums2), expected_output)

        nums1 = [2,4]
        nums2 = [1,2,3,4]
        expected_output = [3,-1]
        self.assertEqual(sol.nextGreaterElement(nums1, nums2), expected_output)

    def test_nextGreaterElementsII(self):
        """docstring for nextGreaterElementII"""
        sol = solution.Solution()

        nums = [1,2,1]
        expected_output = [2,-1,2]
        self.assertEqual(sol.nextGreaterElementsII(nums), expected_output)

        nums = [1,2,3,4,3]
        expected_output = [2,3,4,-1,4]
        self.assertEqual(sol.nextGreaterElementsII(nums), expected_output)

    def test_totalStrength(self):
        """docstring for totalStrength"""
        sol = solution.Solution()

        strength = [1,3,1,2]
        expected_output = 44
        self.assertEqual(sol.totalStrength(strength), expected_output)

        strength = [5,4,6]
        expected_output = 213
        self.assertEqual(sol.totalStrength(strength), expected_output)

    def test_StreamChecker(self):
        """docstring for StreamChecker"""
        streamChecker = solution_design.StreamChecker(["cd", "f", "kl"])

        expected_output = False
        self.assertEqual(streamChecker.query("a"), expected_output)

        expected_output = False
        self.assertEqual(streamChecker.query("b"), expected_output)

        expected_output = True
        self.assertEqual(streamChecker.query("d"), expected_output)

    def test_isSameTree(self):
        """ 100. Same Tree docstring for isSameTree"""
        p = [1,2,3]
        q = [1,2,3]

        root_p = solution.TreeNode(p[0])
        for item in range(1, len(p)):
            root_p.insert(p[item])

        root_q = solution.TreeNode(q[0])
        for item in range(1, len(p)):
            root_q.insert(q[item])

        sol = solution.Solution()

        expected_output = True
        self.assertEqual(sol.isSameTree(root_p, root_q), expected_output)

    def test_minSwaps(self):
        """docstring for minSwaps"""
        sol = solution.Solution()

        nums = [0,1,0,1,1,0,0]
        expected_output = 1
        self.assertEqual(sol.minSwaps(nums), expected_output)

        nums = [0,1,1,1,0,0,1,1,0]
        expected_output = 2
        self.assertEqual(sol.minSwaps(nums), expected_output)

        nums = [1,1,0,0,1]
        expected_output = 0
        self.assertEqual(sol.minSwaps(nums), expected_output)

    def test_merge(self):
        """docstring for merge"""
        sol = solution.Solution()

        intervals = [[1,3],[2,6],[8,10],[15,18]]
        expected_output = [[1,6],[8,10],[15,18]]
        self.assertEqual(sol.merge(intervals), expected_output)

        intervals = [[1,4],[4,5]]
        expected_output = [[1,5]]
        self.assertEqual(sol.merge(intervals), expected_output)


    def test_searchInsert(self):
        """docstring for searchInsert"""
        sol = solution.Solution()

        nums = [1,3,5,6]
        target = 5
        expected_output = 2
        self.assertEqual(sol.searchInsert(nums, target), expected_output)

        nums = [1,3,5,6]
        target = 2
        expected_output = 1
        self.assertEqual(sol.searchInsert(nums, target), expected_output)

    def test_findLucky(self):
        """ 1394. Find Lucky Integer in an Array docstring for findLucky"""
        sol = solution.Solution()

        arr = [2,2,3,4]
        expected_output = 2
        self.assertEqual(sol.findLucky(arr), expected_output)

        arr = [2,2,2,3,3]
        expected_output = -1
        self.assertEqual(sol.findLucky(arr), expected_output)

    def test_isValid(self):
        """docstring for isValid"""
        sol = solution.Solution()

        s = "()"
        expected_output = True
        self.assertEqual(sol.isValid(s), expected_output)

        s = "(]"
        expected_output = False
        self.assertEqual(sol.isValid(s), expected_output)

    def test_mergeTwoLists(self):
        """docstring for mergeTwoLists"""
        sol = solution.Solution()

        list1 = [1,2,4]
        list2 = [1,3,4]

        List1 = solution.ListNode(list1[0])
        for item in range(1, len(list1)):
            List1.insert(list1[item])

        List2 = solution.ListNode(list2[0])
        for item in range(1, len(list2)):
            List2.insert(list2[item])

        output = [1,1,2,3,4,4]
        expected_output = solution.ListNode(output[0])
        for item in range(1, len(output)):
            expected_output.insert(output[item])

        self.assertEqual(sol.mergeTwoLists(List1, List2).PrintListNode(), expected_output.PrintListNode())

    def test_removeNthFromEnd(self):
        """docstring for removeNthFromEnd"""
        sol = solution.Solution()

        list1 = [1,2,3,4,5]
        List1 = solution.ListNode(list1[0])
        for item in range(1, len(list1)):
            List1.insert(list1[item])
        n = 2

        output = [1,2,3,5]
        expected_output = solution.ListNode(output[0])
        for item in range(1, len(output)):
            expected_output.insert(output[item])

        self.assertEqual(sol.removeNthFromEnd(List1, n).PrintListNode(), expected_output.PrintListNode())

    def test_coinChange(self):
        """docstring for coinChange"""
        sol = solution.Solution()

        coins = [1,2,5]
        amount = 11
        expected_output = 3
        self.assertEqual(sol.coinChange(coins, amount), expected_output)

    def test_countBits(self):
        """ 338. Counting Bits docstring for countBits"""
        sol = solution.Solution()

        n = 2
        expected_output = [0,1,1]
#         self.assertEqual(sol.countBits(n), expected_output)

        n = 5
        expected_output = [0,1,1,2,1,2]
#         self.assertEqual(sol.countBits(n), expected_output)

        n = 7
        expected_output = [0,1,1,2,1,2,2,3]
        self.assertEqual(sol.countBits(n), expected_output)

    def test_getSum(self):
        """docstring for getSum"""
        sol = solution.Solution()

        a, b = 2, 3
        expected_output = 5
        self.assertEqual(sol.getSum(a, b), expected_output)

    def test_numIslands(self):
        """ 200. Number of Islands docstring for numIslands"""
        sol = solution.Solution()

        grid = [
              ["1","1","1","1","0"],
              ["1","1","0","1","0"],
              ["1","1","0","0","0"],
              ["0","0","0","0","0"]
            ]
        expected_output = 1
        self.assertEqual(sol.numIslands(grid), expected_output)

        grid = [
              ["1","1","0","0","0"],
              ["1","1","0","0","0"],
              ["0","0","1","0","0"],
              ["0","0","0","1","1"]
            ]
        expected_output = 3
        self.assertEqual(sol.numIslands(grid), expected_output)

    def test_strStr(self):
        """docstring for strStr"""
        sol = solution.Solution()

        haystack = "hello"
        needle = "ll"
        expected_output = 2
        self.assertEqual(sol.strStr(haystack, needle), expected_output)

    def test_myAtoi(self):
        """docstring for myAtoi"""
        sol = solution.Solution()

        s = "-42"
        expected_output = -42
        self.assertEqual(sol.myAtoi(s), expected_output)

    def test_search(self):
        """docstring for search"""
        sol = solution.Solution()

        nums = [4,5,6,7,0,1,2]
        target = 0
        expected_output = 4
        self.assertEqual(sol.search(nums, target), expected_output)

    def test_permute(self):
        """docstring for permute"""
        sol = solution.Solution()

        nums = [1,2,3]
        expected_output = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
        self.assertEqual(sol.permute(nums), expected_output)

    def test_combinationSum(self):
        """docstring for combinationSum"""
        sol = solution.Solution()

        candidates = [2,3,6,7]
        target = 7
        expected_output = [[2,2,3],[7]]
        self.assertEqual(sol.combinationSum(candidates, target), expected_output)

    def test_combinationSum2(self):
        """docstring for combinationSum2"""
        sol = solution.Solution()

        candidates = [10,1,2,7,6,1,5]
        target = 8
        expected_output = [[1,1,6],[1,2,5],[1,7],[2,6]]
        self.assertEqual(sol.combinationSum2(candidates, target), expected_output)

    def test_restoreIpAddresses(self):
        """docstring for restoreIpAddresses"""
        sol = solution.Solution()

        s = "25525511135"
        expected_output = ["255.255.11.135","255.255.111.35"]
        self.assertEqual(sol.restoreIpAddresses(s), expected_output)

        s = "101023"
        expected_output = ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
        self.assertEqual(sol.restoreIpAddresses(s), expected_output)

    def test_letterCombinations(self):
        """docstring for letterCombinations"""
        sol = solution.Solution()

        digits = "23"
        expected_output = ["ad","ae","af","bd","be","bf","cd","ce","cf"]
        self.assertEqual(sol.letterCombinations(digits), expected_output)

    def test_minOperations(self):
        """docstring for minOperations"""
        sol = solution.Solution()

        nums = [1,1,1]
        expected_output = 3
        self.assertEqual(sol.minOperations(nums), expected_output)

        nums = [2, 3, 2, 3]
        expected_output = 4
        self.assertEqual(sol.minOperations(nums), expected_output)

        nums = [1, 4, 3, 2]
        expected_output = 6
        self.assertEqual(sol.minOperations(nums), expected_output)

    def test_rob(self):
        """docstring for rob"""
        sol = solution.Solution()

        nums = [1,2,3,1]
        expected_output = 4
        self.assertEqual(sol.rob(nums), expected_output)

        nums = [2,7,9,3,1]
        expected_output = 12
        self.assertEqual(sol.rob(nums), expected_output)

    def test_reverseBits(self):
        """docstring for reverse"""
        sol = solution.Solution()

        n = 0b00000010100101000001111010011100
        expected_output = 964176192
        self.assertEqual(sol.reverseBits(n), expected_output)

        n = 0b11111111111111111111111111111101
        expected_output = 3221225471
        self.assertEqual(sol.reverseBits(n), expected_output)

    def test_removeDuplicates(self):
        """docstring for removeDuplicates"""
        sol = solution.Solution()

        nums = [0,0,1,1,1,2,2,3,3,4]
        expected_output = 5
        self.assertEqual(sol.removeDuplicates(nums), expected_output)

    def test_rob(self):
        """docstring for rob"""
        sol = solution.Solution()

        nums = [7,2,9,16,1]
        expected_output = 23
        self.assertEqual(sol.rob(nums), expected_output)

    def test_maxArea(self):
        """docstring for maxArea"""
        sol = solution.Solution()

        height = [1,8,6,2,5,4,8,3,7]
        expected_output = 49
        self.assertEqual(sol.maxArea(height), expected_output)

    def test_myPow(self):
        """docstring for myPow"""
        sol = solution.Solution()

        x = 2.00
        n = -2
        expected_output = 0.25
        self.assertEqual(sol.myPow(x, n), expected_output)

    def test_groupAnagrams(self):
        """docstring for gr"""
        sol = solution.Solution()

        strs = ["eat","tea","tan","ate","nat","bat"]
        expected_output = [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
        self.assertEqual(sol.groupAnagrams(strs), expected_output)

    def test_wordBreak(self):
        """docstring for wordBreak"""
        sol = solution.Solution()

        s = "leetcode"
        wordDict = ["leet","code"]
        expected_output = True
        self.assertEqual(sol.wordBreak(s, wordDict), expected_output)

    def test_dailyTemperatures(self):
        """docstring for dailyTemperatures"""
        sol = solution.Solution()

        temperatures = [73,74,75,71,69,72,76,73]
        expected_output = [1,1,4,2,1,1,0,0]
        self.assertEqual(sol.dailyTemperatures(temperatures), expected_output)



if __name__ == '__main__':
    unittest.main()
