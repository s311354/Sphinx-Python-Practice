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
        self.assertEqual(sol.criticalConnections(n, connections), expected_output)

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



if __name__ == '__main__':
    unittest.main()
