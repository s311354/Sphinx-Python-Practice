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


if __name__ == '__main__':
    unittest.main()
