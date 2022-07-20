Readme File
===========

Official Big-O Cheat Sheet
------------------------------

.. image:: images/bigo_cheat.png
    :width: 1000

How to calculate the time and space complexity?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The depth of recursion can affect the space, required by an algorithm. Each recursive function call usually needs to allocate some additional memory (for temproary data) to process its argument, so a recursive algorithm will require space O(depth of recursion).

The examples of linear sum algorithm and binary sum algorithm::

    def linear_sum(S: List[int], stop: int) -> int:
        """ Using linear recursion to calculate sum of all elements of the array """

        # time:O(N) space:O(N)

        if (stop == 1):
            return S[0]
        else:
            return linear_sum(S, stop-1) + S[stop-1]

    def binary_sum(S: List[int], start: int, stop: int) -> int:
        """  Using binary recursion to calculate sum of all elements of array
        Return the sum of the numbers in implicit slice S[start:stop]. """

        # time:O(N) space:O(log n)

        if start >= stop:                      # zero elements in slice
            return 0
        elif start == stop-1:                  # one element in slice
            return S[start]
        else:                                  # two or more elements in slice
            mid = (start + stop) // 2

        return binary_sum(S, start, mid) + binary_sum(S, mid, stop)

The binary sum algorithm uses O(log n) improves over the linear sum algorithm uses O(n).

Solutions list 
---------------------

.. csv-table:: `Leetcode Solutions <https://leetcode.com/qazqazqaz850/>`_
    :header-rows: 1
    :stub-columns: 0

    #, Title, Level, Time, Space Note, Tags
    1, :func:`~leetcode.impl.solution.Solution.twoSum`, Easy, O(N), O(N), Hashtable
    168, :func:`~leetcode.impl.solution.Solution.convertToTitle`, Medium, O(log N), O(log N), basic
    10, :func:`~leetcode.impl.solution.Solution.isMatch`, Hard, O(NM), O(NM), Dynamic Programming
    13, :func:`~leetcode.impl.solution.Solution.romanToInt`, Easy, O(N), O(1), Basic
    1239, :func:`~leetcode.impl.solution.Solution.maxLength`, Medium, O(NM), O(NM), DFS
    1192, :func:`~leetcode.impl.solution.Solution.criticalConnections`, Hard, , , DFS
    565, :func:`~leetcode.impl.solution.Solution.arrayNesting`, Medium, , , Basic
    162, :func:`~leetcode.impl.solution.Solution.findPeakElement`, Medium, O(log N) , , Binary Search
    657, :func:`~leetcode.impl.solution.Solution.judgeCircle`, Easy, O(N) , O(N), Basic
    1048, :func:`~leetcode.impl.solution.Solution.longestStrChain`, Medium, , , Stack
    3, :func:`~leetcode.impl.solution.Solution.lengthOfLongestSubstring`, Medium, , , Stack
    2260, :func:`~leetcode.impl.solution.Solution.minimumCardPickup`, Medium, , , Hashtable
    547, :func:`~leetcode.impl.solution.Solution.findCircleNum`, Medium, , , DFS
    207, :func:`~leetcode.impl.solution.Solution.canFinish`, Medium, , , DFS
    300, :func:`~leetcode.impl.solution.Solution.lengthOfLIS`, Medium, , , Dynamic Programming
    64, :func:`~leetcode.impl.solution.Solution.minPathSum`, Medium, , , DFS
    34, :func:`~leetcode.impl.solution.Solution.searchRange`, Medium, , , Stack
    53, :func:`~leetcode.impl.solution.Solution.maxSubArray`, Easy, , , Basic
    71, :func:`~leetcode.impl.solution.Solution.simplifyPath`, Medium, , , Stack
    78, :func:`~leetcode.impl.solution.Solution.subsets`, Medium, , , Backtracking
    91, :func:`~leetcode.impl.solution.Solution.numDecodings`, Medium, , , Dynamic Programming
    1763, :func:`~leetcode.impl.solution.Solution.longestNiceSubstring`, Easy, , , DFS
    217, :func:`~leetcode.impl.solution.Solution.containDuplicate`, Easy, , , Hashtable
    283, :func:`~leetcode.impl.solution.Solution.moveZeroes`, Easy, , , Fast and Slow Pointers
    36, :func:`~leetcode.impl.solution.Solution.isValidSudoku`, Medium, , , BFS
    1704, :func:`~leetcode.impl.solution.Solution.halvesAreAlike`, Easy, , , Two Pointers
    122, :func:`~leetcode.impl.solution.Solution.maxProfitII`, Medium, , , Basic
    121, :func:`~leetcode.impl.solution.Solution.maxProfit`, Easy, , , Dynamic Programming
    714, :func:`~leetcode.impl.solution.Solution.maxProfitwithfee`, Medium, , , Dynamic Programming
    944, :func:`~leetcode.impl.solution.Solution.minDeletionSize`, Easy, , , Basic
    44, :func:`~leetcode.impl.solution.Solution.WildcardisMatch`, Hard, , , Dynamic Programming
    2280, :func:`~leetcode.impl.solution.Solution.minimumLines`, Medium, , , Basic
    496, :func:`~leetcode.impl.solution.Solution.nextGreaterElement`, Easy, , , Stack
    503, :func:`~leetcode.impl.solution.Solution.nextGreaterElementsII`, Medium, , , Stack
    739, :func:`~leetcode.impl.solution.Solution.dailyTemperatures`, Medium, , , Stack
    2281, :func:`~leetcode.impl.solution.Solution.totalStrength`, Hard, , , Stack
    100, :func:`~leetcode.impl.solution.Solution.isSameTree`, Easy, , , Tree Node
    2134, :func:`~leetcode.impl.solution.Solution.minSwaps`, Medium, , , Sliding Window
    1920, :func:`~leetcode.impl.solution.Solution.buildArray`, Easy, O(N), O(1), Basic
    1480, :func:`~leetcode.impl.solution.Solution.runningSum`, Easy, O(N), O(1), Basic



.. mdinclude:: ../../README.md






