Readme File
===========

Official Big-O Cheat Sheet
------------------------------

Here is the Offical Big-O Sheat Poster:

.. image:: images/bigo_cheat.png
    :width: 1000

Here is the Common Data Structure Operations:

.. image:: images/common_data_struct.png
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
    1, :func:`~leetcode.impl.solution.Solution.twoSum`, Easy, O(N), O(N), HashMap
    168, :func:`~leetcode.impl.solution.Solution.convertToTitle`, Medium, O(log N), O(log N), basic
    10, :func:`~leetcode.impl.solution.Solution.isMatch`, Hard, O(NM), O(NM), Dynamic Programming
    13, :func:`~leetcode.impl.solution.Solution.romanToInt`, Easy, O(N), O(log N), Basic
    1239, :func:`~leetcode.impl.solution.Solution.maxLength`, Medium, O(N), O(N), DFS
    1192, :func:`~leetcode.impl.solution.Solution.criticalConnections`, Hard, O(N), O(N + M), DFS
    565, :func:`~leetcode.impl.solution.Solution.arrayNesting`, Medium, O(N), O(1), Basic
    162, :func:`~leetcode.impl.solution.Solution.findPeakElement`, Medium, O(log N) , O(1), Binary Search
    657, :func:`~leetcode.impl.solution.Solution.judgeCircle`, Easy, O(N) , O(1), Basic
    1048, :func:`~leetcode.impl.solution.Solution.longestStrChain`, Medium, O(N^2), O(N), Stack
    3, :func:`~leetcode.impl.solution.Solution.lengthOfLongestSubstring`, Medium, O(N), O(N), Two Pointers
    2260, :func:`~leetcode.impl.solution.Solution.minimumCardPickup`, Medium, O(N), O(N), Two Pointers
    547, :func:`~leetcode.impl.solution.Solution.findCircleNum`, Medium, O(N^2), O(N^2), DFS
    207, :func:`~leetcode.impl.solution.Solution.canFinish`, Medium, O(N + M), O(N + M), Topological Sorting
    300, :func:`~leetcode.impl.solution.Solution.lengthOfLIS`, Medium, O(N^2), O(N), Dynamic Programming
    64, :func:`~leetcode.impl.solution.Solution.minPathSum`, Medium, O(NM), O(NM), DFS
    34, :func:`~leetcode.impl.solution.Solution.searchRange`, Medium, O(log N), O(1), Stack
    53, :func:`~leetcode.impl.solution.Solution.maxSubArray`, Easy, O(N), O(1), Basic
    71, :func:`~leetcode.impl.solution.Solution.simplifyPath`, Medium, O(N), O(N), Stack
    78, :func:`~leetcode.impl.solution.Solution.subsets`, Medium, O(N*2^N), O(N*2^N), Backtracking
    91, :func:`~leetcode.impl.solution.Solution.numDecodings`, Medium, O(N), O(N), Dynamic Programming
    1763, :func:`~leetcode.impl.solution.Solution.longestNiceSubstring`, Easy, O(N* log N), O(N), Divide and Conquer
    217, :func:`~leetcode.impl.solution.Solution.containDuplicate`, Easy, O(N), O(N), HashMap
    283, :func:`~leetcode.impl.solution.Solution.moveZeroes`, Easy, O(N), O(1), Fast and Slow Pointers
    36, :func:`~leetcode.impl.solution.Solution.isValidSudoku`, Medium, O(N^2), O(N), BFS
    1704, :func:`~leetcode.impl.solution.Solution.halvesAreAlike`, Easy, O(N), O(1), Two Pointers
    122, :func:`~leetcode.impl.solution.Solution.maxProfitII`, Medium, O(N), O(1), Basic
    121, :func:`~leetcode.impl.solution.Solution.maxProfit`, Easy, O(N), O(1), Dynamic Programming
    714, :func:`~leetcode.impl.solution.Solution.maxProfitwithfee`, Medium, O(N), O(1), Dynamic Programming
    944, :func:`~leetcode.impl.solution.Solution.minDeletionSize`, Easy, O(NM), O(1), Basic
    44, :func:`~leetcode.impl.solution.Solution.WildcardisMatch`, Hard, O(NM), O(N), Dynamic Programming
    2280, :func:`~leetcode.impl.solution.Solution.minimumLines`, Medium, O(N log N), O(1), Basic
    496, :func:`~leetcode.impl.solution.Solution.nextGreaterElement`, Easy, O(N + M), O(M), Stack
    503, :func:`~leetcode.impl.solution.Solution.nextGreaterElementsII`, Medium, O(N), O(N), Stack
    739, :func:`~leetcode.impl.solution.Solution.dailyTemperatures`, Medium, O(N), O(N), Stack
    2281, :func:`~leetcode.impl.solution.Solution.totalStrength`, Hard, O(N^2), O(1), Basic
    100, :func:`~leetcode.impl.solution.Solution.isSameTree`, Easy, O(N), O(N), Tree Node
    2134, :func:`~leetcode.impl.solution.Solution.minSwaps`, Medium, O(N), O(1), Sliding Window
    1920, :func:`~leetcode.impl.solution.Solution.buildArray`, Easy, O(N), O(1), Basic
    1480, :func:`~leetcode.impl.solution.Solution.runningSum`, Easy, O(N), O(1), Basic
    215, :func:`~leetcode.impl.solution.Solution.findKthLargest`, Medium, O(N), O(1), Quick Select
    973, :func:`~leetcode.impl.solution.Solution.kClosest`, Medium, O(N log N), O(1), Quick Select
    15, :func:`~leetcode.impl.solution.Solution.threeSum`, Medium, O(N^2), O(N), Two Pointers
    8, :func:`~leetcode.impl.solution.Solution.myAtoi`, Medium, O(N), O(1), Basic
    125, :func:`~leetcode.impl.solution.Solution.isPalindrome`, Easy, O(log N), O(1), Two Pointers
    9, :func:`~leetcode.impl.solution.Solution.isPalindromeNum`, Easy, O(log N), O(1), Two Pointers
    1496, :func:`~leetcode.impl.solution.Solution.isPathCrossing`, Easy, O(N), O(N), HashMap
    290, :func:`~leetcode.impl.solution.Solution.wordPattern`, Easy, O(N), O(N), HashMap
    58, :func:`~leetcode.impl.solution.Solution.lengthOfLastWord`, Easy, O(1), O(N), Basic
    14, :func:`~leetcode.impl.solution.Solution.longestCommonPrefix`, Easy, O(NM), O(1), Basic
    373, :func:`~leetcode.impl.solution.Solution.kSmallestPairs`, Medium, O(N*log N) , O(N), BFS
    704, :func:`~leetcode.impl.solution.Solution.search`, Easy, O(log N) , O(1), Binary Search
    733, :func:`~leetcode.impl.solution.Solution.floodFill`, Easy, O(N*M) , O(N*M), DFS

.. mdinclude:: ../../README.md






