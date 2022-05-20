"""
Use the Solution class to represent Leedcode problems
"""

from typing import List
from typing import Optional
import collections
from collections import defaultdict
from collections import Counter
from functools import lru_cache


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    """
    A :class:`~leetcode.impl.solution` object is the leetcode quiz module

    This module is to implementate the leetcode problems
    """

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """ Two Sum

        Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
        You may assume that each input would have exactly one solution, and you may not use the same element twice.
        You can return the answer in any order.

        :param num:  array of integers
        :type  num:  List[int]
        :param target:  integer target
        :type  target:  int

        :return:  indices of the two numbers such that they add up to target
        :rtype:   List[int]

        """
        store = dict()
        for i in range(len(nums)):
            rest = target - nums[i]
            if rest not in store:
                store[nums[i]] = i
            else:
                return [store[rest], i]

    def minNumberOfSemesters(self, n: int, dependencies: List[List[int]], k: int) -> int:
        """ Parallel Courses II

        You are given an integer n, which indicates that there are n courses labeled from 1 to n. You are also given an array relations where relations[i] = [prevCoursei, nextCoursei], representing a prerequisite relationship between course prevCoursei and course nextCoursei: course prevCoursei has to be taken before course nextCoursei. Also, you are given the integer k.

        In one semester, you can take at most k courses as long as you have taken all the prerequisites in the previous semesters for the courses you are taking.

        Return the minimum number of semesters needed to take all courses. The testcases will be generated such that it is possible to take every course.

        :param n:  courses
        :type n:  int
        :param dependencies: prerequisite relationship between course prevCoursei and course nextCoursei has to be taken before course nextCoursi
        :type dependencies:  List[List[int]]
        :param k:  take at most k courses
        :type  k:  int

        :return:  minimum number of semesters needed to take all courses
        :rtype:  int

        """

        # Compute in-degree and adjacency graph for each node
        in_degree = [0] * n
        graph = collections.defaultdict(list)
        for prerequisite, course in dependencies:
            graph[prerequisite - 1].append(course - 1)
            in_degree[course - 1] += 1

        @lru_cache(None)
        def find_min_semester(courses: int) -> int:
            if not courses:
                return 0

            combinations = self.combinations(courses, k)

            min_semester = 1 << 32
            for k_courses in combinations:
                remaining_courses = courses - k_courses

                next_courses = 0

                # Decrease the in-degree
                for course_idx in range(n):
                    if (1 << course_idx) & k_courses:
                        for nei in graph[course_idx]:
                            in_degree[nei] -= 1
                            if in_degree[nei] == 0:
                                next_courses += 1 << nei

                min_semester = min(min_semester, find_min_semester(remaining_courses + next_courses))

                # Increase the in-degree (backtracking)
                for course_idx in range(n):
                    if (1 << course_idx) & k_courses:
                        for nei in graph[course_idx]:
                            in_degree[nei] += 1
            return min_semester + 1

        initial_courses = 0
        for i in range(n):
            if in_degree[i] == 0:
                initial_courses += 1 << i

        return find_min_semester(initial_courses)

    def count1(self, number: int) -> int:
        ones = 0
        tmp = number
        while tmp > 0:
            if tmp & 1 != 0:
                ones += 1
            tmp >>= 1
        return ones

    def combinations(self, number: int, k: int) -> List[int]:

        def helper(current_number: int, num_ones: int, remain_1s: int) -> List[int]:
            if remain_1s == 0:
                return [0]
            if num_ones <= remain_1s:
                return [current_number]
            else:
                # `(current_number - 1) ^ current_number` will give us all 1s starting from the smallest `1`
                last_one = ((current_number - 1) ^ current_number) & current_number

                # For choosing the last `1`
                p1 = helper(current_number - last_one, num_ones=num_ones - 1, remain_1s=remain_1s - 1)

                # For not choosing the last `1`
                p2 = helper(current_number - last_one, num_ones=num_ones - 1, remain_1s=remain_1s)

                return [p + last_one for p in p1] + p2

        return helper(current_number=number, num_ones=self.count1(number), remain_1s=k)

    def convertToTitle(self, columnNumber: int) -> str:
        """ Excel Sheet Column Title

        Given an integer columnNumber, return its corresponding column title as it appears in an Excel sheet.

        :param columnNumber:  Execel sheet
        :type  columnNumber:  int

        :return:  its corresponding column title
        :rtype:  int

        """

        ans = ""
        while columnNumber:
            columnNumber -= 1
            ans = chr(65 + columnNumber % 26) + ans
            columnNumber //= 26
        return ans

    def isMatch(self, s: str, p: str) -> bool:
        """ Regular Expression Matching

        Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

        '.' Matches any single character.
        '*' Matches zero or more of the preceding element.
        The matching should cover the entire input string (not partial).

        :param s: input string
        :type  s: string

        :param p: pattern character
        :type  p: string

        :return:  whehter the matching pattern cover the entire input string
        :rtype:  bool

        """

        m, n = len(s), len(p)

        # build our dp array depends on the current character in the pattern string
        dp = [[False] * (n+1) for _ in range(m+1)]

        dp[0][0] = True
        for i in range(0, m+1):
            for j in range(1, n+1):
                if p[j-1] == '*':
                    dp[i][j] = dp[i][j-2] or ( i > 0 and dp[i - 1][j] and ( s[i-1] == p[j-2] or p[j-2] == '.') )
                else:
                    dp[i][j] = i > 0 and dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.')

        return dp[m][n]

    def romanToInt(self, s: str) -> int:
        """ Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

        Symbol       Value
        I             1
        V             5
        X             10
        L             50
        C             100
        D             500
        M             1000

        For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simple X + II. The number 27 is written as XXVII, which is XX + V + II.

        Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not III. Instead, the number four is written as IV. Because the one is before the five we substract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where substraction used:
        
        - I can be placed before V (5) and X (10) to make 4 and 9.
        - X can be placed before L (50) and C (100) to make 40 and 90
        - C can be placed before D (500) and M (1000) to make 400 and 900

        Given a roman numeral, convert it to an integer.

        :param s:  Roman numeral
        :type  s:  string

        :return:  integer
        :rtype:   int

        """
        r = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        tot = 0
        for i in range(len(s)-1):
            # substraction rule
            if r[s[i]] < r[s[i+1]]:
                tot -= r[s[i]]
            else:
                tot += r[s[i]]

        tot += r[s[-1]]

        return tot

    def maxLength(self, arr: List[str]) -> int:
        """ Maximum Length of a Concatencated String with Unique Characters

        You are given an array of strings arr. A string s is formed by the concatenation of a subsequence of arr that has unique characters that has unique characters.

        A subsequence is an array that can be derived from another array of by deleting some or no elements without changing the order of the remaining elements.

        :param arr:  array of strings
        :type  arr:  List[str]

        :return:  the maximum possible length of subsequence
        :rtype:   int

        """
        self.maxlen = 0

        def dfs_maxlength(arr, cur, visited, idx):
            if idx == len(arr):
                return
            for i in range(idx, len(arr)):
                if i in visited:
                    continue

                ## not unique
                if len(cur) + len(arr[i]) != len(set(cur + arr[i])):
                    continue
                self.maxlen = max(self.maxlen, len(cur) + len(arr[i]))


                visited.add(i)
                dfs_maxlength(arr, cur + arr[i], visited, i + 1)
                visited.remove(i)

        dfs_maxlength(arr, '', set(), 0)

        return self.maxlen

    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """ Critical Connections in a Network

        :type  param:  Type

        :return:  Description
        :rtype:  Type

        """

        if not connections or n == 0:
            return []
        graph = defaultdict(list)
        for x, y in connections:
            graph[x].append(y)
            graph[y].append(x)

        visited = [False]*n
        parent = [-1]*n
        disc = [float("inf")]*n
        low = [float("inf")]*n
        time = [0]
        res = []

        def dfs_cirtical(node, time):
            visited[node] = True
            disc[node] = low[node] = time[0]
            time[0] += 1
            for neigh in graph[node]:
                if not visited[neigh]:
                    parent[neigh] = node
                    dfs_cirtical(neigh, time)
                    low[node] = min(low[node], low[neigh])
                    if low[neigh] > disc[node]:
                        res.append([node, neigh])
                elif neigh != parent[node]:
                    low[node] = min(low[node], disc[neigh])

        for node in range(n):
            if not visited[node]:
                dfs_cirtical(node, time)

        return res

    def arrayNesting(self, nums: List[int]) -> int:
        """ Array Nesting

        :param param:  Description
        :type  param:  Type

        :return:  Description
        :rtype:  Type

        :raise e:  Description
        """

        ans = cnt = 0
        for i, idx in enumerate(nums):
            if idx < 0:
                continue
            while nums[idx] >= 0:
                cnt, nums[idx], idx = cnt+1, -1, nums[idx]
            else:
                ans, cnt = max(ans, cnt), 0
        return ans

    def findPeakElement(self, nums: List[int]) -> int:
        """ Find Peak Element

        A peak element is an element that is strictly greater than its neighbors.

        Given an integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

        You may imagine that nums[-1] = nums[n] = -∞.

        You must write an algorithm that runs in O(log n) time.

        :param nums:  integer array
        :type  nums:  int

        :return:  peak element
        :rtype:  int

        """
        start = 0
        end = len(nums) - 1
        while start < end:
            mid = start + (end - start) // 2
            if nums[mid] > nums[mid + 1]:
                end = mid
            else:
                start = mid + 1
        return start

    def judgeCircle(self, moves: str) -> bool:
        """ Robot Return to Origin

        There is a robot starting at the position (0, 0), the origin, on a 2D plane. Given a sequence of its moves, judge if this robot ends up at (0, 0) after it completes its moves.

        You are given a string moves that represents the move sequence of the robot where moves[i] represents its ith move. Valid moves are 'R' (right), 'L' (left), 'U' (up), and 'D' (down).

        Return true if the robot returns to the origin after it finishes all of its moves, or false otherwise.

        Note: The way that the robot is "facing" is irrelevant. 'R' will always make the robot move to the right once, 'L' will always make it move left, etc. Also, assume that the magnitude of the robot's movement is the same for each move.

        :param moves:  the represents the move sequence of the robot 
        :type  moves:  str

        :return:  whether or not the robot returns to the origin after it finishes all of its moves
        :rtype:  bool

        """

        d = Counter(moves)
        return d['U'] == d['D'] and d['L'] == d['R']

    def longestStrChain(self, words: List[str]) -> int:
        """ Longest String Chain

        :param words:  an array of words where each word consists of lowercase English letters.
        :type  words:  List[str]

        :return:  the length of the longest possible word chain with words choosen from the given list of words
        :rtype:   int

        """

        if not words:
            return 0

        if len(words) == 1:
            return 1

        words = sorted(words, key=lambda elem: len(elem))

        ref = {word: 1 for word in words}

        for word in words:
            for index in range(len(word)):
                newWord = word[:index] + word[index+1:]
                if newWord in ref:
                    ref[word] = max(ref[word], ref[newWord] + 1)
            if word not in ref:
                ref[word] = 1

        ls = sorted(ref.items(), key=lambda elem: elem[1], reverse=True)

        return ls[0][1]
