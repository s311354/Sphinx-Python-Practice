"""
Use the Solution class to represent Leedcode problems
"""

from typing import List
import collections
from functools import lru_cache


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
            sec = target - nums[i]
            if sec not in store:
                store[nums[i]] = i
            else:
                return [store[sec], i]

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

    def minNumberOfSemesters(self, n: int, dependencies: List[List[int]], k: int) -> int:
        """ Parallel Courses II

        You are given an integer n, which indicates that there are n courses labeled from 1 to n. You are also given an array relations where relations[i] = [prevCoursei, nextCoursei], representing a prerequisite relationship between course prevCoursei and course nextCoursei: course prevCoursei has to be taken before course nextCoursei. Also, you are given the integer k.

        In one semester, you can take at most k courses as long as you have taken all the prerequisites in the previous semesters for the courses you are taking.

        Return the minimum number of semesters needed to take all courses. The testcases will be generated such that it is possible to take every course.

        :param n:  courses
        :type n:  int
        :param dependencies: prerequisite relationship between course prevCoursei and course nextCoursei has to be taken before course nextCoursei
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

    def dpma(self, i, j, memo, s, p):
        if i < 0 and j < 0: # When pattern matches with string
            return True
        if i < 0 or j < 0:  # When pattern or string didnt match
            if i < 0:
                while j >= 0: # Special case : Still some more pattern remain but all of them are .*
                    if p[j] != '*':
                        return False
                    j -=2
                return True
            return False
        if (i,j) in memo: # Caching results in memo
            return memo[i,j]
        a = b = c = False
        if p[j] == '*':
            if p[j-1] == s[i] or p[j-1] == '.':
                a = self.dpma(i-1, j, memo, s, p) # Match character
            b = self.dpma(i, j-2, memo, s, p) # Skip character
        else:
            if p[j] == s[i] or p[j] == '.':
                a = self.dpma(i-1, j-1, memo, s, p)
            memo[i,j] = a or b or c
        return a or b or c

    def isMatch(self, s: str, p: str) -> bool:
        memo = {}
        return self.dpma(len(s)-1, len(p)-1, memo, s, p)



    
