"""
Use the Solution class to represent Leedcode problems
"""

from typing import List
from typing import Optional
import collections
from collections import defaultdict
from collections import Counter
from functools import lru_cache
import math


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

        store = {}

        for i in range(len(nums)):
            rest = target - nums[i]
            if rest not in store:
                store[nums[i]] = i
            else:
                return [store[rest], i]

    def minNumberOfSemesters(self, n: int, dependencies: List[List[int]], k: int) -> int:
        """ Parallel Courses II (HARD)

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
        graph = defaultdict(list)
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
        """ Regular Expression Matching (HARD)

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

        len_s, len_p = len(s), len(p)

        dp = [[False] * (len_p+1) for _ in range(len_s+1)]

        dp[0][0] = True

        for i in range(0, len_s+1):
            for j in range(1, len_p+1):

                if p[j-1] == '*':
                    # check the pattern repeats for 0 time or the pattern repeats for at least 1 time
                    dp[i][j] = dp[i][j-2] or (i > 0 and dp[i-1][j] and (s[i-1] == p[j-2] or p[j-2] == '.'))
                else:
                    # check the pattern is the same
                    dp[i][j] = i > 0 and dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.')

        return dp[len_s][len_p]

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

        total = 0
        len_num = len(s)

        for i in range(len_num-1):
            if r[s[i]] < r[s[i+1]]:
                total -= r[s[i]]
            else:
                total += r[s[i]]

        # The last string
        total += r[s[-1]]

        return total

    def maxLength(self, arr: List[str]) -> int:
        """ Maximum Length of a Concatencated String with Unique Characters

        You are given an array of strings arr. A string s is formed by the concatenation of a subsequence of arr that has unique characters that has unique characters.

        A subsequence is an array that can be derived from another array of by deleting some or no elements without changing the order of the remaining elements.

        :param arr:  array of strings
        :type  arr:  List[str]

        :return:  the maximum possible length of subsequence
        :rtype:   int

        """
        def dfs_maxlength(arr, cur, visited, idx):
            if idx == len(arr):
                return

            for i in range(idx, len(arr)):
                # repetition
                if i in visited:
                    continue

                # not unique
                if len(cur) + len(arr[i]) != len(set(cur + arr[i])):
                    continue

                self.maxlen = max(self.maxlen, len(cur) + len(arr[i]))

                visited.add(i)
                dfs_maxlength(arr, cur + arr[i], visited, i + 1)
                visited.remove(i) # backtracking

        self.maxlen = 0

        dfs_maxlength(arr, '', set(), 0)

        return self.maxlen

    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """ Critical Connections in a Network (HARD)

        There are n servers numbered from 0 to n - 1 connected by undirected server-to-server connections forming a network where connections[i] = [ai, bi] represents a connection between servers ai and bi. Any server can reach other servers directly or indirectly through the network.

        A critical connection is a connection that, if removed, will make some servers unable to reach some other server.

        Return all critical connections in the network in any order.

        :param connections:  Undirected server-to-server connections forming a network
        :type  connections:  List[List[int]]

        :return:  all critical connections in the network in any order
        :rtype:  List[List[int]]

        """

        ''' Time Limit Exceeded '''
        """
        critConns = []

        def buildGraph(exclude, conns):
            graph = defaultdict(list)
            for conn in conns:
                # critical connection (bridge)
                if conn == exclude:
                    continue
                graph[conn[0]].append(conn[1])
                graph[conn[1]].append(conn[0])
            return graph

        def dfs_traversal(graph, visited, curNode):
            visited.add(curNode)
            for neighbor in graph[curNode]:
                if neighbor in visited:
                    continue
                dfs_traversal(graph, visited, neighbor)

        for conn in connections:
            # build graph
            graph = buildGraph(conn, connections)
            visited = set()

            dfs_traversal(graph, visited, 0)

            # low link value
            print(len(visited))
            if n != len(visited):
                critConns.append(conn)

        return critConns

        """
        def dfs_visit(node, from_node=None):
            if node in low:
                return low[node]

            cur_id = low[node] = len(low)

            # Traversal
            for neigh in graph[node]:
                if neigh == from_node:
                    continue

                # Track the smallest low link value
                low[node] = min(low[node], dfs_visit(neigh, node))

            # Determine critical connection (bridge)
            # according to when the low link value is equal to visited time.
            if cur_id == low[node] and from_node is not None:
                results.append([from_node, node])

            return low[node]

        # build graph
        graph = defaultdict(set)
        for sor, dst in connections:
            graph[sor].add(dst)
            graph[dst].add(sor)

        low = {}
        results = []

        dfs_visit(0)

        return results

    def arrayNesting(self, nums: List[int]) -> int:
        """ Array Nesting

        You are given an integer array nums of length n where nums is a permutation of the numbers in the range [0, n - 1].

        You should build a set s[k] = {nums[k], nums[nums[k]], nums[nums[nums[k]]], ... } subjected to the following rule:

        The first element in s[k] starts with the selection of the element nums[k] of index = k.
The next element in s[k] should be nums[nums[k]], and then nums[nums[nums[k]]], and so on.
We stop adding right before a duplicate element occurs in s[k].

        Return the longest length of a set s[k].

        :param nums:  integer array nums of length n
        :type  nums:  List[int]

        :return:  the longest length of a set s[k]
        :rtype:  int
        """
        ans = 0

        for i in range(len(nums)):
            cnt = 0
            while nums[i] != -1:
                cnt += 1
                nums[i], i = -1, nums[i]
            ans = max(ans, cnt)

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

    def lengthOfLongestSubstring(self, s: str) -> int:
        """ Longest Substring Without Repeating Characters

        Given a string s, find the length of the longest substring without repeating characters.

        :param s:  string
        :type  s:  str

        :return:  the length of the longest substring without repeating characters
        :rtype:  int

        """

        maxCount = 0

        for i in range(len(s)):
            subStr = s[i]
            currentCount = 1
            for j in range(i+1, len(s)):
                if(s[j] in subStr):
                    break
                subStr += s[j]
                currentCount += 1

            maxCount = max(maxCount, currentCount)

        return maxCount

    def minimumCardPickup(self, cards: List[int]) -> int:
        """ 2260. Minimum Consecutive Cards to Pick Up

        You are give an integer array cards where cards[i] represents the value of the ith card. A pair of cards are matching if the cards have the same value.

        Return the minimum number of consecutive cards you have to pick up to have a pair of matching cards among the picked cards. If it is impossible to have matching cards, return -1.

        :param cards:  integer array cards
        :type  cards:  List[int]

        :return:  the minimum number of consecutive cards
        :rtype:  int

        """

        d = {}
        ans = math.inf

        for i in range(len(cards)):
            if cards[i] in d:
                # number of consecutive cards
                ans = min(i - d[cards[i]] + 1, ans)

            d[cards[i]] = i

        return ans if ans != math.inf else -1

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        """ Number of Provinces

        There are n cities. Some of them are connected, while some are not. If city a is connected directly with city b, and city b is connected directly with city c, then city a is connected indirectly with city c.

        A province is a group of directly or indirectly connected cities and no other cities outside of the group.

        You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

        Return the total number of provinces.

        :param isConnected:  a province is a group of directly or indirectly connected cities and no other cities outside of the group
        :type  isConnected:  List[List[int]]

        :return:  the total number of provinces
        :rtype:  int

        """

        rows = len(isConnected)
        seen = set()

        def dfs_connected(r):
            for idx, val in enumerate(isConnected[r]):
                if val == 1 and idx not in seen:
                    seen.add(idx)
                    dfs_connected(idx)

        count = 0

        for i in range(rows):
            if i not in seen:
                dfs_connected(i)
                count += 1

        return count

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """ Course Schedule

        There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

        For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

        Return true if you can finish all courses. Otherwise, return false.

        :param numCourses:  total of number courses you have to take
        :type  numCourses:  int

        :param prerequisites:  array prerequisites
        :type  prerequisites:  List[List[int]]


        :return:  whether or not you can finish all courses.
        :rtype:  bool

        """
        def dfs_course(course: int, graph: dict, marks: dict) -> bool:

            marks[course] = 1

            for neighbor in graph[course]:
                if marks[neighbor] == 0:
                    if not dfs_course(neighbor, graph, marks):
                        return False
                elif marks[neighbor] == 1:
                    return False

            marks[course] = 2
            return True

        graph = {n: [] for n in range(numCourses)}

        marks = {n: 0 for n in range(numCourses)}

        # create graph
        for course, prereq in prerequisites:
            graph[prereq].append(course)

        for course in range(numCourses):
            if marks[course] == 0:
                if not dfs_course(course, graph, marks):
                    return False

        return True

    def lengthOfLIS(self, nums: List[int]) -> int:
        """ Longest Increasing Subsequence

        Given an integer array nums, return the length of the longest strictly increasing subsequence.

        A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].

        :param nums:  array nums
        :type  nums:  List[int]

        :return:  the longest strictly increasing subsequence
        :rtype:  int

        """
        size = len(nums)

        len_LIS = [1 for _ in range(size)]

        # for each subsequence ending at index i
        for i in range(size):
            for k in range(i):
                if nums[k] < nums[i]:
                    len_LIS[i] = max(len_LIS[i], len_LIS[k] + 1)

        return max(len_LIS)

    def originalDigits(self, s: str) -> str:
        """ Reconstruct Original Digits from English

        Given a string s containing an out-of-order English representation of digits 0-9, return the digits in ascending order.

        :param s:  string containing an out-of-order English
        :type  s:  str

        :return:  the digits in ascending order
        :rtype:  str

        """
        c = dict()

        c[0] = s.count("z")
        c[2] = s.count("w")
        c[4] = s.count("u")
        c[6] = s.count("x")
        c[8] = s.count("g")

        c[3] = s.count("h") - c[8]
        c[5] = s.count("f") - c[4]
        c[7] = s.count("s") - c[6]

        c[9] = s.count("i") - (c[8] + c[5] + c[6])
        c[1] = s.count("o") - (c[0] + c[2] + c[4])

        c = sorted(c.items(), key=lambda x: x[0])

        ans = ""

        for k, v in c:
            ans += (str(k) * v)

        return ans

    def strongPasswordChecker(self, password: str) -> int:
        """ Strong Password Checker (HARD)

        A password is considered strong if the below conditions are all met:

        It has at least 6 characters and at most 20 characters.
        It contains at least one lowercase letter, at least one uppercase letter, and at least one digit.
        It does not contain three repeating characters in a row (i.e., "...aaa..." is weak, but "...aa...a..." is strong, assuming other conditions are met).
        Given a string password, return the minimum number of steps required to make password strong. if password is already strong, return 0.

        In one step, you can:

        Insert one character to password,
        Delete one character from password, or
        Replace one character of password with another character.


        :param password:  password
        :type  password:  str

        :return:  the minimum number of steps required to make password strong
        :rtype:  int

        """
        n = len(password)

        # character check (replace)
        containsUpper, containsLower, containsDigit = 0, 0, 0
        for c in password:
            if not containsUpper and c.isupper():
                containsUpper = 1
            if not containsLower and c.islower():
                containsLower = 1
            if not containsDigit and c.isdigit():
                containsDigit = 1

        c_swaps = (3 - (containsUpper + containsLower + containsDigit))

        # repeating check (replace)
        i, j, reps = 0, 1, list()
        while i < n:
            while j < n and password[i] == password[j]:
                j += 1
            reps.append(j-i)
            i, j = j, j+1

        # length (addition, subtraction)
        if n < 6:
            adds = 6 - n
            return max(adds, c_swaps)
        elif n <= 20:
            r_swaps = sum([elem // 3 for elem in reps])
            return max(c_swaps, r_swaps)
        else:
            subs = n - 20
            r = len(reps)
            for i in range(r):
                if subs >= 1 and reps[i] % 3 == 0:
                    reps[i] -= 1
                    subs -= 1
            for i in range(r):
                if subs >= 2 and reps[i] % 3 == 1 and reps[i] > 3:
                    reps[i] -= 2
                    subs -= 2
            for i in range(r):
                if subs > 0 and reps[i] > 2:
                    removed = min(subs, reps[i] - 2)
                    reps[i] -= removed
                    subs -= removed

            r_swaps = sum([elem // 3 for elem in reps])

            return max(c_swaps, r_swaps) + (n - 20)

    def maxProfit(self, k: int, prices: List[int]) -> int:
        """ Best Time to Buy and Sell Stock IV (HARD)

        You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.

        Find the maximum profit you can achieve. You may complete at most k transactions.

        Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

        :param prices:  the price of a given stock
        :type  prices:  List[int]

        :return:  the maximum profit you can achieve
        :rtype:  int

        """
        n = len(prices)

        if n == 0:
            return 0

        dp = [[0 for _ in range(n)] for _ in range(k+1)]

        # Update by each transction as well as each trading day
        for trans_k in range(1, k+1):
            # Balance before 1st transaction must be zero and Buy stock on first day means -prices[0]
            cur_balance_with_buy = 0 - prices[0]

            for day_d in range(1, n):
                # Either we have finished all k transactions before, or just sell out stock and finished k-th transaction today
                dp[trans_k][day_d] = max(dp[trans_k][day_d-1], cur_balance_with_buy + prices[day_d])

                # Either keep holding the stock we bought before, or just buy in today
                cur_balance_with_buy = max(cur_balance_with_buy, dp[trans_k-1][day_d-1] - prices[day_d] )

        return dp[k][n-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """ Minimum Path Sum

        Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

        Note: You can only move either down or right at any point in time.

        :param grid:  m x n grid filled with non-negative numbers
        :type  grid:  List[List[int]]

        :return:  minimizes the sum of all numbers along its path
        :rtype:  int

        """
        def dp(row, col):
            if row == 0 and col == 0:
                return grid[0][0]

            if row < 0 or col < 0:
                return float('inf')

            if (row, col) in pathsum:
                return pathsum[(row, col)]

            # Up or Left
            pathsum[(row, col)] = min(dp(row-1, col), dp(row, col-1)) + grid[row][col]

            return pathsum[(row, col)]

        pathsum = {}

        return dp(len(grid)-1, len(grid[0])-1)

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """  Find First and Last Position of Element in Sorted Array

        Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

        If target is not found in the array, return [-1, -1].

        You must write an algorithm with O(log n) runtime complexity.

        :param nums:  array of integers
        :type  nums:  List[int]

        :return:  the starting and ending position of a given target value
        :rtype:  List[int]

        """
        l = []
        r = nums[::-1]

        if target in nums:
            l.append(nums.index(target))
            a = r.index(target) + 1
            l.append(len(nums) - a)
        else:
            l.append(-1)
            l.append(-1)

        return l

    def maxSubArray(self, nums: List[int]) -> int:
        """docstring for maxSubArray"""
        """ Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

        A subarray is a contiguous part of an array.

        :param nums:  integer array
        :type  nums:  List[int]

        :return:  the contiguous subarray
        :rtype:  int

        """

        curr_max = nums[0]
        curr_sum = nums[0]

        for num in nums[1:]:
            curr_sum = max(num, curr_sum + num)
            curr_max = max(curr_sum, curr_max)

        return curr_max

    def simplifyPath(self, path: str) -> str:
        """ Simplify Path

        Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.

        In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'. For this problem, any other format of periods such as '...' are treated as file/directory names.

        The canonical path should have the following format:

        The path starts with a single slash '/'.
        Any two directories are separated by a single slash '/'.
        The path does not end with a trailing '/'.
        The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')
        Return the simplified canonical path.

        :param path:  an absolute path to a file or directory in a Uinx-style file system
        :type  path:  str

        :return:  canonical path
        :rtype:  str
        """
        path = path.split('/')

        stack = []

        for i in path:
            if i == '.' or i == '':
                continue
            elif i == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(i)

        return '/' + '/'.join(stack)

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ Subsets

        Given an integer array nums of unique elements, return all possible subsets (the power set).

        The solution set must not contain duplicate subsets. Return the solution in any order.

        :param param:  integer array nums of unique elements
        :type  param:  List[int]

        :return:  all possible subsets (the power set)
        :rtype:  List[List[int]]
        """
        results = []

        def recurse(start_index, current_subset):
            results.append(list(current_subset))

            for split_index in range(start_index, len(nums)):
                recurse(split_index + 1, current_subset + [nums[split_index]])

        recurse(0, [])

        return results

    def numDecodings(self, s: str) -> int:
        """ Decode Ways

        A message containing letters from A-Z can be encoded into numbers using the following mapping:

        To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

        "AAJF" with the grouping (1 1 10 6)
        "KJF" with the grouping (11 10 6)

        Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

        Given a string s containing only digits, return the number of ways to decode it.

        The test cases are generated so that the answer fits in a 32-bit integer.

        :param s:  string containing only digits
        :type  s:  str

        :return:  the number of ways to decode it
        :rtype:  int
        """
        if not s or s[0] == '0':
            return 0

        dp = [0 for x in range(len(s) + 1)]
        dp[0] = 1

        if s[0] == 0:
            dp[1] = 0
        else:
            dp[1] = 1

        for i in range(2, len(s) + 1):
            if 0 < int(s[i-1:i]) <= 9:
                dp[i] = dp[i - 1] + dp[i]
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] = dp[i - 2] + dp[i]

        return dp[-1]

    def getMinimumDays(self, parcels):
        cut = 0
        while parcels:
            minCapacity = min(parcels)
            for i in range(len(parcels)):
                parcels[i] = parcels[i] - minCapacity

            for i in range(len(parcels)):
                if 0 in parcels:
                    parcels.remove(0)

            cut += 1

        return cut

    def longestNiceSubstring(self, s: str) -> str:
        """ Longest Nice Substring

        A string s is nice if, for every letter of the alphabet that s contains, it appears both in uppercase and lowercase. For example, "abABB" is nice because 'A' and 'a' appear, and 'B' and 'b' appear. However, "abA" is not because 'b' appears, but 'B' does not.

        Given a string s, return the longest substring of s that is nice. If there are multiple, return the substring of the earliest occurrence. If there are none, return an empty string.

        :param s:  string
        :type  s:  str

        :return:  the longest substring
        :rtype:  str
        """
        def helper(start, end):
            chars = set(s[start: end])

            for i in range(start, end):
                if s[i].upper() in chars and s[i].lower() in chars:
                    continue
                s1 = helper(start, i)
                s2 = helper(i + 1, end)
                return s1 if len(s1) >= len(s2) else s2
            return s[start:end]

        return helper(0, len(s))


