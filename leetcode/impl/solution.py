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

    def insert(self, data):
        """docstring for insert"""
        if self.val:
            if self.next is None:
                self.next = ListNode(data)
            else:
                self.next.insert(data)
        else:
            self.val = data

    def PrintListNode(self):
        if self.next:
            self.next.PrintListNode()
        return self.val
#         print(self.val)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def insert(self, data):
        """docstring for insert"""
        if self.val:
            if data < self.val:
                if self.left is None:
                    self.left = TreeNode(data)
                else:
                    self.left.insert(data)
            elif data > self.val:
                if self.right is None:
                    self.right = TreeNode(data)
                else:
                    self.right.insert(data)
        else:
            self.val = data

    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print(self.val),
        if self.right:
            self.right.PrintTree()


class Solution(object):
    """
    A :class:`~leetcode.impl.solution` object is the leetcode quiz module

    This module is to implementate the leetcode problems
    """

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """ 1. Two Sum (Easy)

        Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
        You may assume that each input would have exactly one solution, and you may not use the same element twice.
        You can return the answer in any order.

        :param num:  array of integers
        :type  num:  List[int]
        :param target:  integer target
        :type  target:  inEasyt

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
        """ 1496. Parallel Courses II (HARD)

        You are given an integer n, which indicates that there are n courses labeled from 1 to n. You are also given an array relations where
        relations[i] = [prevCoursei, nextCoursei], representing a prerequisite relationship between course prevCoursei and course nextCoursei:
        course prevCoursei has to be taken before course nextCoursei. Also, you are given the integer k.


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
        """ 168. Excel Sheet Column Title (Medium)

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
        """ 10. Regular Expression Matching (HARD)

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
        """ 13. Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M. (Easy)

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
        """ 1239. Maximum Length of a Concatencated String with Unique Characters (Medium)

        You are given an array of strings arr. A string s is formed by the concatenation of a subsequence of arr that has unique characters that has unique characters.

        A subsequence is an array that can be derived from another array of by deleting some or no elements without changing the order of the remaining elements.

        :param arr:  array of strings
        :type  arr:  List[str]

        :return:  the maximum possible length of subsequence
        :rtype:   int
        """
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
        """ 
        self.ans, temp = 0, ""

        def dfs_length(arr, temp):

            self.ans = max(self.ans, len(temp))

            for i in range(len(arr)):
                if len(temp) + len(arr[i]) != len(set(temp + arr[i])):
                    continue

                dfs_length(arr[i+1:], temp+arr[i])

        dfs_length(arr, temp)

        return self.ans

    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """ 1192. Critical Connections in a Network (HARD)

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
            if node in low_link:
                return low_link[node]

            cur_id = low_link[node] = len(low_link)

            # Traversal
            for neigh in graph[node]:

                if neigh == from_node:
                    continue

                # Track the smallest low link value
                low_link[node] = min(low_link[node], dfs_visit(neigh, node))

            # Determine critical connection (bridge)
            # according to when the low link value is equal to visited time.
            if cur_id == low_link[node] and from_node is not None:
                results.append([from_node, node])

            return low_link[node]

        # build undirected graph
        graph = defaultdict(set)
        for sor, dst in connections:
            graph[sor].add(dst)
            graph[dst].add(sor)

        low_link, results = {}, []

        dfs_visit(0)

        return results

    def arrayNesting(self, nums: List[int]) -> int:
        """ 565. Array Nesting (Medium)

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
        """ 162. Find Peak Element (Medium)

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
        """ 657. Robot Return to Origin (Easy)

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
        """ 1048. Longest String Chain (Medium)

        You are given an array of words where each word consists of lowercase English letters.

        wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA without changing the order of the other characters to make 
        it equal to wordB.

        For example, "abc" is a predecessor of "abac", while "cba" is not a 
        predecessor of "bcad".

        A word chain is a sequence of words [word1, word2, ..., wordk] with k >= 1, where word1 is a predecessor of word2, word2 is a predecessor of word3, 
        and so on. A single word is trivially a word chain with k == 1.

        Return the length of the longest possible word chain with words chosen from the given list of words.

        :param words:  an array of words where each word consists of lowercase English letters.
        :type  words:  List[str]

        :return:  the length of the longest possible word chain with words choosen from the given list of words
        :rtype:   int
        """

        if not words:
            return 0

        if len(words) == 1:
            return 1

        # sorted array of words
        words = sorted(words, key=lambda elem: len(elem))

        ref = {word: 1 for word in words}

        for word in words:
            for index in range(len(word)):
                newWord = word[:index] + word[index+1:]

                if newWord in ref:
                    ref[word] = max(ref[word], ref[newWord] + 1)

        lengthwordchain = sorted(ref.items(), key=lambda elem: elem[1])

        return lengthwordchain[-1][1]

    def lengthOfLongestSubstring(self, s: str) -> int:
        """ 3. Longest Substring Without Repeating Characters (Medium)

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

            # Substring
            for j in range(i+1, len(s)):
                if(s[j] in subStr):
                    break
                subStr += s[j]
                currentCount += 1

#             print(subStr)
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
        matchedcard = {}
        ans = math.inf

        # Pair of matching cards among picked cards
        for i in range(len(cards)):
            if cards[i] in matchedcard:
                # number of consecutive cards
                ans = min(i - matchedcard[cards[i]] + 1, ans)

            matchedcard[cards[i]] = i

        return ans if ans != math.inf else -1

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        """ 547. Number of Provinces (Medium)

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
        group = set()

        def dfs_connected(node):
            for idx, val in enumerate(isConnected[node]):
                # a group of directly connected cities
                if val == 1 and idx not in group:
                    group.add(idx)
                    dfs_connected(idx)

        count = 0

        for node in range(rows):
            if node not in group:
                dfs_connected(node)
                count += 1

        return count

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """ 207. Course Schedule (Medium)

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

            # finish the prerequisites and course
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
        """ 300. Longest Increasing Subsequence (Medium)

        Given an integer array nums, return the length of the longest strictly increasing subsequence.

        A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].

        :param nums:  array nums
        :type  nums:  List[int]

        :return:  the longest strictly increasing subsequence
        :rtype:  int

        """
        size = len(nums)

        len_subseq = [1 for _ in range(size)]

        # for each subsequence ending at index i
        for i in range(size):
            for k in range(i):
                if nums[k] < nums[i]:
                    len_subseq[i] = max(len_subseq[i], len_subseq[k] + 1)

        return max(len_subseq)

    def originalDigits(self, s: str) -> str:
        """ 423. Reconstruct Original Digits from English (Medium)

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

        for idx, val in c:
            ans += (str(idx) * val)

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

    def maxProfitIV(self, k: int, prices: List[int]) -> int:
        """ 188. Best Time to Buy and Sell Stock IV (HARD)

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

        for trans_k in range(1, k+1):
            cur_balance_with_buy = 0 - prices[0]

            for day_d in range(1, n):
                dp[trans_k][day_d] = max(dp[trans_k][day_d-1], cur_balance_with_buy + prices[day_d])

                cur_balance_with_buy = max(cur_balance_with_buy, dp[trans_k-1][day_d-1] - prices[day_d] )

        return dp[k][n-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """ 64. Minimum Path Sum (Medium)

        Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

        Note: You can only move either down or right at any point in time.

        :param grid:  m x n grid filled with non-negative numbers
        :type  grid:  List[List[int]]

        :return:  minimizes the sum of all numbers along its path
        :rtype:  int

        """
        def dp_path(row, col):
            if row == 0 and col == 0:
                return grid[0][0]

            if row < 0 or col < 0:
                return math.inf

            # Top Left
            if (row, col) in pathsum:
                return pathsum[(row, col)]

            # Up or Left
            pathsum[(row, col)] = min(dp_path(row-1, col), dp_path(row, col-1)) + grid[row][col]

            return pathsum[(row, col)]

        pathsum = {}

        # begin from bottom right
        return dp_path(len(grid)-1, len(grid[0])-1)

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """ 34. Find First and Last Position of Element in Sorted Array (Medium)

        Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

        If target is not found in the array, return [-1, -1].

        You must write an algorithm with O(log n) runtime complexity.

        :param nums:  array of integers
        :type  nums:  List[int]

        :return:  the starting and ending position of a given target value
        :rtype:  List[int]

        """
        elem = []
        reverse = nums[::-1]

        if target in nums:
            # first position
            elem.append(nums.index(target))
            # last position
            a = reverse.index(target) + 1
            elem.append(len(nums) - a)
        else:
            elem.append(-1)
            elem.append(-1)

        return elem

    def maxSubArray(self, nums: List[int]) -> int:
        """ 53. Maximum Subarray docstring for maxSubArray (Easy)"""
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
        """ 71. Simplify Path (Medium)

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
        """ 78. Subsets (Medium)

        Given an integer array nums of unique elements, return all possible subsets (the power set).

        The solution set must not contain duplicate subsets. Return the solution in any order.

        :param param:  integer array nums of unique elements
        :type  param:  List[int]

        :return:  all possible subsets (the power set)
        :rtype:  List[List[int]]
        """

        """
        subset = []

        def recurse(start_index, current_subset):

            subset.append(list(current_subset))

            for split_index in range(start_index, len(nums)):
                recurse(split_index + 1, current_subset + [nums[split_index]])

        recurse(0, [])

        return subset

        """
        ans, temp = [], []

        def helper(nums, temp, ans):
            """docstring for helper"""
            for i in range(len(nums)):
                helper(nums[i+1:], temp + [nums[i]], ans)

            ans.append(temp)

        helper(nums, temp, ans)
        return ans

    def numDecodings(self, s: str) -> int:
        """ 91. Decode Ways (Medium)

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
        """ 1763. Longest Nice Substring (Easy)
        A string s is nice if, for every letter of the alphabet that s contains, it appears both in uppercase and lowercase. For example, "abABB" is nice because 'A' and 'a' appear, and 'B' and 'b' appear. However, "abA" is not because 'b' appears, but 'B' does not.

        Given a string s, return the longest substring of s that is nice. If there are multiple, return the substring of the earliest occurrence. If there are none, return an empty string.

        :param s:  string
        :type  s:  str

        :return:  the longest substring
        :rtype:   str
        """
        def dfs_substring(start: int, end: int) -> str:
            chars = set(s[start: end])

            for i in range(start, end):

                # Nice substring (conditions)
                if s[i].upper() in chars and s[i].lower() in chars:
                    continue

                s1 = dfs_substring(start, i) # left-side nice substring
                s2 = dfs_substring(i + 1, end) # right-side nice substring
                return s1 if len(s1) >= len(s2) else s2

            # longest nice substring
            return s[start:end]

        return dfs_substring(0, len(s))

    def containDuplicate(self, nums: List[int]) -> bool:
        """ 217. Contains Duplicate docstring for containDuplicate"""
        """ Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

        :param List[int]:  an integer array nums
        :type  List[int]:  List[int]

        :return:  Whether if any value appears at least twice in the array
        :rtype:   bool
        """
        """
        dict = []

        for i in range(0, len(nums)):
            if nums[i] in dict:
                return True
            else:
                dict.append(nums[i])
        return False
        """

        dict = {}

        for i in range(0, len(nums)):
            if nums[i] in dict:
                return True
            else:
                dict[nums[i]] = i

        return False

    def moveZeroes(self, nums: List[int]) -> List[int]:
        """ 283. Move Zeroes docstring for fname (Easy)"""
        """ Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

        Note that you must do this in-place without making a copy of the array.

        :param nums:  integer array nums
        :type  nums:  List[int]

        :return:  integer array nums that being move all o's to the end of it while maintaing the relative order of the non-zero elements
        :rtype:  List[int]
        """
        slow = fast = 0

        while fast < len(nums):
            if nums[fast] == 0:
                fast += 1
            else:
                # swap
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
                fast += 1

        return nums

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        """ 36. Valid Sudoku (Medium)

        Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

        Each row must contain the digits 1-9 without repetition.
        Each column must contain the digits 1-9 without repetition.
        Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
        Note:

        A Sudoku board (partially filled) could be valid but is not necessarily solvable.
        Only the filled cells need to be validated according to the mentioned rules.

        :param board:  Sudoku board
        :type  board:  List[List[str]]

        :return:  Determine if Sudoku board is valid
        :rtype:  bool
        """
        def checkbox(row, col):
            """docstring for checkbox"""
            hashmap = {}
            for i in range(row, row+3):
                for j in range(col, col+3):
                    if(board[i][j] != '.'):
                        if(board[i][j] in hashmap):
                            return False
                        else:
                            hashmap[board[i][j]] = 1
            return True

        def checkhorizontalline(row):
            """docstring for checkhorizontalline"""
            hashmap = {}
            for i in range(9):
                if board[row][i] == '.':
                    continue
                elif(board[row][i] != '' and board[row][i] in hashmap):
                    return False
                else:
                    hashmap[board[row][i]] = 1

            return True

        def checkverticalline(col):
            """docstring for checkverticalline"""
            hashmap = {}
            for i in range(9):
                if board[i][col] == '.':
                    continue
                elif(board[i][col] != '' and board[i][col] in hashmap):
                    return False
                else:
                    hashmap[board[i][col]] = 1

            return True

        for i in range(9):
            if (not checkverticalline(i) or not checkhorizontalline(i)):
                return False

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                if(not checkbox(i, j)):
                    return False

        return True

    def halvesAreAlike(self, s: str) -> bool:
        """ 1704. Determine if String Halves Are Alike docstring for halvesAreAlike (Easy)"""
        """ You are given a string s of even length. Split this string into two halves of equal lengths, and let a be the first half and b be the second half.

        Two strings are alike if they have the same number of vowels ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'). Notice that s contains uppercase and lowercase letters.

        Return true if a and b are alike. Otherwise, return false.

        :param s:  string
        :type  s:  str

        :return:  Determine if string halves are alike
        :rtype:  bool
        """
        def countVowels(s):
            vowel = set("aeiouAEIOU")
            return sum(1 for char in s if char in vowel)

        size = len(s)
        midpoint = size//2
        a, b = s[:midpoint], s[midpoint:]

        return countVowels(a) == countVowels(b)

    def maxProfitII(self, prices: List[int]) -> int:
        """ 122. Best Time to Buy and Sell Stock II docstring for maxProfit (Medium)"""
        """
        You are given an integer array prices where prices[i] is the price of a given stock on the ith day.

        On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.

        Find and return the maximum profit you can achieve.

        :param prices:  integer array prices
        :type  prices:  List[int]

        :return:  the maximum profit you can achieve
        :rtype:  int
        """
        total = 0
        start = prices[0]
        for price in prices[1:]:
            if price > start:
                total += price - start
            start = price
        return total

    def maxProfit(self, prices: List[int]) -> int:
        """ 121. Best Time to Buy and Sell Stock (Easy) """
        """ You are given an array prices where prices[i] is the price of a given stock on the ith day.

        You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

        Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

        :param prices:  integer array prices
        :type  prices:  List[int]

        :return:  maximum profit you can achieve from this transaction
        :rtype:  int
        """
        dp_hold, dp_not_hold = -math.inf, 0

        for stock_price in prices:

            # either keep in hold, or just buy today with stock price
            dp_hold = max(dp_hold, - stock_price)

            # either keep in not holding, or just sell today with stock price
            dp_not_hold = max(dp_not_hold, dp_hold + stock_price)

        # max profit must be in not-hold state
        return dp_not_hold

    def maxProfitwithfee(self, prices: List[int], fee: int) -> int:
        """ 714. Best Time to Buy and Sell Stock with Transaction Fee (Medium)"""
        """ You are given an array prices where prices[i] is the price of a given stock on the ith day, and an integer fee representing a transaction fee.

Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

        :param prices:  integer array prices
        :type  prices:  Lint[int]

        :param fee:  transaction fee
        :type  fee:  int

        :return:  the maximum profit you can achieve
        :rtype:  int
        """
        dp_hold, dp_sell = -math.inf, 0

        for stock_price in prices:
            dp_sell = max(dp_sell, dp_hold + stock_price)
            dp_hold = max(dp_hold, dp_sell - stock_price - fee)

        return dp_sell

    def minDeletionSize(self, strs: List[str]) -> int:
        """ 944. Delete Columns to Make Sorted """
        """ You are given an array of n strings strs, all of the same length.

            The strings can be arranged such that there is one on each line, making a grid. For example, strs = ["abc", "bce", "cae"] can be arranged as:

            abc
            bce
            cae

            You want to delete the columns that are not sorted lexicographically. In the above example (0-indexed), columns 0 ('a', 'b', 'c') and 2 ('c', 'e', 'e') are sorted while column 1 ('b', 'c', 'a') is not, so you would delete column 1.
            Return the number of columns that you will delete.

        :param strs:  array of n string
        :type  strs:  List[str]

        :return:  the number of columns that you will delete
        :rtype:  int
        """

        count = 0

        for i in range(len(strs[0])):
            temp = ""
            for j in range(len(strs)):
                temp += strs[j][i]
            if ''.join(sorted(temp)) != temp:
                count += 1

        return count

    def isMatch(self, s: str, p: str) -> bool:
        """ 44. Wildcard Matching (Medium) """
        """ Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where:
        
        '?' Matches any single character.
        '*' Matches any sequence of characters (including the empty sequence).
        The matching should cover the entire input string (not partial).

        :param s:  input string
        :type  s:  str

        :param p:  pattern
        :type  p:  str

        :return:  whether or not input staring and pattern are matched
        :rtype:  bool
        """

        dp = [[False] * (len(s)+1) for i in range(len(p)+1)]

        dp[0][0] = True

        # Matches any sequence of characters (including the empty sequence).
        for i in range(1, len(dp)):
            if p[i-1] == '*':
                dp[i][0] = dp[i-1][0]

        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):

                if(p[i-1] == s[j-1] or p[i-1] == "?"):
                    dp[i][j] = dp[i-1][j-1]

                if(p[i-1] == "*"):
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]

        return dp[-1][-1]

    def minimumLines(self, stockPrices: List[List[int]]) -> int:
        """ 2280. Minimum Lines to Represent a Line Chart (Medium) """
        """ You are given a 2D integer array stockPrices where stockPrices[i] = [dayi, pricei] indicates the price of the stock on day dayi is pricei. A line chart is created from the array by plotting the points on an XY plane with the X-axis representing the day and the Y-axis representing the price and connecting adjacent points. One such example is shown below:

        Return the minimum number of lines needed to represent the line chart.

        :param stockPrices:  2D integer array stockPrices
        :type  stockPrices:  List[List[int]]

        :return:  the minimum number of lines needed to represent the line chart
        :rtype:  int
        """

        if len(stockPrices) == 1:
            return 0
        if len(stockPrices) == 2:
            return 1
        n = len(stockPrices)
        lines = n-1

        stockPrices.sort()
        for i in range(1,n-1):
            a , b , c = stockPrices[i-1] , stockPrices[i] , stockPrices[i+1]
            if (b[0]-a[0])*(c[1] - b[1]) == (c[0]-b[0])*(b[1] - a[1]):
                lines -= 1

        return lines

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """ Next Greater Element I  """
        """ The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.

        You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.

        For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2. If there is no next greater element, then the answer for this query is -1.

        Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.

        :param nums1:  integer arrays
        :type  nums1:  List[int]

        :param nums2:  integer arrays
        :type  nums2:  List[int]

        :return:  an array is the next greater element
        :rtype:  Line[int]
        """
        ans = []
        for i in nums1:
            i_index = nums2.index(i)
            for j in range(i_index, len(nums2)):
                if nums2[j] > i:
                    ans.append(nums2[j])
                    break
            else:
                ans.append(-1)

        return ans

    def nextGreaterElementsII(self, nums: List[int]) -> List[int]:
        """ 503. Next Greater Element II (Medium)  """
        """ Given a circular integer array nums (i.e., the next element of nums[nums.length - 1] is nums[0]), return the next greater number for every element in nums.

        The next greater number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, return -1 for this number.

        :param nums:  circular integer array
        :type  nums:  List[int]

        :return:  the next greater number for every element in nums
        :rtype:  List[int]
        """
        s = []
        size = len(nums)
        res = [-1 for i in range(size)]

        for i in range(2 * size):
            i = i % size
            while len(s) != 0 and nums[s[-1]] < nums[i]:
                item = s.pop()
                res[item] = nums[i]
            s.append(i)
        return res

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """ 739. Daily Temperatures (Medium) """
        """ Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.

        :param temperatures:  an array of integers temperatures
        :type  temperatures:  List[int]

        :return: an array is the number of days you have to wait after the ith day to get warmer temperature.
        :rtype:  List[int]
        """
        stack = []
        res = [0] * len(temperatures)
        for i, temp in enumerate(temperatures):
            while stack and temp > stack[-1][0]:
                res[stack[-1][1]] = i - stack[-1][1]
                stack.pop()
            stack.append((temp, i))
        return res

    def totalStrength(self, strength: List[int]) -> int:
        """ 2281. Sum of Total Strength of Wizards (Hard) """
        """ As the ruler of a kingdom, you have an army of wizards at your command.

        You are given a 0-indexed integer array strength, where strength[i] denotes the strength of the ith wizard. For a contiguous group of wizards (i.e. the wizards' strengths form a subarray of strength), the total strength is defined as the product of the following two values:

        The strength of the weakest wizard in the group.
        The total of all the individual strengths of the wizards in the group.
        Return the sum of the total strengths of all contiguous groups of wizards. Since the answer may be very large, return it modulo 109 + 7.

        A subarray is a contiguous non-empty sequence of elements within an array.

        :param strength:  integer array strength
        :type  strength:  Lint[int]

        :return:  the sum of the total strengths of all contiguous groups of wizards
        :rtype:  int
        """
        res, ac, mod, stack, acc = 0, 0, 10 ** 9 + 7, [-1], [0]
        strength += [0]

        for r, a in enumerate(strength):
            ac += a
            acc.append(ac + acc[-1])
            while stack and strength[stack[-1]] > a:
                i = stack.pop()
                j = stack[-1]
                lacc = acc[i] - acc[max(j, 0)]
                racc = acc[r] - acc[i]
                ln, rn = i - j, r - i
                res += strength[i] * (racc * ln - lacc * rn) % mod
            stack.append(r)
        return res % mod

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """ 100. Same Tree (Easy) """
        """ Given the roots of two binary trees p and q, write a function to check if they are the same or not.

        Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

        :param p:  binary tree
        :type  p:  Optional[TreeNode]

        :param q:  binary tree
        :type  q:  Optional[TreeNode]

        :return:  Whether two binary trees are considered the same
        :rtype:  bool
        """
        def is_same(node1, node2):
            if not node1 and not node2:
                return True
            if not node1 or not node2:
                return False
            if node1.val != node2.val:
                return False
            return is_same(node1.left, node2.left) and is_same(node1.right, node2.right)

        return is_same(p, q)

    def minSwaps(self, nums: List[int]) -> int:
        """ 2134. Minimum Swaps to Group All 1's Together II """
        """ A swap is defined as taking two distinct positions in an array and swapping the values in them.

        A circular array is defined as an array where we consider the first element and the last element to be adjacent.

        Given a binary circular array nums, return the minimum number of swaps required to group all 1's present in the array together at any location.

        :param nums:  circular array
        :type  nums:  Lint[int]

        :return:  the minimum number of swaps required to group all 1's present in the array together at any location
        :rtype:  int
        """
        n, ones = len(nums), sum(nums)
        window = max_window = sum(nums[i] for i in range(ones))

        for i in range(n - 1):
            # shift window
            window += nums[(i + ones) % n] - nums[i]
            max_window = max(max_window, window)

        return ones - max_window

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """ 56. Merge Intervals (Medium) """
        """ Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input

        :param param:  array of intervals
        :type  param:  List[List[int]]

        :return:  an array of the non-overlapping intervals that cover all the intervals in the input
        :rtype:  Lint[Lint[int]]
        """
        if len(intervals) == 1:
            return intervals

        intervals = sorted(intervals, key=lambda x: x[0])

        ans = []
        cur = intervals[0]

        for i in range(1, len(intervals)):
            if cur[1] < intervals[i][0]:
                ans.append(cur)
                cur = intervals[i]
            elif cur[1] <= intervals[i][1]:
                cur[1] = intervals[i][1]

        ans.append(cur)

        return ans

    def searchInsert(self, nums: List[int], target: int) -> int:
        """ 35. Search Insert Position """
        """ Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

        You must write an algorithm with O(log n) runtime complexity.

        :param nums:  sorted array of distinct integers
        :type  nums:  List[int]

        :return:  the index where it would be if it were inserted in order
        :rtype:  int
        """
        start = 0
        end = len(nums) - 1

        while start <= end:
            middle = start + (end - start)//2

            if nums[middle] < target:
                start = middle + 1
            elif nums[middle] > target:
                end = middle - 1
            elif nums[middle] == target:
                return middle

        return start

    def findLucky(self, arr: List[int]) -> int:
        """ 1394. Find Lucky Integer in an Array """
        """ Given an array of integers arr, a lucky integer is an integer that has a frequency in the array equal to its value.

        Return the largest lucky integer in the array. If there is no lucky integer return -1.

        :param arr:  an array of integers
        :type  arr:  Lint[int]

        :return:  the largest lucky integer
        :rtype:  int
        """
        frequency = Counter(arr)
        lucky = []

        for n in frequency:
            if n == frequency[n]:
                lucky.append(frequency[n])

        return max(lucky) if len(lucky) > 0 else -1

    def isValid(self, s: str) -> bool:
        """ 20. Valid Parentheses """
        """ Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

        An input string is valid if:

        Open brackets must be closed by the same type of brackets.
        Open brackets must be closed in the correct order.

        :param s:  string containing just characters
        :type  s:  str

        :return:  determine if the input string is valid
        :rtype:  bool
        """
        stack = []
        for char in s:
            if len(stack):
                if (char == ")" and stack[-1] == "(") or (char == "]" and stack[-1] == "[") or (char == "}" and stack[-1] == "{"):
                    stack.pop()
                    continue

            stack.append(char)

        return len(stack) == 0

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """ 21. Merge Two Sorted Lists """
        """ You are given the heads of two sorted linked lists list1 and list2.

        Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.

        Return the head of the merged linked list.

        :param list1:  sorted linked list
        :type  list1:  Optional[ListNode]

        :param list2:  sorted linked list
        :type  list2:  Optional[ListNode]

        :return:  the head of the merged two lists
        :rtype:  Optional[ListNode]
        """
        if list1 and list2:
            a, b = list1, list2
            if b.val < a.val:
                a, b = b, a
            a.next = self.mergeTwoLists(a.next, b)
            return a

        return list1 or list2

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """ 19. Remove Nth Node From End of List """
        """ Given the head of a linked list, remove the nth node from the end of the list and return its head.

        :param param:  the head of a linked list
        :type  param:  Optional[ListNode]

        :return:  the head of a linked list that removed the nth node from the end of the list
        :rtype:  Optional[ListNode]
        """
        dummyNode = ListNode()
        dummyNode.next = head

        slow = dummyNode
        fast = dummyNode

        for i in range(n):
            fast = fast.next

        while fast.next is not None:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next

        return dummyNode.next

    def coinChange(self, coins: List[int], amount: int) -> int:
        """ 322. Coin Change """

        """ You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

        Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

        You may assume that you have an infinite number of each kind of coin.

        :param param:  an integer array coins
        :type  param:  List[int]

        :return:  the fewest number of coins that you need to make up that amount
        :rtype:  int
        """
        # minimum number of coins that makes up a
        dp = [math.inf]*(amount+1)

        dp[0] = 0
        for a in range(1, amount + 1):
            for coin in coins:
                if a - coin >= 0:
                    dp[a] = min(dp[a], 1 + dp[a - coin])

        return dp[-1] if dp[-1] != math.inf else -1

    def countBits(self, n: int) -> List[int]:
        """ 338. Counting Bits """
        """ Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

        :param n:  integer
        :type  n:  int

        :return:  an array of length n + 1 such that each i, ans[i] is the number of 1's in the binary representation of i
        :rtype:  List[int]
        """
        # Stack
        ans, prev = [0], 0

        for i in range(1, n+1):

            # multiple number of 1's in the binary representation of i
            if not i & (i-1):
                prev = i

            ans.append(ans[i - prev] + 1)

        return ans

    def getSum(self, a: int, b: int) -> int:
        """ 371. Sum of Two Integers """
        """ Given two integers a and b, return the sum of the two integers without using the operators + and -

        :param a:  integer
        :type  a:  int

        :param b:  integer
        :type  b:  int

        :return:  the sum of the two integers
        :rtype:  int
        """
        # use 32bit mask to limit int size to 32bit to prevent overflow
        mask = 0xffffffff

        while b & mask > 0:
            carry = (a & b) << 1
            a = a ^ b
            b = carry

        return (a & mask) if b > mask else a

    def numIslands(self, grid: List[List[str]]) -> int:
        """ 200. Number of Islands (Medium) """
        """ Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

        An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

        :param param:  m x n 2D binary grid
        :type  param:  List[List[str]]

        :return:  the number of islands
        :rtype:  int
        """
        if not grid:
            return 0

        m, n = len(grid), len(grid[0])
        ans = 0

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    # parent node
                    q = collections.deque([(i, j)])
                    grid[i][j] = '-1'

                    # BFS traversal
                    while q:
                        x, y = q.popleft()
                        for dx, dy in (0, 1), (0, -1), (1, 0), (-1, 0):
                            xx, yy = x+dx, y+dy

                            if 0 <= xx < m and 0 <= yy < n and grid[xx][yy] == '1':
                                q.append((xx, yy))
                                grid[xx][yy] = '-1'

                    ans += 1

        return ans

    def strStr(self, haystack: str, needle: str) -> int:
        """ 28. Implement strStr() """
        """ Implement strStr().

        Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

        :param haystack:  string haystack
        :type  haystack:  str

        :param needle:  string needle
        :type  needle:  str

        :return: the index of the first occurrence of needle in haystack
        :rtype:  int
        """
        for i in range(len(haystack)-len(needle)+1):
            if haystack[i:i+len(needle)] == needle:
                return i
        return -1

    def myAtoi(self, s: str) -> int:
        """ 8. String to Integer (atoi) """
        """ Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).

        :param param:  string
        :type  param:  str

        :return: 32-bit signed integer
        :rtype:  int
        """
        def str2num(string):
            num = 0
            for s in string:
                if not s.isdigit():
                    break
                num = 10 * num + int(s)

            return num

        if s == "":
            return 0

        if s[0] != "+" and s[0] != "-" and not s[0].isdigit():
            return 0
        else:
            if s[0] in ["+", "-"]:
                sign = s[0]
                res = str2num(s[1:])
                return min(res, 2**31 - 1) if sign == "+" else max(0-res, -2**31)
            else:
                return min(str2num(s), 2**31 - 1)

    def search(self, nums: List[int], target: int) -> int:
        """ 33. Search in Rotated Sorted Array """
        """ Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

        You must write an algorithm with O(log n) runtime complexity.

        :param param:  array
        :type  param:  List[int]

        :param target:  integer target
        :type  target:  int

        :return:  the index of target if it is in nums
        :rtype:  int
        """
        start, end = 0, len(nums)-1

        while(start <= end):
            mid = start + (end - start) // 2

            if(target == nums[mid]):
                return mid

            if(nums[mid] >= nums[start]):
                if(target < nums[mid] and target >= nums[start]):
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if(target <= nums[end] and target > nums[mid]):
                    start = mid + 1
                else:
                    end = mid - 1

        return -1

    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 46. Permutations (Medium) """
        """  Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

        :param nums:  array
        :type  nums:  List[int]

        :return:  all the possible permutations
        :rtype:  List[List[int]]
        """
        ans, temp = [], []

        def helper(nums, temp, ans):
            if len(nums) == 0:
                ans.append(temp)
                return

            for i in range(len(nums)):
                helper(nums[:i]+nums[i+1:], temp+[nums[i]], ans)

        helper(nums, temp, ans)
        return ans


    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 39. Combination Sum  (Medium) """
        """ Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

        The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

        It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

        :param candidates:  distinct integer candidates
        :type  candidates:  List[int]

        :return:  a list of all unique combinations of candidates where the chosed numbers sum to target
        :rtype:  List[List[int]]
        """
        ans, temp = [], []

        def backtracking(candidates, target, temp, ans):
            if target == 0:
                ans.append(temp)
                return

            for i in range(len(candidates)):
                if candidates[i] > target:
                    continue
                backtracking(candidates[i:], target-candidates[i], temp+[candidates[i]], ans)

        backtracking(candidates, target, temp, ans)
        return ans

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """ 40. Combination Sum II (Medium) """
        """ Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

        Each number in candidates may only be used once in the combination.

        :param param:  a collection of candidate numbers
        :type  param:  List[int]

        :param target:  a target number
        :type  target:  int

        :return:  all unique combinations in candidates where the candidates numbers sum to target
        :rtype:  List[List[int]]

        :raise e:  Description
        """
        res, temp = [], []

        candidates = sorted(candidates)

        def backtracking(candidates, target, temp, res):
            if target == 0:
                res.append(temp)
                return

            for i in range(len(candidates)):
                if candidates[i] > target:
                    continue

                # Each number in candidates may only be used once in the combination.
                if i >= 1 and candidates[i] == candidates[i-1]:
                    continue

                backtracking(candidates[i+1:], target-candidates[i], temp + [candidates[i]], res)

        backtracking(candidates, target, temp, res)
        return res

    def restoreIpAddresses(self, s: str) -> List[str]:
        """ 93. Restore IP Addresses (Medium) """
        """ A valid IP address consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (inclusive) and cannot have leading zeros.

        For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses, but "0.011.255.245", "192.168.1.312" and "192.168@1.1" are invalid IP addresses.

        Given a string s containing only digits, return all possible valid IP addresses that can be formed by inserting dots into s. You are not allowed to reorder or remove any digits in s. You may return the valid IP addresses in any order.

        :param s:  string s containing only digits
        :type  s:  str

        :return:  all possible valid IP addresses
        :rtype:  List[str]
        """
        ans, temp = [], ''
        k = 0

        def backtrack(s, k, temp, ans):
            if k == 4 and len(s) == 0:
                ans.append(temp[:-1])
                return
            if k == 4 or len(s) == 0:
                return

            for i in range(3):
                if k > 4 or i+1 > len(s):
                    break

                if int(s[:i+1]) > 255:
                    continue

                if i != 0 and s[0] == '0':
                    continue

                backtrack(s[i+1:], k+1, temp+s[:i+1]+'.', ans)

        backtrack(s, k, temp, ans)
        return ans

    def letterCombinations(self, digits: str) -> List[str]:
        """ 17. Letter Combinations of a Phone Number (Medium) """
        """ Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

        A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

        :param param:  a string containing digits from 2-9 inclusive
        :type  param:  str

        :return:  all possible letter combinations that the number could represent
        :rtype:  List[str]
        """
        if not digits:
            return []

        ans, temp = [], ''
        mapping = {'2': "abc", '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}

        def backtracking(mapping, digits, temp, ans):
            if len(digits) == 0:
                ans.append(temp)
                return

            for j in mapping[digits[0]]:
                backtracking(mapping, digits[1:], temp + j, ans)

        backtracking(mapping, digits, temp, ans)

        return ans

    def minOperations(self, nums: List[int]) -> int:
        """ 1827. Minimum Operations to Make the Array Increasing """
        """ You are given an integer array nums (0-indexed). In one operation, you can choose an element of the array and increment it by 1.

        For example, if nums = [1,2,3], you can choose to increment nums[1] to make nums = [1,3,3].
        Return the minimum number of operations needed to make nums strictly increasing.

        An array nums is strictly increasing if nums[i] < nums[i+1] for all 0 <= i < nums.length - 1. An array of length 1 is trivially strictly increasing.

        :param param:  integer array nums
        :type  param:  List[int]

        :return:  the minimum number of operations needed to make nums strictly increasing.
        :rtype:   int
        """
        if len(nums) == 1:
            return 0

        output = 0

        for i in range(1, len(nums)):
            if nums[i] <= nums[i-1]:
                output += nums[i-1] - nums[i] + 1
                nums[i] = nums[i-1] + 1

        return output

    def rob(self, nums: List[int]) -> int:
        """ 198. House Robber """
        """ You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

        Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

        :param nums:  [1,2,3,1]
        :type  nums:  4

        :return:  the maximum amount of money you can rob tonight without alerting the police
        :rtype:  int
        """
        """
        n = len(nums)
        if n > 2:
            nums[2] += nums[0]

        for i in range(3, n):
            nums[i] += max(nums[i-2], nums[i-3])
            print(i, nums[i])

        return max(nums[n-1], nums[n-2])
        """
        prev, cur = 0, 0
        for n in nums:
            cur, prev = max(prev + n, cur), cur

        return cur

    def reverseBits(self, n: int) -> int:
        """ 190. Reverse Bits """
        """ Reverse bits of a given 32 bits unsigned integer.

        :param n:  32 bits unsigned integer
        :type  n:  int

        :return:  Reverse bits
        :rtype:   int
        """
        res = 0

        for i in range(32):
            if n & 1:
                res += 1 << (31-i)
            n >>= 1

        return res

    def removeDuplicates(self, nums: List[int]) -> int:
        """ 26. Remove Duplicates from Sorted Array """
        """ Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.
        
        Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.
        
        Return k after placing the final result in the first k slots of nums.

        :param nums:  integer array nums sorted in non-decreasing order.
        :type  nums:  List[int]

        :return:  k after placing the final result in the first k slots of nums
        :rtype:  int
        """
        duplicates = 0

        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                duplicates += 1
            else:
                # replace the duplicate number
                nums[i - duplicates] = nums[i]

        return len(nums) - duplicates

    def rob(self, nums: List[int]) -> int:
        """ 198. House Robber """
        """ You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

        Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

        :param param:  integer array representing the amount of money of each house
        :type  param:  List[int]

        :return:  the maximize amount of money you can rob tonight without alerting the police
        :rtype:  int
        """
        rob1, rob2 = 0, 0

        for n in nums:
            temp = max(n + rob1, rob2)
            rob1 = rob2
            rob2 = temp

        return rob2

    def maxArea(self, height: List[int]) -> int:
        """ 11. Container With Most Water """
        """ You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

        Find two lines that together with the x-axis form a container, such that the container contains the most water.

        Return the maximum amount of water a container can store.

        :param height:  integer array height of length n
        :type  height:  List[int]

        :return:  the maximum amount of water a container can store
        :rtype:  int
        """
        start = 0
        end = len(height) - 1

        maxArea = min(height[start], height[end])*(end-start)

        area = 0

        # Binary search
        while start <= end:
            if height[start] < height[end]:
                start += 1
            else:
                end -= 1

            area = min(height[start], height[end])*(end-start)

            if area >= maxArea:
                maxArea = area

        return maxArea

    def myPow(self, x: float, n: int) -> float:
        """ 50. Pow(x, n)

        Implement pow(x, n), which calculates x raised to the power n (i.e., xn).

        :param x:  the number
        :type  x:  float

        :param n:  power
        :type  n:  int

        :return:  calculates x raised to the power n
        :rtype:  float
        """
        if n < 0:
            return self.myPow(1/x, -1*n)

        if n == 1:
            return x

        if n > 1 and n % 2 == 0:
            num = self.myPow(x, n/2)
            return num * num

        return x * self.myPow(x, n-1)

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """ 49. Group Anagrams """
        """ Given an array of strings strs, group the anagrams together. You can return the answer in any order.

        An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

        :param strs:  array of strings
        :type  strs:  List[str]

        :return:  the answer in any order
        :rtype:  List[List[str]]
        """
        hmap = collections.defaultdict(list)

        for i in range(len(strs)):
            ele = "".join(sorted(strs[i]))
            hmap[ele].append(strs[i])

        return [i for i in hmap.values()]

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """ 112. Path Sum """
        """ Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

        A leaf is a node with no children.

        :param root:  the root of a binary tree 
        :type  root:  Optional[TreeNode]

        :param targetSum:  target Sum
        :type  targetSum:  int

        :return:  if the tree has a root-to-leaf path that adding up all the values along the path equals targetSum
        :rtype:  bool
        """
        if not root:
            return False

        if not root.left and not root.right and root.val == targetSum:
            return True

        targetSum -= root.val

        return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)

    def minDepth(self, root: Optional[TreeNode]) -> int:
        """ 111. Minimum Depth of Binary Tree """
        """ Given a binary tree, find its minimum depth.

        The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

        :param root:  Binary Tree
        :type  root:  Optional[TreeNode]

        :return:  minimum depth
        :rtype:  int
        """
        if not root:
            return 0

        if not root.left and not root.right:
            return 1

        elif not root.right:
            return self.minDepth(root.left) + 1

        elif not root.left:
            return self.minDepth(root.right) + 1

        else:
            return min( map(self.minDepth, (root.left, root.right) ) ) + 1

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """ 139. Word Break """
        """ Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

        Note that the same word in the dictionary may be reused multiple times in the segmentation.

        :param s:  string
        :type  s:  str

        :param wordDict:  dictionary of strings
        :type  wordDict:  List[str]

        :return:  If s be segmented into a space-separated sequence of one or more dictionary words
        :rtype:  bool
        """

        dp = [True] + [False] * len(s)

        for indx in range(1, len(s)+1):
            for word in wordDict:
                if dp[indx - len(word)] and s[:indx].endswith(word):
                    dp[indx] = True

        return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """ 62. Unique Paths """
        """ There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

        Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

        :param m:  integer
        :type  m:  int

        :param n:  integer
        :type  n:  int

        :return:  the number of possible unique paths that the robot can take to reach the bottom-right corner
        :rtype:  int
        """
        hs = {} # hashtable

        def dp_path(m, n, hs):
            if m == 1 and n == 1:
                return 1

            if m == 0 or n == 0:
                return 0

            if (m, n) in hs:
                return hs[(m, n)]
            else:
                hs[(m, n)] = dp_path(m - 1, n, hs) + dp_path(m, n - 1, hs)
                return hs[(m, n)]

        return dp_path(m, n, hs)

    def findMin(self, nums: List[int]) -> int:
        """ 153. Find Minimum in Rotated Sorted Array """
        """ Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

        [4,5,6,7,0,1,2] if it was rotated 4 times.
        [0,1,2,4,5,6,7] if it was rotated 7 times.

        :param nums:  sorted rotated array of unique elements
        :type  nums:  List[int]

        :return:  the minimum element of this array
        :rtype:  int
        """
        return min(nums)

    def isSubsequence(self, s: str, t: str) -> bool:
        i = 0
        for c in t:
            if i < len(s) and s[i] == c:
                i += 1

        return i == len(s)

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """ 347. Top K Frequent Elements """
        """ Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order

        :param :  integer array
        :type  : int

        :return:  the k most frequent elements
        :rtype:  List[int]
        """
        myHash = Counter(nums)

        array = [[key, value] for key, value in myHash.items()]

        array.sort(key=lambda x: x[1], reverse=True)

        res = []

        for i in range(k):
            res.append(array[i][0])

        return res

    def firstUniqChar(self, s: str) -> int:
        """ 387. First Unique Character in a String """
        """ Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.

        :param s:  string
        :type  s:  str

        :return:  the first non-repeating character
        :rtype:  Type

        :raise e:  Description
        """
        count = {}
        for i in s:
            if i not in count:
                count[i] = 1
            else:
                count[i] += 1

        for i in s:
            if count[i] == 1:
                return s.index(i)

        return -1

    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 617. Merge Two Binary Trees """
        """ You are given two binary trees root1 and root2.

        Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree.

        Return the merged tree.

        :param root1:  binary tree
        :type  root1:  Optional[TreeNode]

        :param root2:  binary tree
        :type  root2:  Optional[TreeNode]

        :return:  the merged tree
        :rtype:  Optional[TreeNode]
        """
        if root1 and root2:
            root1.val += root2.val

            root1.left = self.mergeTrees(root1.left, root2.left)
            root1.right = self.mergeTrees(root1.right, root2.right)

        return root1 or root2
