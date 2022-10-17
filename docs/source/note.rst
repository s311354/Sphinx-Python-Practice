Recap Python Basic Conception and Alogrithm
================================================

Welcome! This is the personal documentation for learning the basic conception of python.

Dataclasses - Data Classes
--------------------------

This module provides a decorator and functions for automatically adding generated special methods such as __init__() and __repr__() to user-defined classes.

The dataclass() decorator examines the class to find fields. A field is defined as class variable that has a type annotation.

The parameters to dataclass() are:

+ The __init_()  method is automatically added to the class: it is not directly specified in the InventoryItem definition.

+ The __repr__() method compute the "offical" string representation of an object. This is typeically used for debugging, so it is important that the representation is information-rich and unambiguous.

The example this code::

    from dataclasses import dataclass

    @dataclass
    class InventoryItem:
        """ Class for keeping track of an item in inventory. """
        name: str
        unit_price: float
        quantity_on_hand: int = 0

        def total_cost(self) -> float:
            return self.unit_price * self.quantity_on_hand

        def __init__(self, name: str, unit_price: float, quantity_on_hand: int=0):
            self.name = name
            self.unit_price = unit_price
            self.quantity_on_hand = quantity_on_hand

See the `Python dataclasses page <https://docs.python.org/3.7/library/dataclasses.html?highlight=class#module-dataclasses>`_ for more info.

Container Datatypes (list, tuple, set and dict)
-----------------------------------------------

This module implements specialized container datatypes providing alternatives to python's general purpose built-in container, **list**, **tuple**, **set** and **dict**.

Sequence Types - **list**, **tuple**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The common sequence operation are supported by most sequence types, both mutable and immutable, and shows in the following table:


.. list-table:: Common Sequence Operations
    :header-rows: 1
    :stub-columns: 0

    * - Operations
      - Result
    * - x in s
      - True if an item of s is equal to x, else False
    * - len(s)
      - length of s
    * - min(s)
      - smallest item of s
    * - max(s)
      - largest item of s
    * - s.index(x[, i[, j]])
      - index of the first occurence of x in s (at ot after index i and before index j)

The example using common operations::

    >>> stack = [i for i in range(5) ]
    >>> len(stack)
    5 
    >>> stack.index(1)
    1

See the `Python Common Sequence Operations page <https://docs.python.org/3.7/library/stdtypes.html#common-sequence-operations>`_ for more info.

Lists
'''''''''''''

Lists are *mutable sequences*, typeically used to store collections of *homogeneous items*.

class list([iterable]) may be constructed in several ways:

+ Using a pair of square brackets to denote the empty list: []
+ Using square brackets, separating items with commas: [a], [a, b, c]
+ Using a list comprehension: [x for x in iterable]
+ Using the type constructor: list() or list(iterable)

The example all return a list equal to [1, 2, 3]

    >>> a = [1, 2, 3]
    >>> b = list( (1, 2, 3) )

See the `Python lists page <https://docs.python.org/3.7/library/stdtypes.html#lists>`_ for more info.

Tuples
'''''''''''''

Tuples are *immutable sequences*, typeically used to store collections of *heretrogeneous data*. Tuples are also used for cases where ann immutable sequence of homogeneous data is needed. 

class tuple([iterable]) may be constructed in a number of ways:

+ Using a pair of parentheses to denote the empty tuple: ()
+ Using a trailing comma for a singleton tuple: a, or (a, )
+ Separating items with commas: a, b, c or (a, b, c)
+ Using the tuple() built-in: tuple() or tuple(iterable)

The example all return ('a', 'b', 'c')

    >>> a = tuple('abc')
    >>> b = tuple( ['a', 'b', 'c'] )

See the `Python tuples page <https://docs.python.org/3.7/library/stdtypes.html#tuples>`_ for more info.

Mapping Types - **dict**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*A mapping object maps hashable values to arbitrary objects*. A dictionary's keys are almost arbitrary values. Values that are not hashable, that is, values containing lists, dictionaries or other mutable types may not be used as keys.

class dict(** kwargs) dictionaries can be created by several mean:

+ Use a comma-separeted list or key: value pairs within braces: {'jack': 4098, 'sjoerd': 4227}
+ Use a dict comprehension: {}
+ Use the type constructor: dict()

The examples all return a dictionary equal to {"one": 1, "two": 2, "three": 3}

    >>> a = dict(one = 1, two = 2, three = 3)
    >>> b = {'one': 1, 'two': 2, 'three': 3}

There are the operations that dictionaries support:

+ items(): Return the value for key if key is in the dictionary, else default.

+ setdefault(key[, default]): If key is in the dictionary, return its value. If not, insert key with a value of default and return default.

+ values(): Return a new view of the dictionary's values.

See the `Python dict page <https://docs.python.org/3.7/library/stdtypes.html#mapping-types-dict>`_ for more info.

Set Types - **set**, frozenset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A set object is an *unordered collection of distinct hashable objects*. Common uses include membership testing, removing duplicates from a sequence, and computing mathematical operations. such as intersection, union, difference, and symmetric difference.

class set([iterable]) can be created by several means:

+ Use a comma-separated list of elements within braces: {'jack', 'sjoerd'}
+ Use the type constructor: set(), set('foobar')

The example all return a set equal to {'a', 'b'}::

    >>> a = {'a', 'b'}
    >>> b = set('a', 'b')

See the `Python set, forzenset page <https://docs.python.org/3.7/library/stdtypes.html#set-types-set-frozenset>`_ for more info.


Collections — Container datatypes (defaultdict, Container)
----------------------------------------------------------

This module implements specialized container datatypes providing alternatives to Python's general purpose built-in containers, dict, list, set, and tuple.

defaultdict objects
^^^^^^^^^^^^^^^^^^^^

dict subclass that calls a factory function to supplty missing values

Return a *new dictionary-like object*. defaultdict is a subclass of the built-in dict class. It overrides one method and adds one writable instance variable. The remaining functionality is the same as for the dict class and is not documented here.

When each key is encountered for the first time, it is not already in the mapping; so an entry is automatically created using the default_factory function which returns an empty list. The list.append() operation then attaches the value to the new list. When keys are encountered again, the look-up proceeds normally (returning the list for that key) and the list.append() operation adds another value to the list.

The example using list as the default_factory::

    >>> from collections import defaultdict
    >>> s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
    >>> d = defaultdict(list)
    >>> for k, v in s:
    ...     d[k].append(v)
    ...
    >>> sorted(d.items())
    [('blue', [2, 4]), ('red', [1]), ('yellow', [1, 3])]

    >>> d = defaultdict(list)
    >>> for k, v in s:
    ...     d[k] = v
    ...
    >>> sorted(d.items())
    [('blue', 4), ('red', 1), ('yellow', 3)]

Note:

+ Using a defaultdict to handle missing keys can be faster than using dict.setdefault().

See the `Python defaultdict object page <https://docs.python.org/3.7/library/collections.html?highlight=collections%20defaultdict#defaultdict-objects>`_ for more info.

Counter objects
^^^^^^^^^^^^^^^^^^^^

dict subclass for counting hashable objects

A Counter is a dict subclass for counting hashable objects. It is a collection where elements are stored as dictionary keys and their counts are stored as dictionary values.

The example using Counter objects that have a dictionary::

    >>> from collections import Counter
    >>> Counter(['egg', 'ham'])
    Counter({'egg': 1, 'ham': 1})

See the `Python Counter object page <https://docs.python.org/3.7/library/collections.html?highlight=collections%20defaultdict#counter-objects>`_ for more info.

Built-in Types (str, range)
--------------------------------

The principle built-in types are numerics, sequences, mapping, class, instance and exceptions.

Text Sequence Type - (str)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Textual data in Python is handled with str objects, or strings.

str(object='') return a string version of object. If object is not provided, returns the empty string. Otherwise, the behavior of str() depends on whether encoding or errors is given.

The standard library covers a number of other modules that provide various text related utilities:

+ str.upper(): return a copy of the string with all the cased characters coverted to uppercase.

+ str.lower(): return a copy of the string with all the cased characters coverted to lowercase.

+ str.find(sub[, start[, end]]): return the lowest index in the string where substring sub is found within the slice s[start:end].

+ str.split(sep=None, maxsplit=-1): return a list of the words in the string, using sep as the delimiter string.

+ str.endswith(suffix[, start[, end]]): return True if the string ends with the specified suffix, otherwise return False. suffix can be a tuple of suffixs to look for. With optional start, test begining at that position. With optional end, stop comparing at that position.

+ str.strip([chars]): return a copy of the string with the leading and trailing characters removed::

    >>> '   spacious   '.strip()
    spacious

+ str.index(sub[, start[, end]]): like find(), but raise ValueError when the substring is not found.

+ str.isalnum(): return True if all characters in the string are alphanumeric and there is at least one character, False otherwise.

+ str.isalpha(): return True if all characters in the string are alphabetic and there is at least one character, False otherwise.

+ str.isdigit(): return True if all characters in the string are digits and there is at least one characters, False otherwise.

See the `Python str page <https://docs.python.org/3.7/library/stdtypes.html?highlight=strip#str>`_ for more info.

Ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The range type represents an immutable sequence of numbers and is commonly used for looping a specific number of times in for loops.

class range(stop), class range(start, stop[, step]) may be constructed in integers.

Range example::

    >>> list(range(10))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> list(range(1, 11))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

The example using range to initialize a list of lists::

    >>> row = 3
    >>> matrix = [[] for _ in range(row)]
    >>> matrix[0].append(1)
    >>> matrix[0].append(2)
    >>> matrix[1].append(3)
    >>> matrix[1].append(4)
    >>> matrix
    [[1, 2], [3, 4]]

See the `Python ranges page <https://docs.python.org/3.7/library/stdtypes.html?highlight=range#ranges>`_ for more info.

Built-in Functions (enumerate, map)
----------------------------------------------------------

The Python interpreter has a number of functions and types built into it that are always available.

Enumerate
^^^^^^^^^^^^^^^^^^^^

enumerate(iterable, start=0) returns an enumerate object.

The example using enumerate objects that have a List::

    >>> list(enumerate([1, 1, 0]))
    [(0, 1), (1, 1), (2, 0)]

See the `Python enumerate page <https://docs.python.org/3.7/library/functions.html?highlight=enumerate#enumerate>`_ for more info.

Map
^^^^^^

map(function, iterable, ...) return an iterator that applies function to every item of iterable, yielding the result. If additional iterable arguments are passed, function must take that many arguments and is applied to the items from all iterables in parallel, With multiple iterables, the iterator stops when the shortest iterable is exhausted.

See the `Python map page <https://docs.python.org/3.7/library/functions.html#map>`_ for more info.

Heap queue algorithm (heapq)
------------------------------------

Heapq
^^^^^^

This module provides an implementation of the heap queue algorithm, also known as the **priority queue** algorithm.

This implementation uses arrays for which heap[k] <= heap[2*k + 1] and heap[k] <= heap[2*k+2] for all k, counting elements from zero. These make it possible to view that heap[0] is the smallest item, and heap.sort() maintains the heap invariant!

Heaps are binary trees for which every parent node has a value less than or equal to any of its children.

The example of using heapq::

    >>> heapq.heappush(heap, (3, 1, 2, 0))
    >>> heapq.heappush(heap, (1, 1, 0, 0))
    >>> heap
    [(1, 1, 0, 0), (3, 1, 2, 0)]
    [(1, 1, 0, 0), (3, 1, 2, 0)]

The API below differs from textbook heap algorithm in two aspects: 

+ use zero-based indexing.
+ pop method returns the smallest item, not the largest.

The following functions are provided:

+ heapq.heappush(heap, item): push the value item onto the heap, maintaining the heap invariant.
+ heapq.heappop(heap): pop, and return the smallest item from the heap, maintaining the heap invariant. If the heap is empty, IndexError is raised.

See the `Python heapq page <https://docs.python.org/3.7/library/heapq.html?highlight=heappush#module-heapq>`_ for more info.

Supporting for type hits (TypeVar, List, Optional)
----------------------------------------------------------

TypeVar
^^^^^^^^

Type variable

the example using typevar::

    >>> from typing import TypeVar
    >>> T = TypeVar('T', int, float)
    >>> def vec2(x: T, y: T) -> List[T]: return [x, y]
    >>> vec2(1, 2.2)
    [1, 2.2]

See the `Python TypeVar page <https://docs.python.org/3.7/library/typing.html?highlight=optional#typing.TypeVar>`_ for more info.

List
^^^^^^

Generic version of list. Useful for annotating return types. To annotate argument it is preferred to use an abstract collection type such as Sequence of Iterable.

The example using List::

    >>> from typing import List
    >>> Vector = List[float]
    >>> def scale(scalar: float, vector: Vector) -> Vector: return [scalar * num for num in vector]
    >>> scale(2.0, [1.0, -4.2, 5.4])
    [2.0, -8.4, 10.8]

See the `Python List page <https://docs.python.org/3.7/library/typing.html?highlight=optional#typing.List>`_ for more info.


Optional
^^^^^^^^^

Optional type is equivalent to Union[X, None]

The example using Optional::

    >>> from typing import Optional
    >>> def test(a: Optional[dict] = None) -> None: print(a)
    >>> test({'a': 1234})
    {'a': 1234}

See the `Python Optional page <https://docs.python.org/3.7/library/typing.html?highlight=optional#typing.Optional>`_ for more info.

Supporting for enumerations
------------------------------

An enumeration is a set of symbolic names (members) bound to unique, constant values. Within an enumeration, the members can be compared by identity, and the enumeration itself can be iterated over.


The example using to create an enum::

    >>> from enum import Enum
    >>> class Color(Enum):
    ...     RED = 1
    ...     GREEN = 2
    ...     BLUE = 3
    ...

See the `Python Enum page <https://docs.python.org/3.7/library/enum.html?highlight=enum#module-enum>`_ for more info.

Expressions (lambda)
----------------------------------------------------------

Lambdas
^^^^^^^^^^^^^^^^^^^^

lambda expressions (sometimes called called lambda forms) are used to created anonymous functions.

The expression ``` lambda parameters: expression ``` yield a function object.

The example using lambda expressions::

    >>> words = ['ea', 'bcd', 'ay']
    >>> sorted(words, key=lambda elem: len(elem))
    ['ea', 'ay', 'bcd']

    >>> nums = [1, 3, 6, 7]
    >>> list(map(lambda n: n ** 2, nums))
    [1, 9, 36, 49]

See the `Python Lambdas page <https://docs.python.org/3.7/reference/expressions.html?highlight=lambda#lambda>`_ for more info.

Functools (lru_cache)
------------------------

The functools module is for higher-order functions: functions that act on or return other functions. In general, any callable object can be treated as a function for the purposes of this module.

@functools.lru_cache
^^^^^^^^^^^^^^^^^^^^^^^^

Decorator to wrap a function with a memorizing callable that saves up to the maxsize most recent calls. It can save time when an expensive or I/O bound function is periodcally called with the same argument.

An LRU (least recently used) cache works best when the most recent calls are the best predictors of upcoming calls (for example, the most popular articles on a news server tend to change each day). The cache’s size limit assures that the cache does not grow without bound on long-running processes such as web servers.

In general, the LRU cache should only be used when you want to reuse previously computed values. Accordingly, it doesn’t make sense to cache functions with side-effects, functions that need to create distinct mutable objects on each call, or impure functions such as time() or random().

The example using lru_cache to computing Fibonacci numbers to implement a dynamic programming::

    >>> @lru_cache(maxsize=None)
    >>> def fib(n):
    >>>     if n < 2:
    >>>         return n
    >>>     return fib(n-1) + fib(n-2)

    >>> [fib(n) for n in range(16)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


os - Miscellaneous operating system interfaces
------------------------------------------------

This module provides a portable way of using operating system dependent functionality.

Notes on the availability of these functions:

+ The design of all built-in operation system dependent modules of Python is such that as long as the same functionality is available, it uses the same interface; for example, the function os.stat(path) returns stat information about path in the same format (which happens to have originated with the POSIX interface).

+ Extensions peculiar to a particular operating system are also available through the os module, but using them is of course a threat to portability.

+ All functions accepting path or file names accept both bytes and string objects, and result in an object of the same type, if a path or file name is returned.

Process Parameters
^^^^^^^^^^^^^^^^^^^

These functions and data items provide information and operate on the current process and user.

+ os.getenv(key, default= None): Return the value of the environment varialbe key if it exists, or default if it doesn't. key, default and the result are str.

+ os.getenvb(key, default=None): Retrun the value of the environment varialbe key if it exists, or default if it doesn't. key, default and the result are bytes.

+ os.chdir(path)

+ os.uname(): Retruns information identifying the current operating system. The return value iscontent an object with five attributes (sysname, nodename, release, version, machine)

+ os.unsetenv(key): Unset (delete) the environment variable named key. Such changes to the environment affect subprocesses started with os.system(), popen() or fork() and execv().

See the `Python os Process Parameters page <https://docs.python.org/3.7/library/os.html#process-parameters>`_ for more info.

File Descriptor Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These functions operate on I/O streams referenced using file descriptors.

+ os.close(fd): Close file descriptor fd

+ os.write(fd, str): Write the bytestring in str to file descriptor fd. Return the number of bytes actually written.

+ ...

See the `Python os File Descriptor Operations page <https://docs.python.org/3.7/library/os.html#file-descriptor-operations>`_ for more info.

Files and Directories
^^^^^^^^^^^^^^^^^^^^^^

On some Unix platforms, many of these functions support one or more of these feature.

+ os.access(path, mode, \*, dir_fd=None, effective_ids=False, follow_symlinks=True): Use the real uid/gid to test for access to path. Note that most operations will use the effective uid/gid, therefore this routine can be used in a suid/sgid environment to test if the invoking user has the specified access to path.

+ os.getcwd(): Return a string representing the current working directory

+ os.mkdir(path, mode=0o777, \*, dir_fd=None): Create a directory named path with numeric mode mode

+ os.walk(top, topdown=True, onerror=None, followlinks=False): Generate the file names in a directory tree by walking the tree either top-down or bottom-up. For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames). dirpath is a string, the path to the directory. dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). filenames is a list of the names of the non-directory files in dirpath. Note that the names in the lists contain no path components.

The eaxmple using os.walk to display the sum of bytes taken by non-directory files in each directory under the starting directory::

    >>> import os
    >>> from os.path import join, getsize
    >>> for dirpath, dirnames, filenames in os.walk('source'):
    >>>     print(dirpath, "consumes", end=" ")
    >>>     print(sum(getsize(join(dirpath, name)) for name in filenames), end=" ")
    >>>     print("bytes in", len(files), "non-directory files")
    source consumes 73 bytes in 1 non-directory files
    source/_posts consumes 26775 bytes in 6 non-directory files
    ...

+ ...

See the `Python os Files and Directories page <https://docs.python.org/3.7/library/os.html#files-and-directories>`_ for more info.

Porcess Management
^^^^^^^^^^^^^^^^^^^

These functions may be used to create and manage processes.

+ os.popen(cmd, mode='r', buffering=-1): Open a pipe to or from command cmd, The return value is an open file object connected to the pipe, which can be read or written depending on whether mode is 'r' (default) or 'w'. The buffering argument has the same meaning as the corresponding argument to the built-in open() function. The returned file object reads or writes text strings rather than bytes.

+ os.wait(): Wait for completion of a child process, and return a tuple containing its pid and exit status indication: a 16-bit number, whose low bytes is the singal number that killed the process, and whose high byte is the exit status (if the signal number is zero); the high bit of the low byte is set if a core file was produced.

+ ...

See the `Python os Process Management page <https://docs.python.org/3.7/library/os.html#process-management>`_ for more info.

Inheritance of File Descriptors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ os.stat(path, \*, dir_fd=None, follow_syslinks=True): Get the status of a file or a file descriptor. Perform the equivalent of stat() system call on the given path. path may be specified as either a string or bytes - directly or indirectly through the PathLike interface - or as an open file descriptor. Return a stat_result object.

The example using os.stat to display the size of the file in bytes, if it is a regular file or a symbolic link::

    >>> import os
    >>> statinfo = os.stat('Gemfile')
    >>> statinfo.st_size
    60

+ ...

See the `Python os Inheritance of File Descriptors page <https://docs.python.org/3.7/library/os.html#inheritance-of-file-descriptors>`_ for more info.

Miscellaneous System Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ os.cpu_count(): Return the number of CPUs in the system. Return None if undetermined.

+ os.sep: The character used by the operating system to separate pathname components. This is '/' for POSIX and '\\' for Windows. Note the knowing this is not sufficient to be able to parse or concatenate pathnames pathnames - use os.path.split() and os.path.join() - but it is occasionally useful.

The example using os.sep to travel only the first level of directory::

    In [1]: import os
    In [2]: for dirpath, dirnames, filenames in os.walk('.') :
       ...:     for name in dirnames:
       ...:         depth = os.path.relpath(dirpath, name) .count (os.sep)
       ...:         if depth == 0:
       ...:             print(name)

    Pictures
    _pycache
    twse
    duu
    ...


The other example using os.sep to travel only the second level of files::

    In [1]: import os
    In [2]: for dirpath, dirnames, filenames in os.walk('.') :
       ...:     for name in filenames:
       ...:         depth = os.path.relpath(dirpath, name).count(os.sep)
       ...:         if depth == 1:
       ...:             print(name)

    index.md
    index.md
    ...

+ ...

See the `Python os Miscellaneous System Information page <https://docs.python.org/3.7/library/os.html#miscellaneous-system-information>`_ for more info.

Common pathname manipulation (os.path)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This module implements some useful functions on pathnames.

There are several versions of this module in the standard library:

+ os.path.join(path, paths): join one or more path components intelligently. The return value is the concatenation of path and any members of paths with exactly one directory separator.

+ os.path.getsize(path): Return the size, in bytes, of path.

+ os.path.normpath(path): Normalize a pathname by collapsing redundant separators and up-level reference so that A//B, A/B/, A/./B and A/foo/../B all become A/B. This string manipulation may change the meaning of a path that cotains symbolic links.

+ os.path.realpath(path, \*, strict=False): Return the canonical path of the specified filename, eliminating any symbolic links encountered in the path (if they are supported by the operating system)

+ os.path.relpath(path, start=os.curdir): Return a relative filepath to path either from the current directory or from an optional start directory. This is a path computation: the filesystem is not accessed to confirm the existence or nature of path or start.

+ os.path.split(path): Split the pathname path into a pair, (head, tail) where tail is the last pathname component and head is everything leading up to that. The tail part will never contain a slash, if path ends in a slash, tail will be empty. If there is no slash in path, head will be empty. If path is empty, both head and tail are empty. Trailing slashes are stripped from head unless it is the root (one or more slashes only). In all cases, join(head, tail) returns a path to the same location as path (but the strings may differ).

The example using os.path.getsize to get the size, in bytes, of files in current directory::

    In [1]: import os
    In [2]: from os.path import getsize, join
    In [3]: for dirpath, dirnames, filenames in os.walk('.') :
       ...:     for name in filenames:
       ...:         print(join(dirpath, name), " consumes", end=" ")
       ...:         print(getsize(join(dirpath, name)))
       ...:         print(" bytes")))

    ./duu/README.md consumes 2244 bytes
    ...

+ os.path.isdir(path): Return True if path is an existing directory.

+ os.path.splitext(path): Split the pathname path into a pair (root, ext) such that root + ext == path, and extension, ext, is empty or begins with a period and contains at most one period.

The example using os.path.splitext() to split the pathname path::

    In [1]: from os.path import splitext
    In [2]: splitext('404.md')
    Out [2]: splitext('404', '.md')


+ ...

See the `Python os.path page <https://docs.python.org/3.7/library/os.path.html>`_ for more info.

Concurrent.futures - Launching parallel tasks
------------------------------------------------

The concurrent.futures module provides a high-level interface for asynchronously executing callables. 

The asynchronous execution can be performed with threads, using ThreadPoolExecutor, or separate processes, using ProcessPoolExecutor. Both implement the same interface, which is defined by the abstract Executor class.

ThreadPoolExecutor
^^^^^^^^^^^^^^^^^^^

ThreadPoolExecutor is an Executor subclass that uses a pool of threads to execute calls asynchronously.

class concurrent.futures.ThreadPoolExecutor(max_workers=None, thread_name_prefix='', initializer=None, initargs())

    An Executor subclass that uses a pool of at most max_workers threads to execute calls asynchronously.

The example using ThreadPoolExecutor to ensure threads are operating each future with its URL::

    >>> import concurrent.futures
    >>> import urllib.request
    >>> URLS = ['http://www.foxnews.com/',
    >>>         'http://www.cnn.com/',
    >>>         'http://some-made-up-domain.com/']
    >>> # Retrieve a single page and report the URL and contents
    >>> def load_url(url, timeout):
    >>>     with urllib.request.urlopen(url, timeout=timeout) as conn:
    >>>         return conn.read()
    >>> # We can use a with statement to ensure threads are cleaned up promptly
    >>> with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    >>>     # Start the load operations and mark each future with its URL
    >>>     future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    >>>     for future in concurrent.futures.as_completed(future_to_url):
    >>>         url = future_to_url[future]
    >>>         try:
    >>>             data = future.result()
    >>>         except Exception as exc:
    >>>             print('%r generated an exception: %s' % (url, exc))
    >>>         else:
    >>>             print('%r page is %d bytes' % (url, len(data)))
    'http://www.cnn.com/' page is 1142986 bytes
    'http://www.foxnews.com/' page is 292598 bytes
    'http://some-made-up-domain.com/' generated an exception: HTTP Error 403: Forbidden

See the `Python concurrent.futures page <https://docs.python.org/3.7/library/concurrent.futures.html>`_ for more info.
