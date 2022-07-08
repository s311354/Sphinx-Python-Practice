
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

+ str.isdigit(): return True if all characters in the string are digits and there is at least one characters, False otherwise.

+ str.split(sep=None, maxsplit=-1): return a list of the words in the string, using sep as the delimiter string.

+ str.endswith(suffix[, start[, end]]): return True if the string ends with the specified suffix, otherwise return False. suffix can be a tuple of suffixs to look for. With optional start, test begining at that position. With optional end, stop comparing at that position.

+ str.strip([chars]): return a copy of the string with the leading and trailing characters removed::

    >>> '   spacious   '.strip()
    spacious

+ str.index(sub[, start[, end]]): like find(), but raise ValueError when the substring is not found.


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

This module provides an implementation of the heap queue algorithm, also known as the priority queue algorithm.

This implementation uses arrays for which heap[k] <= heap[2*k + 1] and heap[k] <= heap[2*k+2] for all k, counting elements from zero. These make it possible to view that heap[0] is the smallest item, and heap.sort() maintains the heap invariant!

Heaps are binary trees for which every parent node has a value less than or equal to any of its children.

The example of using heapq::

    >>> In [17]: heapq.heappush(heap, (3, 1, 2, 0))
    >>> In [18]: heapq.heappush(heap, (1, 1, 0, 0))
    >>> In [19]: heap
    Out[19]: [(1, 1, 0, 0), (3, 1, 2, 0)]

Heapq
^^^^^^

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

