
Recap Python basic conception and alogrithm
================================================

Welcome! This is the personal documentation for learning the basic conception of python.

dataclasses - Data Classes
--------------------------

This module provides a decorator and functions for automatically adding generated special methods such as __init__() and __repr__() to user-defined classes.

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

See the `Python dataclasses page <https://docs.python.org/3.7/library/dataclasses.html?highlight=class#module-dataclasses>`_ for more info.

Container Datatypes
--------------------------

This module implements specialized container datatypes providing alternatives to python's general purpose built-in container, **list**, **tuple**, **set** and **dict**.

Sequence Types - list, tuple, range
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

See the `Python Tuples page <https://docs.python.org/3.7/library/stdtypes.html#tuples>`_ for more info.

Ranges
'''''''''''''

The range type represents an immutable sequence of numbers and is commonly used for looping a specific number of times in for loops.

class range(stop), class range(start, stop[, step]) may be constructed in integers.

Range example::

    >>> list(range(10))
    >>> list(range(1, 11))


See the `Python Ranges page <https://docs.python.org/3.7/library/stdtypes.html?highlight=range#ranges>`_ for more info.

Mapping Types - dict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*A mapping object maps hashable values to arbitrary objects*. A dictionary's keys are almost arbitrary values. Values that are not hashable, that is, values containing lists, dictionaries or other mutable types may not be used as keys.

class dict(** kwargs) dictionaries can be created by several mean:

+ Use a comma-separeted list or key: value pairs within braces: {'jack': 4098, 'sjoerd': 4227}
+ Use a dict comprehension: {}
+ Use the type constructor: dict()

The examples all return a dictionary equal to {"one": 1, "two": 2, "three": 3}

    >>> a = dict(one = 1, two = 2, three = 3)
    >>> b = {'one': 1, 'two': 2, 'three': 3}

See the `Python collections - Container datatypes page <https://docs.python.org/3.7/library/collections.html#module-collections>`_ for more info.

Set Types - set, frozenset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A set object is an *unordered collection of distinct hashable objects*. Common uses include membership testing, removing duplicates from a sequence, and computing mathematical operations. such as intersection, union, difference, and symmetric difference.

class set([iterable]) can be created by several means:

+ Use a comma-separated list of elements within braces: {'jack', 'sjoerd'}
+ Use the type constructor: set(), set('foobar')

The example all return a set equal to {'a', 'b'}::

    >>> a = {'a', 'b'}
    >>> b = set('a', 'b')

See the `Python Set Types - set, forzenset page <https://docs.python.org/3.7/library/stdtypes.html#set-types-set-frozenset>`_ for more info.

