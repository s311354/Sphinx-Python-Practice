# Leetcode Practice Project

### Quick Start

```
$ cd doc/

$ sphinx-quickstart --ext-autodoc --ext-doctest --ext-mathjax --ext-viewcode
Welcome to the Sphinx 3.1.2 quickstart utility

Please enter values for the following settings (just press Enter to
accept a default value, if one is given in brackets
Selected root path: . 

You have two options for placing the build directory for Sphinx output.
Either, you use a directory "_build" within the root path, or you separate
"source" and "build" directories within the root path.
> Separate source and build directories (y/n) [n]: y

The project name will occur in several places in the built documentation.
> Project name: leetcode practice
> Author name(s): Louis Liu
> Project release []: 1.0

If the documents are to be written in a language other than English,
you can select a language here by its language code. Sphinx will then
translate text that it generates into that language.

For a list of supported codes, see
https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language.
> Project language [en]: 

Creating file /Users/shi-rongliu/Documents/GitHub/python_practice/Sphinx-Python-Practice/source/conf.py.
Creating file /Users/shi-rongliu/Documents/GitHub/python_practice/Sphinx-Python-Practice/source/index.rst.
Creating file /Users/shi-rongliu/Documents/GitHub/python_practice/Sphinx-Python-Practice/Makefile.
Creating file /Users/shi-rongliu/Documents/GitHub/python_practice/Sphinx-Python-Practice/make.bat.

Finished: An initial directory structure has been created.

You should now populate your master file /Users/shi-rongliu/Documents/GitHub/python_practice/Sphinx-Python-Practice/source/index.rst and create other documentation
source files. Use the Makefile to build the docs, like so:
   make builder
where "builder" is one of the supported builders, e.g. html, latex or linkcheck.
```

### Create HTML

```
$ cd doc/
$ make clean
$ make html
```

### Create PDF

```
$ cd doc/
$ make clean
$ make latexpdf
```

### Run Doctest

```
$ make doctest
Running Sphinx v3.1.2
WARNING: while setting up extension sphinx.addnodes: node class 'meta' is already registered, its visitors will be overridden
loading pickled environment... done
building [mo]: targets for 0 po files that are out of date
building [doctest]: targets for 3 source files that are out of date
updating environment: 0 added, 0 changed, 0 removed
looking for now-outdated files... none found
running tests...

Doctest summary
===============
    0 tests
    0 failures in tests
    0 failures in setup code
    0 failures in cleanup code
build succeeded, 1 warning.

Testing of doctests in the sources finished, look at the results in build/doctest/output.txt.
```

### View Documentation

```
$ cd doc/build/html/
$ python -m http.server
Serving HTTP on :: port 8000 (http://[::]:8000/) ...
```

We could then view the documentation in the web browser at [http://[::]:8000/](http://[::]:8000/).
