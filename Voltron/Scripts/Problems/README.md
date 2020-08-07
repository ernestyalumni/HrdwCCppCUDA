# Previous Versions

## `/Scripts/Problems/__init__.py`

```
import pathlib
import sys

dummy_variable = 2

def add_Voltron_system_path():

    print("__init__, pathlib.Path(__file__):", pathlib.Path(__file__).resolve())
    print("__init__, pathlib.Path(__file__):", pathlib.Path(__file__).resolve().parent.parent.parent)

    #sys.path.insert(0, pathlib.Path(__file__).resolve().parent.parent.parent)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
```

cf. https://stackoverflow.com/questions/31291608/effect-of-using-sys-path-insert0-path-and-sys-pathappend-when-loading-modul
python insert sys path

https://stackoverflow.com/questions/8663076/python-best-way-to-add-to-sys-path-relative-to-the-current-running-script
Python: Best way to add to sys.path relative to the current running script

https://stackoverflow.com/questions/3430372/how-do-i-get-the-full-path-of-the-current-files-directory
How do I get the full path of the current file's directory?

https://stackoverflow.com/questions/3144089/expand-python-search-path-to-other-source
Expand Python Search Path to Other Source
sys path add python 3

https://docs.python.org/3/library/importlib.html#module-importlib
importlib — The implementation of import¶

## `/Scripts/Problems/playground.py`

"""
@file playground.py
"""
import pathlib

import sys

from __init__ import *

print("\nBefore sys.path:", sys.path)

add_Voltron_system_path()

print("\nAfter sys.path:", sys.path)


import Voltron

from Voltron import Algorithms

#from playground2 import *

if __name__ == "__main__":

    print("\ndir(Voltron):", dir(Voltron))
    print("\ndir(Algorithm):", dir(Algorithms))


    # For the directory of the script being run:
    print(pathlib.Path(__file__).parent.absolute())
    # For the current working directory:
    print(pathlib.Path().absolute())

    print(pathlib.Path(__file__).resolve().parent.parent.parent)

    print(dummy_variable)

    #add_Voltron_system_path()    

    print("sys.path:", sys.path)
    print("\n\nsys.modules", sys.modules, "\n\n")
    print("In __main__, printing")