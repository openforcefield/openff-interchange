# Experimental features

Some Interchange features are available in the public API but NOT suitable for production. These are the sharper edges of the project that we hope to make more stable and reliable in the future. Generally these features are somewhat complete and partially tested, but not completely finished and in need of more extensive testing before their use is considered safe.

In order to facilitate testing these features, though, they are included in the package on an "opt-in" basis. These features are accessible by setting the environment variable `INTERCHANGE_EXPERIMENTAL=1`. By setting this variable the user accepts that they are accessing a portion of the codebase that is **not stable**, highly likely to produce **inaccurate results**, and not guaranteed to run without error. For these reasons, all features tagged as experimental are NOT recommended for production.

## Current set of experimental features

See each function's docstring for more detailed information.

* `Interchange.from_gromacs`: Import data from GROMACS files.

## Using experimental features

Testing and user feedback on experimental features is very welcome! To access an experimental feature, set  `INTERCHANGE_EXPERIMENTAL=1` when starting a Python interpreter:

```shell
$ INTERCHANGE_EXPERIMENTAL=1 python my_script.py
...
```

or Jupyter tool of choice:

```shell
$ INTERCHANGE_EXPERIMENTAL=1 jupyter-lab my_notebook.ipynb
...
```

or, with cloud services like Google Colab, use the [`%env`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-env) magic command:

```
%env INTERCHANGE_EXPERIMENTAL=1
```

Feedback of any sort is useful - even (and especially) if something crashes early or produces surprising results. Please raise an [issue on GitHub](https://github.com/openforcefield/openff-interchange/issues/new).

## Implementation

Experimental features are tagged via a decorator that wraps the particular functions/methods. An exception is raised if the environment variable error `INTERCHANGE_EXPERIMENTAL` is not set to `1`. Consider a simple script `hello.py`:

```python
from openff.interchange._experimental import experimental


@experimental
def say_hello():
    """Greet the user."""
    print("Hello!")


say_hello()
```

By default, this function errors when called:

```shell
$ python hello.py
Traceback (most recent call last):
  File "/Users/mattthompson/software/openff-interchange/hello.py", line 8, in <module>
    say_hello()
  File "/Users/mattthompson/software/openff-interchange/openff/interchange/_experimental.py", line 27, in wrapper
    raise ExperimentalFeatureException(
openff.interchange.exceptions.ExperimentalFeatureException:
Function or method say_hello is experimental. This feature is not complete, not yet reliable, and/or needs more testing to be considered suitable for production.
    To use this feature on a provisional basis, set the environment variable INTERCHANGE_EXPERIMENTAL=1.
```

but runs as expected after setting the environment variable:

```shell
$ INTERCHANGE_EXPERIMENTAL=1 python hello.py
Hello!
```
