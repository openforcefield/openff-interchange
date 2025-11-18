# Compiling Interchange documentation

The docs for this project are built with [MyST](https://myst-parser.readthedocs.io/en/latest/index.html).
To compile the docs, first ensure that the MyST parser is installed.

```shell
conda install -c conda-forge myst-parser
```

Once installed, you can use the `Makefile` in this directory to compile static HTML pages by

```shell
make html
```

The compiled docs will be in the `_build` directory and can be viewed by opening `index.html`.
