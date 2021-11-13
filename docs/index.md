# OpenFF Interchange

A project (and object) for storing, manipulating, and converting molecular mechanics data.

**Please note that this software in an early and experimental state and unsuitable for production.**

```{toctree}
---
caption: Installation
maxdepth: 2
---
installation.md
developing.md
releasehistory.md
```

```{toctree}
---
caption: User's guide
maxdepth: 2
---

using/intro.md
using/design.md
using/output.md
using/migrating.md
using/energy_tests.md

```

<div class="toctree-wrapper"><p class="caption" role="heading"><span class="caption-text">
API Reference
</span></p></div>

<!--
The autosummary directive renders to rST,
so we must use eval-rst here
-->
```{eval-rst}
.. autosummary::
    :recursive:
    :caption: API Reference
    :toctree: _autosummary
    :nosignatures:

    openff.interchange
```
