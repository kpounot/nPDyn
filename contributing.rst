Contributions
=============

All contributions are welcome.
The developers try to follow the
`PEP8 <https://www.python.org/dev/peps/pep-0008/>` for coding style and
the `PEP257 <https://www.python.org/dev/peps/pep-0257/>`_ for docstrings.

To ensure proper commit, we use `pre-commit <https://pre-commit.com/>`_.

Code
----
Below is the expected workflow for code contributions:

- clone the repository, ``git clone https://github.com/kpounot/nPDyn``.
- install dependencies, ``python3 -m pip install -r requirements-dev.txt``
- initialize pre-commit, ``pre-commit install``
- create a new branch, ``git checkout -b <myNewBranchName>``
- start coding
- run pytest, ``pytest nPDyn``
- make sure the documentation is complete,
  ``cd docs``
  ``make html``
  and check your new doc, if any, with your web browser.
- push your work, ``git push origin``
- open a Pull Request

Ideas, comments
---------------
Please use the `google group <https://groups.google.com/g/npdyn>`_.
