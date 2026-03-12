Release Workflow
================

GitHub Actions builds the package on every push and pull request to ``main``.

Publishing is commit-message driven:

- ``#major`` bumps ``+1.0.0``
- ``#minor`` bumps ``+0.1.0``
- ``#patch`` bumps ``+0.0.1``

If none of those markers are present, CI runs but no release is created.

Automated PyPI publishing requires PyPI trusted publishing to be configured for project ``pldflow``.
