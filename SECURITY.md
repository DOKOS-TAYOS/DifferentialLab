# Security Policy

DifferentialLab is a desktop research and teaching tool. Please report security issues privately so they can be fixed before public discussion.

## Supported Versions

Security fixes target the latest development state on `dev` and the latest release branch/tag when one exists.

## Reporting a Vulnerability

Use GitHub private vulnerability reporting if it is enabled for the repository. If that is not available, contact the maintainer at the email listed in `pyproject.toml`.

Please include:

- a short description of the issue
- steps to reproduce it
- the affected platform and Python version
- any safe proof-of-concept input, without real secrets or personal data

## Dependency and Code Scanning

The repository uses Dependabot for Python and GitHub Actions updates, `pip-audit` for Python dependency advisories, and GitHub CodeQL default setup for code scanning.
