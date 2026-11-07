# Security policy

promptfence is a detection layer, not a guarantee. A bypass, meaning an input that should be flagged but scores below the "suspicious" threshold, is the most useful report you can send.

## Reporting

- **Bypasses and false negatives:** open a regular issue with the input (shortened if possible), the score you got, and the version. These are expected and help the rule pack and dataset improve.
- **Vulnerabilities in the library itself** (for example a crash, catastrophic regex backtracking, or anything that lets untrusted input affect the host process): use GitHub's private vulnerability reporting on this repository instead of a public issue.

## Supported versions

Only the latest release on `main` receives fixes.
