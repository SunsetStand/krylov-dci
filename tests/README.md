# Test organization

- `unit/`: focused checks of one module or helper contract.
- `integration/`: end-to-end workflows and real backend integration checks.
- `regression/`: stable numerical baselines that protect scientific behavior.

These files are executable test entry points rather than an importable test
package, so the category directories intentionally do not add `__init__.py`.
On the PKU server, run them only through launchers in `batch/diagnostics/`.
