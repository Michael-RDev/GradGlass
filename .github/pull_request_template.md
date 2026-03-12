## Summary

<!-- One sentence: what does this PR do and why? -->

Fixes #<!-- issue number, if applicable -->

---

## Type of change

<!-- Mark all that apply with an `x` -->

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `refactor` — code change that is neither a fix nor a feature
- [ ] `test` — adding or updating tests only
- [ ] `docs` — documentation only
- [ ] `chore` — build, CI, or tooling change
- [ ] `perf` — performance improvement

---

## Checklist

<!-- Every box must be checked before merging. Strike through (~~item~~) items that are genuinely not applicable and briefly explain why. -->

### Code quality
- [ ] `ruff check .` passes with no errors
- [ ] `ruff format --check .` passes (run `ruff format .` to auto-fix)
- [ ] All public functions and classes have docstrings
- [ ] All public signatures are type-annotated

### Tests
- [ ] New behaviour is covered by at least one test
- [ ] All existing tests pass (`pytest tests/ --ignore=tests/data_leakage.py`)
- [ ] No test file is left without a `test_` prefix (standalone scripts won't be auto-collected)

### Documentation
- [ ] Relevant docs under `docs/` are updated or added
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]`

### Analysis tests (only if adding/modifying a built-in analysis test)
- [ ] Test ID is `SCREAMING_SNAKE_CASE` and unique across `gradglass/analysis/builtins.py`
- [ ] Correct `TestCategory` and `TestSeverity` chosen
- [ ] A corresponding pytest case exists in `tests/`

---

## How to test

<!-- Step-by-step instructions a reviewer can follow to manually verify the change. -->

```bash
# Example
pip install -e ".[dev,torch]"
pytest tests/ --ignore=tests/data_leakage.py -v
```

---

## Screenshots / recordings

<!-- Delete this section if not applicable (UI changes, dashboard, CLI output). -->
