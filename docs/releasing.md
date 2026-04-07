# Releasing GradGlass

GradGlass is launched from the GitHub repository [`Michael-RDev/GradGlass`](https://github.com/Michael-RDev/GradGlass).

## CI and release flow

1. Open a pull request into `main`.
2. Let the `CI` workflow pass:
   - Ruff lint and format checks
   - Python test matrix
   - Dashboard utility tests
   - Package build verification
3. Merge into `main`.
4. Tag the release commit as `vX.Y.Z`.
5. Publish a GitHub Release for that tag.
6. Let the `Release` workflow build artifacts, attach them to the release, and publish to PyPI.

## PyPI Trusted Publisher

GradGlass is configured to publish from GitHub Actions using PyPI Trusted Publisher rather than a long-lived API token.

Configure PyPI to trust:

- Owner: `Michael-RDev`
- Repository: `GradGlass`
- Workflow: `release.yml`
- Environment: `pypi`

If the PyPI name `gradglass` is unavailable, stop and rename the package before enabling the publisher.

## Local checks before tagging

Run:

```bash
pytest
npm --prefix gradglass/dashboard test
python -m build
```

`python -m build` now enforces the presence of `gradglass/dashboard/dist`. If `node_modules` is already present in the
dashboard project, the packaging step attempts `npm --prefix gradglass/dashboard run build` automatically; otherwise it
fails with instructions to build the frontend first.

Make sure local workspaces, caches, build outputs, and generated example artifacts are not part of the release commit.
