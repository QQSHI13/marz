# Releasing

Three registries, one tag. Nothing is published by hand, and there are no
publish tokens stored in this repository.

- **PyPI** — `marz` (wheels + sdist)
- **npm** — `marz-search` (TypeScript wrapper + the compiled `.wasm`)
- **crates.io** — `marz-core` (the engine as a Rust library)

`marz-wasm` and the `marz` Rust crate are `publish = false`. Neither is usable
as a Rust dependency: one is a `cdylib` for wasm-bindgen, the other a CPython
extension that links against no Python at build time.

## How the credentials work

All three registries support **trusted publishing**: GitHub Actions mints a
short-lived OpenID Connect token, the registry verifies it came from this
repository and this workflow file, and exchanges it for an upload credential
valid for a few minutes.

This is worth the setup over an API token for reasons that matter here:

- Nothing secret lives in the repository, so nothing can leak from it and there
  is nothing to rotate.
- The registry checks *which workflow file* asked. A token stolen from CI logs
  can publish from anywhere; an OIDC claim cannot.
- PyPI publishes get a provenance attestation automatically, so anyone can
  verify a wheel came from this repo at this commit.

The cost is that each registry needs configuring once, in its web UI, before the
first publish. That's the rest of this document.

> The registry UIs change. Where a field name below does not match what you see,
> the concepts still map: each registry wants to know the repository, the
> workflow filename, and (for PyPI) the GitHub environment.

## One-time setup

### 1. GitHub environments

`release.yml` names three environments: `pypi`, `npm`, `crates-io`. PyPI's
trusted publisher matches on the environment name, so this comes first.

Settings → Environments → New environment, once for each name. No secrets or
variables — the environments exist to be named.

Optionally add yourself as a **required reviewer** on each. That turns every
publish into a button you press, which is a reasonable thing to want given that
none of these uploads can be taken back.

### 2. PyPI

The project does not exist yet, so this is a **pending publisher** — PyPI holds
the configuration and creates the project on first upload. This also reserves
the name.

PyPI → Your account → Publishing → Add a new pending publisher (GitHub):

| Field | Value |
|---|---|
| PyPI Project Name | `marz` |
| Owner | `QQSHI13` |
| Repository name | `marz` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The workflow name is the **filename**, not the `name:` inside the file.

Consider doing this on [TestPyPI](https://test.pypi.org) first with the same
values. It's a real end-to-end rehearsal of the credential exchange, and a
mistake there costs nothing.

### 3. npm

npm's trusted publishing requires the package to **already exist**, which means
the first publish cannot use it. So:

1. Publish `0.1.0` once from your machine with a token:
   ```sh
   cd js && npm login && npm run build && npm publish
   ```
   Check the tarball first — see "Verifying before you publish" below.
2. Then npm → the `marz-search` package → Settings → Trusted Publisher →
   GitHub Actions, with repository `QQSHI13/marz` and workflow `release.yml`.
3. Set **Publishing access** to *Require trusted publishing*, which disables
   token publishing for this package. Then revoke the token from step 1.

From `0.1.1` on, npm publishes through the workflow like the other two.

### 4. crates.io

crates.io also requires the crate to exist before a trusted publisher can be
configured, and it does not have a pending-publisher flow.

1. Publish once from your machine:
   ```sh
   cargo login          # paste a token from crates.io/settings/tokens
   cargo publish -p marz-core
   ```
2. Then crates.io → `marz-core` → Settings → Trusted Publishing → Add, with
   repository `QQSHI13/marz`, workflow `release.yml`, environment `crates-io`.
3. Revoke the token.

`cargo publish` is **permanent** — a version can be yanked, which stops new
dependents from resolving it, but never deleted or replaced.

## Cutting a release

1. Bump the version in **two** places — they must agree, and `check-version`
   fails the release if they don't:
   - `Cargo.toml` → `[workspace.package]` → `version`
   - `js/package.json` → `version`
2. Commit, and wait for CI to pass on `main`.
3. Tag and push:
   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```

The tag triggers `release.yml`. It re-runs the full test suite at the tag,
builds wheels for Linux x86_64/aarch64, macOS universal2 and Windows x64, tests
each wheel by installing it, builds and checks the sdist, then publishes to all
three registries.

Tags are matched as `v*`, so the `v` prefix is required.

### Rehearsing

Actions → Release → Run workflow, with **publish** left off. Every job runs and
uploads nothing. Worth doing before the first real release, and after any change
to this workflow.

## Verifying before you publish

The npm tarball has shipped without its engine before. wasm-pack writes a
`.gitignore` containing `*` into its output directory; npm reads a nested
`.gitignore` as an `.npmignore`, and that beats the `files` allowlist in
package.json. The result installs, imports, and fails on the first search.

`scripts/build-wasm.sh` removes that file, and both CI and the release workflow
assert the tarball contains `pkg/marz_wasm_bg.wasm`. To check by hand:

```sh
cd js && npm run build && npm pack --dry-run
```

Expect ~8 files and ~98 kB, including a 175 KB `pkg/marz_wasm_bg.wasm`. If you
see 5 files and 6 kB, the engine is missing — do not publish.

## What cannot be undone

- **PyPI**: a version can be deleted, but the number can never be reused. A bad
  `0.1.0` means the next release is `0.1.1`.
- **crates.io**: versions are permanent. Yanking hides a version from new
  resolution but does not remove it.
- **npm**: unpublish is allowed within 72 hours, and only if nothing depends on
  it. After that, `npm deprecate` is the remedy.

None of these are reasons to be slow; they are reasons the workflow runs the
tests at the tag and inspects the artifacts before uploading.
