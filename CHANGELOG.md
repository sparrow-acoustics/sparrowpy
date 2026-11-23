# Changelog

## Unreleased

### Added

- CI: allow manual trigger of the tests on GitHub Actions (#164)
- CI: run the tests on ``main`` and ``develop`` once a week (#164)
- DOC: note in the README and on the documentation landing page that sparrowpy is part of [CHORAS](https://github.com/choras-org/CHORAS), including the limitations of the integration (#171)

### Changed

- compute the energy exchange of ``DirectionalRadiosityFast`` for all frequency bins in parallel (#166)
- ``DirectionalRadiosityFast.bake_geometry`` uses a ``visibility_matrix`` passed to the constructor instead of computing it again (#167)
- ``DirectionalRadiosityFast.init_source_energy`` accepts a ``source_visibility`` and ``collect_energy_receiver_mono`` / ``collect_energy_receiver_patchwise`` accept a ``receiver_visibility``, so the point to patch visibility can be computed outside of sparrowpy (#167)

### Fixed

- DOC: adapted the documentation to sphinx-design 0.8.0 (#172)
- compile ``pt_solution`` and ``load_stokes_entries`` without ``parallel=True``, as they are only called from parallel loops; the nested parallel launch aborted the process with numba's default ``workqueue`` threading layer (e.g. on macOS arm64) (#165)
- cache numba compilations on disk (``cache=True``), so only the first run pays the JIT compilation time (#165)
- ``DirectionalRadiosityFast.calculate_direct_sound`` does not add the direct sound if source and receiver do not see each other; the visibility is checked separately for each source-receiver pair (#170)
- ``_point_in_polygon`` counted a polygon vertex lying exactly on its test line once per adjacent side, so points level with a vertex could be reported inside; this made ``_check_patch2patch_visibility`` report visible patch pairs as blocked (#168)

## 1.1.1 (2026-08-31)

### Fixed

- fix missing early energy in impulse response synthesis (#163)
- force complex brdfs to real to prevent numba errors (#162)

## 1.1.0 (2026-08-25)

### Added

- added module ``dsp`` to generate impulse responses from energy time curves (#124)
- added Python 3.14 support

## 1.0.2 (2026-08-25)

### Fixed

- changed default python version to 3.13, as blender did (#160)
- adapted to new pyfar version (#160)
- remove ``numpy<2`` restriction, as it is not relevant anymore (#160)
- DOC: Small typo in Readme, numba does jit compilation (#135)


## 1.0.1 (2025-09-16)

### Added

- added deploy instructions to contribution guidelines (#121)
- added Zenodo publication config (#130)

## 1.0.0 (2025-08-25)

This is a major release, so backwards compatibility is not guaranteed.

### Added

- Read and write support for DirectionalRadiosityFast (#81)
- Source-patch and patch-receiver visibility (#90, #91)
- Source directivity (#98)

### Changed

- Improved API and documentation for RadiosityFast (#70, #97)
- Improved form-factor calculation (#77, #100)
- Improved test performance (#99)
- Updated icons (#87)
- Replaced HISTORY.rst by CHANGELOG.md (#117)

### Removed

- Removed progress bar and its dependency tqdm (#113)

### Fixed

- Fixed license definition in project config (#118)

## 0.1.1 (2025-08-25)

### Fixed

- Fix wrong definition of the brdf calculation in the `brdf` module (#104)
- Fix form factor for patches of difference area (#62)

### Changed

- Improve documentation structure and content (#66, #72, #92, #93, #107)
- Add a note on a known issue with the `DirectionalRadiosityFast` class for
  non-perfectly Lambertian surfaces (#108)
- improve ci testing and release on Github Actions (#114, #115)

### Deprecated

- Deprecate Python 3.9 support (#116)

## 0.1.0 (2025-03-18)

- First release on PyPI.
