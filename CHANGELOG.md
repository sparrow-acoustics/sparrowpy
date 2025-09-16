# Changelog

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
