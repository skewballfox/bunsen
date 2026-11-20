# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.43.0](https://github.com/skewballfox/bunsen/compare/bunsen-catalog-v0.42.0...bunsen-catalog-v0.43.0) - 2026-11-20

### Added

- *(qwen3_tts)* the Qwen3-TTS catalog rows and their loader ([#769](https://github.com/skewballfox/bunsen/pull/769))
- *(zonos3)* the `zyphra/zonos3/zonos3-mini` catalog row and its loader ([#709](https://github.com/skewballfox/bunsen/pull/709))

### Other

- [**breaking**] the rotary base is an `f64` ([#792](https://github.com/skewballfox/bunsen/pull/792))
- crutcher/tts voice info ([#729](https://github.com/skewballfox/bunsen/pull/729))
- [**breaking**] the attention softmax widens through `Widening`, as `softmax_widening` ([#715](https://github.com/skewballfox/bunsen/pull/715))

## [0.42.0](https://github.com/zspacelabs/bunsen/compare/bunsen-catalog-v0.41.0...bunsen-catalog-v0.42.0) - 2026-11-13

### Added

- *(resources)* provider descriptions, Hub mirrors, digest-filed downloads, and the cards rebuilt ([#686](https://github.com/zspacelabs/bunsen/pull/686))
- *(datasets)* [**breaking**] datasets are catalog builds ([#684](https://github.com/zspacelabs/bunsen/pull/684))
- *(pretrained)* [**breaking**] the kits load from the built-in catalog ([#678](https://github.com/zspacelabs/bunsen/pull/678))
- *(catalog)* the built-in catalog crate, and ryl YAML across the tree ([#677](https://github.com/zspacelabs/bunsen/pull/677))

### Other

- *(data)* [**breaking**] the catalog, resource and build names ([#681](https://github.com/zspacelabs/bunsen/pull/681))
- *(data)* [**breaking**] the `data` tree: resources, catalog, formats, builds, datasets ([#680](https://github.com/zspacelabs/bunsen/pull/680))
