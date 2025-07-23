# Installation

Here you will learn how to install the `nonconform` package.

## Prerequisites

- Python 3.12 or higher is recommended

## Installation with pip

To install `nonconform`, run the following command in your terminal:

```bash
pip install nonconform
```
## Optional dependencies

Existing optional dependencies are grouped into `[data]`, `[dev]`, `[docs]`, `[deep]`, `[fdr]` and `[all]`:
- `[data]`: Dataset loading functionality (includes `pyarrow`)
- `[dev]`: Development dependencies
- `[docs]`: Documentation dependencies
- `[deep]`: Deep Learning dependencies (`pytorch`)
- `[fdr]`: Online False Discovery Rate control (`online-fdr`)
- `[all]`: All optional dependencies

### Installing with specific dependencies

To install with datasets support:
```bash
pip install nonconform[data]
```

**Note**: The datasets are downloaded automatically when first used and cached both in memory and on disk (in `~/.cache/nonconform/`) for faster subsequent access.

To install with online FDR control for streaming scenarios:
```bash
pip install nonconform[fdr]
```

To install with all optional dependencies:
```bash
pip install nonconform[all]
```

## Get started

You are all set to find your first anomalies!

```bash
import nonconform
```