# Installation

Here you will learn how to install the `unquad` package.

## Prerequisites

- Python 3.12 or higher is recommended

## Installation with pip

To install `unquad`, run the following command in your terminal:

```bash
pip install unquad
```
## Optional dependencies

Existing optional dependencies are grouped into `[data]`, `[dev]`, `[docs]`, `[dl]`, `[fdr]` and `[all]`:
- `[data]`: Dataset loading functionality (includes `pyarrow`)
- `[dev]`: Development dependencies
- `[docs]`: Documentation dependencies
- `[dl]`: Deep Learning dependencies (`tensorflow`, `pytorch`)
- `[fdr]`: Online False Discovery Rate control (`online-fdr`)
- `[all]`: All optional dependencies

### Installing with specific dependencies

To install with datasets support:
```bash
pip install unquad[data]
```

**Note**: The datasets are downloaded automatically when first used and cached in memory. This approach has zero disk footprint - no files are stored on your system.

To install with online FDR control for streaming scenarios:
```bash
pip install unquad[fdr]
```

To install with all optional dependencies:
```bash
pip install unquad[all]
```

## Get started

You are all set to find your first anomalies!

```bash
import unquad
```