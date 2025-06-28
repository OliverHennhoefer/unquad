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

Existing optional dependencies are grouped into `[data]`, `[dev]`, `[docs]`, `[dl]` and `[all]`:
- `[data]`: Dataset loading functionality (includes `pyarrow`)
- `[dev]`: Development dependencies
- `[docs]`: Documentation dependencies
- `[dl]`: Deep Learning dependencies (`tensorflow`, `pytorch`)
- `[all]`: All optional dependencies

### Installing with specific dependencies

To install with datasets support:
```bash
pip install unquad[data]
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