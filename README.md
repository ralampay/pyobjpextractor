# PyObjPextractor

Tool for extracting object proposals

## Installing Dependencies (via `conda`)

1. Rename `environment.yml.dist` to `environment.yml`

2. Install the packages by running `conda env update --prefix [environment_location] --file environment.yml --prune`

## Installing the Package

This will also install the cli utility `pyobjpextractor-cli`

```
pip install .
```

## Uninstalling

```
pip uninstall pyobjpextractor
```

## Sample Usage

```
pyobjpextractor-cli --input-file [path_to_image]
```
