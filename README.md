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

## Sample Usage (single input image)

```
pyobjpextractor-cli --input-file [path_to_image]
```

## Sample Usage (for use with video)


```
pyobjpextractor-cli --min-area 1250 --max-area -1 --mode sfg --video-index 0
```

This will fetch object proposals that have a minimum area of `1250` pixels without any maximum area limit (`-1`)
