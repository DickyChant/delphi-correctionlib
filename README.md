# delphi-correctionlib

Corrections for the DELPHI open data in EDM4hep format, stored as
[correctionlib](https://github.com/cms-nanoAOD/correctionlib) JSON files.

## Contents

| Directory | Correction | Datasets |
|---|---|---|
| [`tracking/`](tracking/README.md) | charged-track momentum scale | 94C2, 95D2 (data) |

## Dependencies

You need correctionlib (Python and C++, version 2.6 or later). The scripts in
`tracking/scripts`, to remake the corrections, also need numpy, scipy, iminuit and
podio; the key4hep stack provides them.

Compile C++ code with `$(correction config --cflags --ldflags --rpath)`, or in CMake use
`find_package(correctionlib)` with `$(correction config --cmake)`.
