# niconverter

This library allows converting raw multispot (96-ch) data files to Photon-HDF5 files. 
The input data files contains photon timestamps and detectors in a binary format
saved by a [multichannel LabVIEW-FPGA timestamping code](https://github.com/multispot-software/MultichannelTimestamper).

The data processing is done in chunks so that
arbitrary large data files can be converted with constant RAM usage.

For more info see the niconverter module's docstring.

# Installation

To install, download the repository, enter the source folder and type:

`pip install .`

----
Copyright (C) 2017 The Regents of the University of California, Antonino Ingargiola and contributors.

*This work was supported by NIH grants R01 GM069709 and R01 GM095904.*
