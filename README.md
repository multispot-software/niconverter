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

# Cite

If you use this code, please cite the paper:

> 48-spot single-molecule FRET setup with periodic acceptor excitation <br>
> A. Ingargiola, M. Segal, A. Gulinatti, I. Rech, I. Labanca, P. Maccagnani, M. Ghioni, S. Weiss, X. Michalet <br>
> *bioRxiv* 156182; **2017**. doi: https://doi.org/10.1101/156182

----
Copyright (C) 2017 The Regents of the University of California, Antonino Ingargiola and contributors.

*This work was supported by NIH grants R01 GM069709 and R01 GM095904.*
