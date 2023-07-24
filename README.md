# ProteiML
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/{666693326}.svg)](https://zenodo.org/badge/latestdoi/{666693326})
#[![Build Status]()

### CITATION
**If you use the machine learning platform, please cite our [pre-print](https://doi.org/10.1101/2023.04.13.536801)!**  



`ProteiML` is a machine learning ensemble pipeline that reads in sequence-to-function mutational datasets and outputs suggested mutations in order to optimize protein functionality. 

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Setting up the development environment](#setting-up-the-development-environment)
- [License](#license)
- [Issues](https://github.com/sarahwaity/ProteiML/issues)

# Overview
`ProteiML` represents is a plug-and-play tool tailored for protein engineers seeking to enhance the optimization of functional proteins. `ProteiML`'s power and utility resides in its human-out-of-loop decision-making capability. `ProteiML` learns from large mutation datasets and extracts mutational trends that often evade researchers capacity to percieve in large datasets. These insights, which researchers may find challenging to ascertain, are synthesized into a comprehensive list of suggested mutations. Furthermore, the platform accomplishes this feat within a remarkably short timeframe, delivering results in a matter of hours—a process that would otherwise demand days when employing traditional avenues like protein structure analysis and literature review.


# System Requirements
## Hardware requirements
`ProteiML` requires only a standard computer with enough RAM to support the in-memory operations, however the speed of computation can be improved with larger RAMs.

## Software requirements
### OS Requirements
This package is supported for *macOS* and *Linux*. The package has been tested on the following systems:
+ macOS: Ventura (13.0.1)
+ Linux: Ubuntu 16.04

### Python Dependencies
`ProteiML` mainly depends on the Python scientific stack. 
*A full list of dependencies can be found in the pyproject.toml file.*

- [numpy](http://www.numpy.org/) (>=1.16.0)
- [pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [scikit-learn] (https://scikit-learn.org/stable/)
- [seaborn] (https://seaborn.pydata.org/)



# Installation Guide

### Setting up the development environment
<img src=DOCS/workflow.png width=500 align="right" vspace = "50">


# License
This project is covered under the **MIT License**.


# Issues
Report a bug by [opening a new issue](https://github.com/sarahwaity/ProteiML/issues).
