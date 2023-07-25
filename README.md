# ProteiML
![Build Status](https://github.com/sarahwaity/ProteiML/actions/workflows/config.yml/badge.svg)
![coverage](https://img.shields.io/badge/coverage-94%25-green)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/proteiml/badge/?version=latest)](https://proteiml.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/666693326.svg)](https://zenodo.org/badge/latestdoi/666693326)

`ProteiML` is a machine learning ensemble pipeline that reads in sequence-to-function mutational datasets and outputs suggested mutations in order to optimize protein functionality. 

- [Overview](#overview)
- [Documentation](#documentation)
- [Methodology Description](#Methodology-Description)
- [Tutorial](#Tutorial)
- [System Requirements](#system-requirements)
- [Installation & Run Guide](#installation-&-Run-guide)
- [License](#license)
- [Issues](https://github.com/sarahwaity/ProteiML/issues)


### CITATION
*If you use `ProteiML`, please cite our [pre-print](https://doi.org/10.1101/2023.04.13.536801)!*  


# Overview
`ProteiML` represents is a plug-and-play tool tailored for protein engineers seeking to optimize functional proteins. `ProteiML`'s power and utility resides in its human-out-of-loop decision-making capability. `ProteiML` learns from large mutation datasets and extracts mutational trends that often evade researchers capacity to percieve in large datasets. These insights, which researchers may find challenging to ascertain, are synthesized into a comprehensive list of suggested mutations. Furthermore, the platform accomplishes this feat within a remarkably short timeframe, delivering results in a matter of hours—a process that would otherwise demand days when employing traditional avenues like protein structure analysis and literature review.

# Documentation
All Documentation for ProteiML is hosted through [ReadTheDocs](https://proteiml.readthedocs.io/en/latest/index.html)

# Tutorial
## Running a Demo of the code performing local testing:
[ProteiML Code Demo](https://www.youtube.com/watch?v=2YPP6vm1DtA)

# Methodology Description
<img src=workflow.png width=500 align="right" vspace = "50">
The user will provide and updated "user_input.csv" as well as a known mutation library into the backend data folder. The user also indicates whether to use a truncated version of the AAINDEX encoding dataset for a test run, or the full AAINDEX encoding dataset for true output prediction generation. Upon running of `ProteiML`, known mutation dataset will be read and formatted to match requirements for algorithm reading. The formatted variant library will then be used to create and novel point mutation library, as well as train three different regressor types {Random Forest Regressor (RFR), Multilayer Perceptron Regressor (MPNR), and K-Neighbors Regressor (KNR)}. The optimized models as well as the novel library are then used to generate predictions on novel sequences and cross-validation predictions on the witheld test set. The outputs from this step are advanced for form final ensemble predictions and cross-validation scores, which are provided to the user at the end for downstream processing. 


# System Requirements
## Hardware requirements
`ProteiML` requires only a standard computer with enough RAM to support the in-memory operations, however the speed of computation can be improved with faster CPUs.

## Software requirements
### OS Requirements
This package is supported for *macOS* and *Linux*. The package has been tested on the following systems:
+ macOS: Ventura (13.0.1)
+ Linux: Ubuntu 16.04

### Python Dependencies
`ProteiML` mainly depends on the Python scientific stack. 
*A full list of dependencies can be found in the pyproject.toml file.*

- [numpy](http://www.numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/)



# Installation & Run Guide
### Clone the GitHub repository
On the Github ProteiML website, click download ZIP or clone the repository using "https://github.com/sarahwaity/ProteiML.git"


### Setting Up the Virtual Environment:
The virtual environment is run through poetry and toml
To activate the virtual environment:
1. Open a new terminal instance
2. Navigate to the cloned ProteiML repository directory
     - if the directory was opened in vscode, this step can be skipped
3. run "pip intall poetry" in the command line
4. run "poetry install" in the command line
    - reads the pyproject.toml file and installs dependencies, if this has already been done it will give a "does not contain any element" warning. This is okay, it indicates the dependencies are already satisfied. 
5. run "poetry shell" in the command line
**Virtual environment is ready!**


### Running ProteiML:
*Test your setup using the 'streamlined_property_matrix.csv' to make sure no errors will cause run to fail!*
1. Run through all steps of "Setting up the Virtual Environment"
2. Edit the Input_Data.csv to include information about your run (more information can be found in docs/)
3. Update data_processing/constants.py to indicate whether to use the 'streamlined_property_matrix.csv' or the 'full_property_matrix.csv'
    - reccomended to test it first using the streamlined, if everything runs correctly, delete outputs and run with the full property matrix!
4. run "python main.py" in the command line
    - for 'streamlined_property_matrix.csv', this should take 5 minutes or less
    - for 'full_property_matrix.csv', this should take ~6 hours. 


### Running the test suite locally:
*To perform testing quickly, it is recommended to use 'streamlined_property_matrix.csv'*
1. Run through all steps of "Setting up the Virtual Environment"
2. run "python -m pytest --cov" in the command line


# License
This project is covered under the **MIT License**.


# Issues
Report a bug by [opening a new issue](https://github.com/sarahwaity/ProteiML/issues).
