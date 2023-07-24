# ProteiML
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/{ProteiML}.svg)](https://zenodo.org/badge/latestdoi/{ProteiML})

Machine learning pipeline that underlies the findings found in: https://doi.org/10.1101/2023.04.13.536801



#[![Build Status]()

###  Abstract
Real-time monitoring of biological activity can be achieved through the use of genetically encoded fluorescent indicators (GEFIs). GEFIs are protein-based sensing tools whose biophysical characteristics can be engineered to meet experimental needs. However, GEFIs are inherently complex proteins with multiple dynamic states, rendering optimization one of the most challenging problems in protein engineering. Most GEFIs are engineered through trial-and-error mutagenesis, which is time and resource-intensive and often relies on empirical knowledge for each GEFI. We applied an alternative approach using machine learning to efficiently predict the outcomes of sensor mutagenesis by analyzing established libraries that link sensor sequences to functions. Using the GCaMP calcium indicator as a scaffold, we developed an ensemble of three regression models trained on experimentally derived GCaMP mutation libraries. We used the trained ensemble to perform an in silico functional screen on a library of 1423 novel, untested GCaMP variants. The mutations were predicted to significantly alter the fluorescent response, and off-rate kinetics were advanced for verification in vitro. We found that the ensemble’s predictions of novel variants’ biophysical characteristics closely replicated what we observed of the variants in vitro. As a result, we identified the novel ensemble-derived GCaMP (eGCaMP) variants, eGCaMP and eGCaMP+, that achieve both faster kinetics and larger fluorescent responses upon stimulation than previously published fast variants. Furthermore, we identified a combinatorial mutation with extraordinary dynamic range, eGCaMP2+, that outperforms the tested 6th, 7th, and 8th generation GCaMPs. These findings demonstrate the value of machine learning as a tool to facilitate the efficient prescreening of mutants for functional characteristics. By leveraging the learning capabilities of our ensemble, we were able to accelerate the identification of promising mutations and reduce the experimental burden associated with screening an entire library. Machine learning tools such as this have the potential to complement emerging high-throughput screening methodologies that generate massive datasets, which can be tedious to analyze manually. Overall, these findings have significant implications for developing new GEFIs and other protein-based tools, demonstrating the power of machine learning as an asset in protein engineering.


### CITATION
**If you use the machine learning platform, please cite our [pre-print](https://doi.org/10.1101/2023.04.13.536801)!**  

## Installation

### System requirements
Linux, Windows, and Mac OS are supported for running the platform. 

### Dependencies
Pipeline relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [numpy](http://www.numpy.org/) (>=1.16.0)
- [pandas](https://pandas.pydata.org/)

### Instructions
<img src=DOCS/workflow.png width=500 align="right" vspace = "50">
The pipeline follows the general workflow included in the image to the right. Briefly, the sequence-to-function mutation dataset is formatted such that it follows the shape of [(# variants) x (len(sequence)+ 1 primary ID column + 1 mutant performance column)]. Next, a novel point mutation library is used by reading the cleaned sequence-to-function mutation dataset and finding each residue position in which a mutation exists. After finding these residue locations, it saturates each position for all of the 20 amino acids. It then removes any redundant sequences that already exist in the known sequence-to-function mutation dataset. This data is then fed to each of three different regressor types, and each regressor is trained/optimized, and the performance of different encoding datasets is monitored. Once top-performing encoding datasets are found, models use these encoding datasets to generate predictions on the novel sequence library and the withheld test dataset. Within the same function/notebook, the results are averaged for an ensemble prediction for each novel sequence and each withheld variant in the test set. These results are cached for downstream analysis and used to determine mutations of interest.

## From main.py

