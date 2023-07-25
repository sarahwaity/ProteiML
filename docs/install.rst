Install
=======

Currently, ``ProteiML`` is not pip installable, it may become a feature with future releases, but currently, the best way to access the tool is through our GitHub Repository: 
ProteiML @ GitHub <https://github.com/sarahwaity/ProteiML/>

Install from Github
-------------------
*GitHub Installation and setup is very fast but requires some python proficiency

To install from Github, run the following from the top-level source directory
using the Terminal::

    $ git clone https://github.com/sarahwaity/ProteiML
    $ cd ProteiML
    $ python3 setup.py install




Setting up the development environment
--------------------------------------
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


Running ProteiML
-----------------
*Test your setup using the 'streamlined_property_matrix.csv' to make sure no errors will cause run to fail!*
1. Run through all steps of "Setting up the Virtual Environment"
2. Edit the Input_Data.csv to include information about your run (more information can be found in docs/)
3. Update data_processing/constants.py to indicate whether to use the 'streamlined_property_matrix.csv' or the 'full_property_matrix.csv'
    - reccomended to test it first using the streamlined, if everything runs correctly, delete outputs and run with the full property matrix!
4. run "python main.py" in the command line
    - for 'streamlined_property_matrix.csv', this should take 5 minutes or less
    - for 'full_property_matrix.csv', this should take ~6 hours. 


Running the test suite locally
------------------------------
*To perform testing quickly, it is recommended to use 'streamlined_property_matrix.csv'*
1. Run through all steps of "Setting up the Virtual Environment"
2. run "python -m pytest --cov" in the command line



- Running Demo Data:



Python package dependencies
---------------------------
``ProteiML`` mainly depends on the Python scientific stack. 
*A full list of dependencies can be found in the pyproject.toml file.*

- [numpy](http://www.numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/)


Hardware requirements
---------------------
``ProteiML`` requires only a standard computer with enough RAM to support the in-memory operations, however the speed of computation can be faster CPUs.


OS Requirements
---------------
This package is supported for *macOS* and *Windows*.


Testing
-------
ProteiML uses the Python ``pytest`` testing package.  If you don't already have
that package installed, follow the directions on the `pytest homepage
<https://docs.pytest.org/en/latest/>`_.