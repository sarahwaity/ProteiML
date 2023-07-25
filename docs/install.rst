Install
=======

Currently, ``ProteiML`` is not pip installable, it may become a feature with future releases, but currently, the best way to access the tool is through our `GitHub Repository <https://github.com/sarahwaity/ProteiML>`_

Install from Github
-------------------
*GitHub Installation and setup is very fast but requires some python proficiency*
To install from Github, run the following from the top-level source directory
using the Terminal::

    $ git clone https://github.com/sarahwaity/ProteiML
    $ cd ProteiML

Setting up the development environment
--------------------------------------
Within a new terminal instance, navigate to the ProteiML directory and run the following commands in the command line::

     $ pip install poetry
     $ poetry install
     $ poetry shell
**Virtual environment is ready!**


Running ProteiML
-----------------
*Test your setup using the 'streamlined_property_matrix.csv' to make sure no errors will cause run to fail!*
Within a new terminal instance, navigate to the ProteiML directory and run the following commands in the command line::

     $ pip install poetry
     $ poetry install
     $ poetry shell

- Edit the Input_Data.csv to include information about your run (more information can be found in docs/)
- Update data_processing/constants.py to indicate whether to use the 'streamlined_property_matrix.csv' or the 'full_property_matrix.csv'. We reccomended testing it first using the streamlined property matrix, if everything runs correctly, delete outputs and run with the full property matrix!

To start the code run::

    $ python main.py

- for 'streamlined_property_matrix.csv', this should take 5 minutes or less
- for 'full_property_matrix.csv', this should take ~6 hours. 



Running the test suite locally
------------------------------
*To perform testing quickly, it is recommended to use 'streamlined_property_matrix.csv'*
Within a new terminal instance, navigate to the ProteiML directory
Run the following commands in the command line::

     $ pip install poetry
     $ poetry install
     $ poetry shell

- Edit the Input_Data.csv to include information about your run (more information can be found in docs/)

- Update data_processing/constants.py to have AAINDEX_DB = "streamlined_property_matrix.csv". 

To start the tests run::

    $ python -m pytest --cov

- Passing and test coverage result should take ~5 minutes to generate. 



- Running Demo Data:



Python package dependencies
---------------------------
``ProteiML`` mainly depends on the Python scientific stack. 
*A full list of dependencies can be found in the pyproject.toml file.*

- numpy
- pandas
- SciPy
- scikit-learn
- seaborn


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