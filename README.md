# py_epanet

Py_EPANET2CSV is a Python-based project that started as a repository for hosting simple EPANET scripts. However, it has evolved into a much richer project that includes data analysis and machine learning analysis using TensorFlow and neural networks.

EPANET is a software tool used for modeling water distribution systems. It is widely used by engineers and researchers in the field of water resources engineering.
Purpose of the Project

The purpose of this project is to provide a Python-based interface to the EPANET software. The main goal of the project is to extract data from EPANET simulations and perform various types of analyses on the data.

## Generate a dataset

The main script of the Py_EPANET2CSV project is `py_epanet.py`. This script does not provide a GUI or a CLI but it is easily configurable from the main function of the source code itself. Most of the parameters are clearly self-explanatory.

To use py_epanet.py, simply indicate the source map and EPANET (or more precisely WNTR) will generate a CSV from that simulation. You can enable leakages, pressure dependent simulation, and more by modifying the parameters in the main() function.

# Process a dataset

