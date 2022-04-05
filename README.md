# EPANET TO CSV

This set of tools convert and elaborates outputs from EPANET 2.2. They can firstly convert output files to CSV readable file format. It currently is in (very) experimental development and might be unstable. Use at your own risk.

## Requirements

All functionalities require Python 3.8 to execute.

## Structure

The main tool is `py_epanet2csv.py` which takes an output analysis file from EPANET 2.2 and (optionally) its `.inp` file in order to extract node coordinates and link start-end nodes.

Other tools are `epanet_map_extractor.py` which extracts a subsection from the analysis in order to analyze specific parts of the network and `csv_to_plot` which plots specifics values from the CSV converted file.

**!TODO!**
