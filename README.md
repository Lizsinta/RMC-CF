# Reverse Monte Carlo-Combination Fitting (RMC-CF)
RMC-CF is a tool for analyzing XAFS spectrum of heterogeneous sample based on Reverse Monte Carlo(RMC), as a supplimentally method of Linear Combination Fitting(LCF).

[![Python 3.12+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview
RMC-CF supports experimental spectrum data with 9809 format or simple (# energy-mu) format. 
Default reference spectra of Cu/Fe/Co are placed on /ref folder classificated by elements. 
For more reference spectra of other elements, creating and placing them on the corresponding element-named folder.
Refernce spectra can be modified and added in the GUI.

Support the analysis of simultated spectrum calcuated from reference spectrum according to customized heterogeneous model structure.
*Preliminary version without parameters transmission. Users are required to switch simulation target (experimental data/model structure) and model parameters in main.py.
