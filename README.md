# Reverse Monte Carlo-Combination Fitting (RMC-CF)
RMC-CF is a tool for analyzing XAFS spectrum of heterogeneous sample based on Reverse Monte Carlo(RMC).

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview
As a supplimentally method of Linear Combination Fitting (LCF), 

RMC-CF can be applied on XAFS spectrum of heterogeneous sample, which are regarded as being distorted and cannot be handled by LCF.

In RMC-CF, a model contains $N$ blocks and individual compositions of components $J$, which has $J×N$ paramters in total, is adopted for simulation.

The spectrum is calculated from model via the following equation:

$$M_{cal}=log \frac{I_0}{∑_N \omega_n I_1^n }=-log∑_N \omega_n exp{[-∑_J{α_j^n μ_j }]}, n∈N, j∈J$$

In **nornal** mode, weights $\omega_n$ are fixed at 1.

In **weighted** model, $\omega_n$ are fitted automatically by least squares during RMC process.

 
## features
RMC-CF supports experimental spectrum data with 9809 format or simple (# energy-mu) format. 

Default reference spectra of Cu/Fe/Co are placed on `ref` folder classificated by elements. 

For more reference spectra of other elements, creating and placing them on the corresponding `element-named` folders in `ref` folder.
Any other reference spectrum files can be modified and added in the main GUI.

Support the analysis of simultated spectrum calcuated from reference spectrum according to customized heterogeneous model structure.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Lizsinta/wRMC-CF.git
cd wRMC-CF
```
### 2. Create and activate virtual environment
Windows
```bash
python -m venv venv
venv\Scripts\activate
```
Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
Install all required packages from requirements.txt:
```bash
pip install -r requirements.txt
```
### Run the program
```bash
python rmccf.py
```

## For Windows Users
You can directly download the compiled program from **Releases**:
1. Go to the [Latest Release](https://github.com/Lizsinta/RMC-CF/releases/latest)
2. Download `RMC-CF.exe` and `RMC-CF data.zip`
3. Extract `RMC-CF data.zip`, place the `data` and `ref` folder in the **same directory** as the executable file
4. Double-click `RMC-CF.exe` to run the program
