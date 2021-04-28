# octapy
## Ocean Connectivity and Tracking Algorithms

This package is pre-release and currently under maintenance.

## Introduction

## Installation
Installation on the below systems require installing GDAL, then setting up a
virtual python environment to install the Python depenencies.

### Ubuntu

Install GDAL and GEOS, reproduced from here:
https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html
and here http://www.sarasafavi.com/installing-gdalogr-on-ubuntu.html

```
sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get install gdal-bin libgeos-dev
```


### macOS

Homebrew page with installation instructions, https://brew.sh

```
brew install gdal
brew install geos
python3 -m venv octvenv
source octvenv/bin/activate
pip3 install -r requirements.txt
pip3 install cartopy # see below
```

Note, due to cartopy not being PEP-517 compliant (as of March 2021, but see
https://github.com/SciTools/cartopy/pull/1681), it is left out of the
requirements.txt, and has to be installed separately.

This creates a python environment you can activate anytime with

`source /path/to/octenv/bin/activate`

## Usage and Examples
Coming Soon
