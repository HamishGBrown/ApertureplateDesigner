# Description

A script for generating condenser aperture designs for transmission electron microscopy (TEM)

![](Demonstration.png)

# System Requirements

Standard desktop computer

[Anaconda Python](https://www.anaconda.com/download) or [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

Python packages:
[numpy](https://numpy.org/)
[matplotlib](https://matplotlib.org/)
[ezdxf](https://pypi.org/project/ezdxf/)

Tested on ezdxf v 4.1.1, numpy v 1.21.5 and matplotlib v 3.5.1 and python v 3.9.12 though it should be compatible with earlier versions of each package


# Installation Guide

Should take 15 minutes with a modern desktop and fast internet connection

Install Anaconda python or miniconda using the graphical installers

Install numpy,matplotlib and ezdxf, within an commandshell with access to anaconda python execute the following command:

`pip install numpy matplotlib ezdxf`

# Demo

Run the script in a python interpreter:
`$ python Generate_apertures.py`

The script should take less than 5 seconds.

A folder "Apertures" will be created which contains AutoCAD dxf files of the apertures and pdf and png versions of the apertures.

Instructions
------------
To generate your own custom aperture plate designs modify the following lines of `Generate_apertures.py`:

```
# Put rotations in degrees here
rotations = np.asarray([0,20,40,60,80]*3)
# Put sizes of aperture plates (in microns) here
sizes = np.asarray([20,40,60]*3)
# For rectangles you need to put both dimensions as a list eg:
sizes = np.asarray([[20,30]]*5+[[40,50]]*5+[[60,70]]*5)
```



