# Introduction to seismo-acoustic waves in the Earth’s spheres, CEG, TU Delft
Course by Läslo Evers, Shahar Shani-Kadmiel, Pieter Smets

## When and where

- **Date:** 6, 7 & 8 March 2019
- **Time:** 13.30-17.30hr
- **Place:** Faculty of Civil Engineering & Geosciences (building 23)
- **6 March (Wednesday):** room 02.110
- **7 March (Thursday):** room 2.62
- **8 March (Friday):** room 03.270

# [Download the notebooks](http://tinyurl.com/y4aehjc5) http://tinyurl.com/y4aehjc5

## Geting ready

The majority of the course will be interactive with a lot of hands-on
demonstrations and exercises purposed to enrich the learning experience.
We will do this by using [Jupyter Notebooks](https://jupyter.org/), which
you should install (or already have) on your laptops that you bring to the course.

We have prepared some instructions for setting up your environment so that you come
prepared and so that we don’t spend precious time on these preparations during the course.
Please follow the instructions at your earliest convenience and send us an email if you run
into any difficulties that you are not able to solve yourself. These instructions work
on Linux and macOS but should work for Windows machines just as well.

---
**Step 1:**

If you do not already have Anaconda or Miniconda installed on your machine (Hint: If you are unsure, go ahead and install a fresh copy), follow this link to download Miniconda and install the right package for your OS: 

https://conda.io/en/latest/miniconda.html

We recommend the 64-bit Python 3.7 version regardless of your OS.

Follow the Installation instructions (https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

---
**Step 2:**

Whether you just installed Miniconda or are using an already existing conda environment, create a new environment by typing the following command in you terminal ('Anaconda Prompt' on Windows):

```shell
conda create -n ctg -c conda-forge -y python=3.7
```

And activate this new environment with:

```shell
conda activate ctg
```

---
**Step 3:**

Now install the packages needed to run the course notebooks with the following command:

```shell
conda config --add channels conda-forge && conda install -y ipython \
    tornado=5.1.1 jupyter notebook ipywidgets numpy scipy numba pandas \
    gdal netcdf4 matplotlib basemap basemap-data-hires pillow obspy
```

---

*See you Wednesday,*

Läslo, Pieter, and Shahar
