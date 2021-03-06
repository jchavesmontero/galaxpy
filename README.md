Jonas Chaves-Montero

Argonne National Laboratory

August 2019

INTRODUCTION
------------

Galaxpy is an open-source python-based stellar population synthesis code that produces precise galaxy SEDs using Galaxev as baseline Stellar Population Synthesis code (Bruzual and Charlot 2003), accounts for attenuation by dust following the prescriptions outlined in Charlot & Fall (2000), and models nebular emission lines as Gutkin et al. (2016).

INPUT
-----
Galaxpy generates full spectral energy distributions (SEDs) for arbitrarily complex star formation histories specified by the user, at a set of custom redshifts. It includes 5 free parameters controlling the impact of different galaxy properties on the SED: initial mass function, metallicity, dust attenuation, and nebular emission line rations.

OUTPUT
------
For each star formation history, Galaxpy provides synthetic SEDs at a set of redshifts specified by the user. If a set of filter transmission curves are also specified, Galaxpy provides observed magnitudes in the input filters at the redshifts of interest.

EXAMPLES
--------
Some examples of how to run Galaxpy can be found in the notebooks folder.

IMPORTANT CONSIDERATIONS
------------------------
The permissions of some GALAXev files may have changed after downloading the code. To fix it, run chmod_galaxev.sh on a terminal. Notebooks not rendering correctly in github can be seen using nbviewer (https://nbviewer.jupyter.org/).
