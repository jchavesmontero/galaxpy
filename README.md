Jonas Chaves-Montero

Argonne National Laboratory

August 2019

INTRODUCTION
------------
GALAXPY is a python based stellar population synthesis model. It combine GALAXev, a code for evolving single stellar populations over time (Bruzual and Charlot 2003), an emission line model (Gutkin et al. 2016), and realistic SFHs from UniverseMachine (Behroozi et al. 2019) and IllustrisTNG (Nelson et al. 2018).

INPUT
-----
Output redshifts, SFH, dust model parameters (two), emission line properties (up to five parameters), metallicity, and IMF.

OUTPUT
------
The output of the code is a synthetic galaxy SED. If a set of filter transmission curves are specified, observed magnitudes in these bands are also computed.

EXAMPLES
--------
Some examples of how to run galaxpy are located in the notebooks folder.

IMPORTANT CONSIDERATIONS
------------------------
The permissions of some GALAXev files may have changed after downloading the code. To fix it, run chmod_galaxev.sh on a terminal.

