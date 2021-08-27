# MAGICC-MESMER coupling GMD paper code repository

## Citing this repository

This repository contains all the python code to reproduce the figures of the following paper:

Beusch, L., Nicholls, Z., Gudmundsson, L., Hauser, M., Meinshausen, M., and Seneviratne, S. I.: From emission scenarios to spatially resolved projections with a chain of computationally efficient emulators: MAGICC (v7.5.1) – MESMER (v0.8.1) coupling, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2021-252, in review, 2021.

If you use any code from this repository, please cite the associated paper.

## Using this repository

To run the scripts, follow their ordering:

- 1 + 2 train different MESMER configurations and create the associated emulations.
- 3 + 4 preprocess the data needed for the plotting.
- 5 - 8 carry out all the plotting.

However, be aware that this code only works if all input data is saved in the same structures as we have stored it and if you have all the required data available. Hence, it’s more likely that individual code snippets and checking out the general structure of our scripts are beneficial to you, rather than directly running the full code in this repository.

## Code versions

Paper submission release: v0.5.0

## License

Copyright (c) 2021 ETH Zurich, Lea Beusch.

The content of this repository is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 or (at your option) any later version.

The content of this repository is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with the content of this repository. If not, see https://www.gnu.org/licenses/.
