# Supplementary materials
This folder containts the supplementary materials for the article Fatela, J., Heimes, A., and Vorländer, M. (2025). “Acoustic radiosity simulation with custom bidirectional reflectance distribution implementation in 3-D.", Journal of the Acoustical Society of America, **?**, 2025, **?**, **pp. **?**.

Getting Started
===============
These materials contain a set of jupyter notebooks which document the step-by-step simulations used to achieve the results of the application examples described in the paper. The prefixes of the notebook examples map to the relevant paper sections.

### [``V_infinite_diffuse_plane.ipynb``](V_infinite_diffuse_plane.ipynb)
This example details the simulation performed in Section V of the paper, where an "infinite" diffuse reflecting plane is simulated. The accuracy of this simulation is assessed by comparison with the results put forth by [Svensson and Savioja, 2020](https://pubs.aip.org/asa/jasa/article-abstract/156/6/3788/3324169/The-Lambert-diffuse-reflection-model-revisited?redirectedFrom=fulltext).

### [``V_infinite_diffuse_plane.ipynb``](V_infinite_diffuse_plane.ipynb)
This example details the simulation performed in Section V of the paper, where an "infinite" diffuse reflecting plane is simulated.

### A note on the simulation datasets
By default, all examples run with the datasets used in the paper. However, you can run the examples based on your own simulation results: by changing the variable ``self_data=True``, the examples will pick the results of your previously simulated data.

Note that some errors may occur, if the simulation output dependencies are not respected. If you decide to run the examples on your own results:
- [``App_BRDF_generation.ipynb``](App_BRDF_generation.ipynb) must be run before [``VIA_triangular_facade.ipynb``](VIA_triangular_facade.ipynb) and [``VIA_infinite_diffuse_plane.ipynb``](V_infinite_diffuse_plane.ipynb)

