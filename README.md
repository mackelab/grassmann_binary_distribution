# Binary distribution in the Grassmann formalism
Implementation of binary distribution in the Grassmann formalism, including conditional version.
The Grassmann formalism was introduced in [1].

See the pdf file for more explanations.

## File structure
- grassmann_distribution/:
  - GrassmannDistribution: definition of Grassmann (gr) as well as mixture of Grassmann (mogr) distribution
  - fit_grassmann: corresponding functions to estimate a gr / mogr (moment matching as well as MLE)
  - conditional_grassmann: implements a conditional mogr in the same spirit as a MDN for a MoGauss
- notebooks/: some example notebooks how to define the distributions and an example to fit a mogr to dichotomized gauss data
- data: samples for a dichotomized gauss distribution, see [2] for details. 





## References

[1] Arai, T. (2021). Multivariate binary probability distribution in the Grassmann formalism. Physical Review E, 103(6), 062104.

[2] Macke, J. H., et al. (2009). Generating spike trains with specified correlation coefficients. Neural computation 21.2: 397-423.
