# Binary distribution in the Grassmann formalism
Implementation of binary distribution in the Grassmann formalism, including conditional version.
The Grassmann formalism was introduced in [1].

See the pdf file for more explanations. 

## File structure
- grassmann_distribution/:
  - GrassmannDistribution: definition of Grassmann (gr) as well as mixture of Grassmann (mogr) distribution
  - fit_grassmann: corresponding functions to estimate a gr / mogr (moment matching as well as MLE)
  - conditional_grassmann: implments a conditiona mogr in the same spirit as a MDN
- notebooks/: some example notebooks how to define the distributions and an example to fit a mogr to dichotomized gauss data
- data: samples for a dichotomized gauss distribution





## References

[1] Arai, T. (2021). Multivariate binary probability distribution in the Grassmann formalism. Physical Review E, 103(6), 062104.
