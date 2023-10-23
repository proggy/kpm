# To do

- implement other reconstruction methods (fct, fft, dft)
- by choice, return cofunc objects
- implement the "stochastical method" to compute ADOS
- keep rescaling factors


## Submodule kern

- implement Wang-and-Zunger kernel.


## Submodule mom

- def trace(...)
- def expec2(...)
- def _expec_csr_real(indices, indptr, data, rank, state, limit, out=None)
- def _expec2_csr_real(indices, indptr, data, state1, state2, limit, ...)
- def _trace_csr_real(indices, indptr, data, randstates, limit, ...)


## Submodule rcstr

- implement fast cosine transform (FCT) algorithm
- create pure Python versions for all the algorithms
- implement fast Fourier transform (FFT) algorithm
- implement discrete Fourier transform (DFT) algorithm
- add VMKPM variants (given a mapping energy -> moment)
- def _fft
- def _fft_real
- def _fft_real_omp
- def _fft_complex
- def _fft_complex_omp


## Submodule resc

- support other matrix formats besides CSR (e.g. dense matrices)
- support expansions in more than one variable (multivariate functions)


## Submodule svect

- create real random-phase vectors using Gaussian distribution
- create real random-phase vectors using uniform distribution


## Submodule disc

- create Cython/OpenMP versions? Not really necessary...
- idea: option to get only part of the spectrum (some specific interval within [-1, 1])
