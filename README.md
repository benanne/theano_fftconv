theano_fftconv
==============

Convolution op for Theano based on CuFFT using scikits.cuda

This is an experiment in implementing an FFT-based convolution op for Theano, using scikits.cuda. It was inspired by this paper, which shows promising speedups for FFT-based convolutions in the Torch7 framework: http://openreview.net/document/aa6ab717-ca19-47e1-a958-823b9a106ca9

Currently this is barely functional and not very fast. Input is welcome!

scikits.cuda: https://github.com/lebedov/scikits.cuda

## Current status

The implementation gives the same result as a valid convolution using Theano's own conv2d. Unfortunately it's quite slow. Interestingly, the FFTs are not the problem, those account for only ~2% of running time according to Theano's profiler.

The problem is the elementwise multiplication of the Fourier-transformed inputs with the Fourier-transformed filters. I have implemented this in a bunch of ways already (check the different mult_and_reduce_* functions), but performance is disappointing for all of them.

Using a routine that is able to work with complex numbers directly would probably speed this up a lot. scikits.cuda comes with such a routine that could be wrapped in another op, but unfortunately it does not support broadcasting, so it cannot be used directly.

Suggestions to speed up the elementwise product (and summing out the input_channels dimension) are welcome.

