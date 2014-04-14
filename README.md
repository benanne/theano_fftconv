theano_fftconv
==============

Convolution op for Theano based on CuFFT using scikits.cuda

This is an experiment in implementing an FFT-based convolution op for Theano, using scikits.cuda. It was inspired by this paper, which shows promising speedups for FFT-based convolutions in the Torch7 framework: http://openreview.net/document/aa6ab717-ca19-47e1-a958-823b9a106ca9

Currently this is barely functional and not very fast. Input is welcome!

scikits.cuda: https://github.com/lebedov/scikits.cuda

## Current status

The implementation gives the same result as a valid convolution using Theano's own conv2d. With the implementation of a BatchedComplexDot op, which executes many complex dot products in multiple streams, the performance seems to be quite good (several times faster than Theano's own conv2d in many cases).

An alternative way to parallelise the batched complex dot product is to use cublasCgemmBatched. This is the next thing to try.
