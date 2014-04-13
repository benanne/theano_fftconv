theano_fftconv
==============

Convolution op for Theano based on CuFFT using scikits.cuda

This is an experiment in implementing an FFT-based convolution op for Theano, using scikits.cuda. It was inspired by this paper, which shows promising speedups for FFT-based convolutions in the Torch7 framework: http://openreview.net/document/aa6ab717-ca19-47e1-a958-823b9a106ca9

Currently this is barely functional and not very fast. Input is welcome!

scikits.cuda: https://github.com/lebedov/scikits.cuda

## Current status

The implementation gives the same result as a valid convolution using Theano's own conv2d. With the implementation of a ComplexDotOp, it seems to surpass Theano's own conv2d in speed in some scenarios. 

On my laptop with a GT 540M (96 cores) the speedup is quite significant. On a workstation with a GTX680 (1536 cores) the improvement is less pronounced, but it's still 2x faster in many cases. This implies that we're not parallelising enough yet. Parallelising the complex dot product (using the lowlevel CUBLAS API with multiple streams, and later using cublasCgemmBatched) is the next step.
