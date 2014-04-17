import time
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.cuda.basic_ops import gpu_from_host

# Theano's own convolution implementation
from theano.tensor.nnet import conv

# cuda-convnet convolution implementation
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
filter_acts_op = FilterActs(stride=1, partial_sum=1, pad=0)

# FFT-based convolution implementation
import fftconv

target_path = "speedtest_data.pkl"

num_runs = 10 # number of times each convolution is run,
# running time is averaged across these runs.

atol = 1e-3
rtol = 1e-5
std = 0.1

shapes_list = [
    # (input_shape, filter_shape)
    # ((minibatch_size, num_input_channels, image_width, image_height),
    #  (num_filters, num_input_channels, filter_width, filter_height))
    ((64, 128, 32, 32), (64, 128, 8, 8)),
    ((128, 32, 54, 54), (64, 32, 6, 6)),
    ((128, 128, 16, 16), (128, 128, 8, 8)),
    ((64, 3, 96, 96), (128, 3, 16, 16)),
    ((128, 1024, 32, 32), (128, 1024, 4, 4)),
]


x = theano.shared(np.zeros((1,1,1,1), dtype=theano.config.floatX))
w = theano.shared(np.zeros((1,1,1,1), dtype=theano.config.floatX))

# for cuda-convnet
x_cc = theano.shared(np.zeros((1,1,1,1), dtype=theano.config.floatX))
w_cc = theano.shared(np.zeros((1,1,1,1), dtype=theano.config.floatX))
    


def estimate_running_time(func):
    start_time = time.time()
    for _ in xrange(num_runs):
        func()
    duration = time.time() - start_time
    return duration / float(num_runs)


results = {}


for shape_x, shape_w in shapes_list:
    print
    print "X: %s" % str(shape_x)
    print "W: %s" % str(shape_w)
    print

    x_val = np.random.randn(*shape_x).astype(theano.config.floatX) * std
    w_val = np.random.randn(*shape_w).astype(theano.config.floatX) * std

    x.set_value(x_val)
    w.set_value(w_val)
    x_cc.set_value(x_val.transpose(1, 2, 3, 0)) # cuda-convnet expects the batch size in the trailing dimension.
    w_cc.set_value(w_val[:, :, ::-1, ::-1].transpose(1, 2, 3, 0)) # cuda-convnet doesn't flip the filters,
    # trailing dimension should be number of output channels.
    # by doing these transformations in advance on the host, these differences
    # cannot affect running times of the convolutions themselves.

    y_theano = conv.conv2d(x, w, image_shape=shape_x, filter_shape=shape_w)
    y_cc = filter_acts_op(x_cc, w_cc)
    y_fft = fftconv.conv2d_fft(x, w, image_shape=shape_x, filter_shape=shape_w)

    print "  compiling: Theano"
    f_theano = theano.function([], gpu_from_host(y_theano)) # don't transfer to host

    print "  compiling: cuda-convnet"
    f_cc = theano.function([], y_cc) # y_cc is already on the GPU

    print "  compiling: FFT"
    f_fft = theano.function([], gpu_from_host(y_fft)) # don't transfer to host

    print

    print "  verifying accuracy"
    # wrapping the function output in np.array causes a transfer to the host.
    out_theano = np.array(f_theano())
    out_cc = np.array(f_cc())
    out_fft = np.array(f_fft())

    assert np.allclose(out_theano, out_cc.transpose(3, 0, 1, 2), atol=atol, rtol=rtol)
    assert np.allclose(out_theano, out_fft, atol=atol, rtol=rtol)

    print

    print "  running time: Theano\t\t",
    t_theano = estimate_running_time(f_theano)
    print "%.5f s" % t_theano

    print "  running time: cuda-convnet\t",
    t_cc = estimate_running_time(f_cc)
    print "%.5f s" % t_cc

    print "  running time: FFT\t\t",
    t_fft = estimate_running_time(f_fft)
    print "%.5f s" % t_fft

    print

    results_run = {
        'theano': t_theano,
        'cuda-convnet': t_cc,
        'fft': t_fft,
    }

    results[(shape_x, shape_w)] = results_run

    # memory cleanup
    del f_theano
    del f_cc
    del f_fft



print "Storing results in %s" % target_path
with open(target_path, 'w') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)