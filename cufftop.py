import numpy as np 
import theano

import theano.sandbox.cuda as cuda
from theano.misc.pycuda_utils import to_gpuarray, to_cudandarray

import scikits.cuda
from scikits.cuda import fft
from scikits.cuda import linalg
from scikits.cuda import cublas


import pycuda.gpuarray
import pycuda.driver

import theano.misc.pycuda_init

import string

linalg.init()


# TODO: implement __eq__ and __hash__ correctly
# TODO: Find out if scikits.cuda.fft.fft is destructive - if so we need to specify a destroy_map

# TODO: investigate FFTW compatibility modes. Can probably set this to the fastest setting.
# TODO: investigate the effect of enabling fastmath on FFT performance.


class CuFFTOpBase(cuda.GpuOp): # base class for shared code between FFT and IFFT
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def output_type(self, inp):
        raise NotImplementedError

    def make_node(self, inp):
        inp = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp))

        assert inp.dtype == "float32"

        return theano.Apply(self, [inp], [self.output_type(inp)()])



class CuFFTOp(CuFFTOpBase):
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim + 1)) # add one extra dim for real/imag

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [ storage_map[v] for v in node.inputs]
        outputs = [ storage_map[v] for v in node.outputs]

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            # construct output shape
            output_shape = list(input_shape)
            output_shape[-1] = output_shape[-1] // 2 + 1 # DFT of real input is symmetric, no need to store redundant coefficients
            output_shape += [2] # extra dimension with length 2 for real/imag
            output_shape = tuple(output_shape)

            z = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = cuda.CudaNdarray.zeros(output_shape)

            input_pycuda = to_gpuarray(inputs[0][0])
            # I thought we'd need to change the type on output_pycuda so it is complex64,
            # but as it turns out scikits.cuda.fft doesn't really care either way and
            # treats the array as if it is complex64 anyway.
            output_pycuda = to_gpuarray(z[0])

            # only initialise plan if necessary
            if plan[0] is None or plan_input_shape[0] != input_shape:
                plan_input_shape[0] = input_shape
                plan[0] = fft.Plan(input_shape[1:], np.float32, np.complex64, batch=input_shape[0])

            fft.fft(input_pycuda, output_pycuda, plan[0])


        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk



class CuIFFTOp(CuFFTOpBase):
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim - 1)) # remove extra real/imag dim

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [ storage_map[v] for v in node.inputs]
        outputs = [ storage_map[v] for v in node.outputs]

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            # construct output shape
            output_shape = list(input_shape[:-1]) # chop off the extra length-2 dimension for real/imag
            output_shape[-1] = (output_shape[-1] - 1) * 2 # restore full signal length
            output_shape = tuple(output_shape)

            z = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = cuda.CudaNdarray.zeros(output_shape)

            input_pycuda = to_gpuarray(inputs[0][0])
            # input_pycuda is a float32 array with an extra dimension, but will be
            # interpreted by scikits.cuda as a complex64 array instead.
            output_pycuda = to_gpuarray(z[0])

            # only initialise plan if necessary
            if plan[0] is None or plan_input_shape[0] != input_shape:
                plan_input_shape[0] = input_shape
                plan[0] = fft.Plan(output_shape[1:], np.complex64, np.float32, batch=output_shape[0])
                # need to chop off the extra dimension for real/imag here as well.

            fft.ifft(input_pycuda, output_pycuda, plan[0]) # , True)
            # strangely enough, enabling rescaling here makes it run very, very slowly.
            # so do this rescaling manually afterwards!


        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk





def to_complex_gpuarray(x, copyif=False):
    """
    adapted version of theano.misc.pycuda_utils.to_gpuarray that takes an array with an extra trailing
    dimension of length 2 for real/imaginary parts, and turns it into a complex64 PyCUDA GPUArray.
    """
    if not isinstance(x, cuda.CudaNdarray):
        raise ValueError("We can transfer only CudaNdarray to pycuda.gpuarray.GPUArray")
    else:
        # Check if trailing dimension has length 2
        assert x.shape[-1] == 2

        # check if dtype is float32
        assert x.dtype == 'float32'

        # Check if it is c contiguous
        size = 1
        c_contiguous = True
        for i in range(x.ndim-1, -1, -1):
            if x.shape[i] == 1:
                continue
            if x._strides[i] != size:
                c_contiguous = False
                break
            size *= x.shape[i]
        if not c_contiguous:
            if copyif:
                x = x.copy()
            else:
                raise ValueError("We were asked to not copy memory, but the memory is not c contiguous.")

        # Now x is always c contiguous
        px = pycuda.gpuarray.GPUArray(x.shape[:-1], np.complex64, base=x, gpudata=x.gpudata)
        return px

def to_complex_cudandarray(x):
    """
    adapted version of theano.misc.pycuda_utils.to_cudandarray that takes a complex64 array
    and turns it into a float32 CudaNdarray with an extra trailing dimension of length 2
    for real/imaginary parts.
    """
    if not isinstance(x, pycuda.gpuarray.GPUArray):
        raise ValueError("We can transfer only pycuda.gpuarray.GPUArray to CudaNdarray")
    elif x.dtype != "complex64":
        raise ValueError("Only conversion from complex64 arrays is supported")
    else:
        # TODO: figure out what is going on here and adapt it for the complex64-float32 case.
        strides = [1, 2]
        for i in x.shape[::-1][:-1]:
            strides.append(strides[-1]*i)
        strides = tuple(strides[::-1])
        shape = tuple(list(x.shape) + [2])
        ptr = int(x.gpudata) # in pycuda trunk, y.ptr also works, which is a little cleaner
        z = cuda.from_gpu_pointer(ptr, shape, strides, x)

        return z



class ComplexDotOp(CuFFTOpBase):
    def make_node(self, inp1, inp2):
        inp1 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp1))
        inp2 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 3
        assert inp2.dnim == 3

        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * inp.type.ndim) # add one extra dim for real/imag

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [ storage_map[v] for v in node.inputs]
        outputs = [ storage_map[v] for v in node.outputs]

        def thunk():
            x = inputs[0]
            y = inputs[1]

            # chop off the real/imag dimension
            input_shape_x = x[0].shape # (a, b, 2)
            input_shape_y = y[0].shape # (b, c, 2)

            output_shape = (input_shape_x[0], input_shape_y[1], 2) # (a, c, 2)

            input_x_pycuda = to_complex_gpuarray(x[0])
            input_y_pycuda = to_complex_gpuarray(y[0])

            output_pycuda = linalg.dot(input_x_pycuda, input_y_pycuda)

            outputs[0][0] = to_complex_cudandarray(output_pycuda)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk




class MultiStreamComplexDotOp(CuFFTOpBase):
    def make_node(self, inp1, inp2):
        inp1 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp1))
        inp2 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 3
        assert inp2.ndim == 3

        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [ storage_map[v] for v in node.inputs]
        outputs = [ storage_map[v] for v in node.outputs]

        num_streams = 32 # 32

        handle = [cublas.cublasCreate()]
        stream_pool = [pycuda.driver.Stream() for _ in xrange(num_streams)]
        current_stream = [0]

        def thunk():
            x = inputs[0]
            y = inputs[1]

            # chop off the real/imag dimension
            input_shape_x = x[0].shape # (a, b, 2)
            input_shape_y = y[0].shape # (b, c, 2)

            output_shape = (input_shape_x[0], input_shape_y[1], 2) # (a, c, 2)

            input_x_pycuda = to_complex_gpuarray(x[0])
            input_y_pycuda = to_complex_gpuarray(y[0])

            # multistream experiment
            # print "DEBUG: Setting stream to %d" % current_stream[0]

            # prev_stream_obj = stream_pool[(current_stream[0] - 1) % num_streams]
            # print "PREV STREAM IS DONE?"
            # print prev_stream_obj.is_done()
            # print

            stream_obj = stream_pool[current_stream[0]]
            cublas.cublasSetStream(handle[0], stream_obj.handle)
            current_stream[0] += 1
            current_stream[0] %= num_streams
            # print "DEBUG: set next stream id to %d" % current_stream[0]

            output_pycuda = linalg.dot(input_x_pycuda, input_y_pycuda, handle=handle[0])

            outputs[0][0] = to_complex_cudandarray(output_pycuda)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk






def sc_complex_dot(x_gpu, y_gpu, c_gpu, transa='N', transb='N', handle=None):
    """
    modified version of linalg.dot which allows for the target output array to be specified.
    This function does not return anything.
    """
    if handle is None:
        handle = scikits.cuda.misc._global_cublas_handle

    assert len(x_gpu.shape) == 2
    assert len(y_gpu.shape) == 2
    assert len(c_gpu.shape) == 2
    assert x_gpu.dtype == np.complex64
    assert y_gpu.dtype == np.complex64 
    assert c_gpu.dtype == np.complex64

    # Get the shapes of the arguments
    x_shape = x_gpu.shape
    y_shape = y_gpu.shape
    
    # Perform matrix multiplication for 2D arrays:
    alpha = np.complex64(1.0)
    beta = np.complex64(0.0)
    
    transa = string.lower(transa)
    transb = string.lower(transb)

    if transb in ['t', 'c']:
        m, k = y_shape
    elif transb in ['n']:
        k, m = y_shape
    else:
        raise ValueError('invalid value for transb')

    if transa in ['t', 'c']:
        l, n = x_shape
    elif transa in ['n']:
        n, l = x_shape
    else:
        raise ValueError('invalid value for transa')

    if l != k:
        raise ValueError('objects are not aligned')

    if transb == 'n':
        lda = max(1, m)
    else:
        lda = max(1, k)

    if transa == 'n':
        ldb = max(1, k)
    else:
        ldb = max(1, n)

    ldc = max(1, m)

    cublas.cublasCgemm(handle, transb, transa, m, n, k, alpha, y_gpu.gpudata,
                lda, x_gpu.gpudata, ldb, beta, c_gpu.gpudata, ldc)



def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.

    taken from scikits.cuda tests/test_cublas.py
    """
    
    return pycuda.gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)



def sc_complex_dot_batched(bx_gpu, by_gpu, bc_gpu, transa='N', transb='N', handle=None):
    """
    uses cublasCgemmBatched to compute a bunch of complex dot products in parallel
    """
    if handle is None:
        handle = scikits.cuda.misc._global_cublas_handle

    assert len(bx_gpu.shape) == 3
    assert len(by_gpu.shape) == 3
    assert len(bc_gpu.shape) == 3
    assert bx_gpu.dtype == np.complex64
    assert by_gpu.dtype == np.complex64 
    assert bc_gpu.dtype == np.complex64

    # Get the shapes of the arguments
    bx_shape = bx_gpu.shape
    by_shape = by_gpu.shape
    
    # Perform matrix multiplication for 2D arrays:
    alpha = np.complex64(1.0)
    beta = np.complex64(0.0)
    
    transa = string.lower(transa)
    transb = string.lower(transb)

    if transb in ['t', 'c']:
        N, m, k = by_shape
    elif transb in ['n']:
        N, k, m = by_shape
    else:
        raise ValueError('invalid value for transb')

    if transa in ['t', 'c']:
        N2, l, n = bx_shape
    elif transa in ['n']:
        N2, n, l = bx_shape
    else:
        raise ValueError('invalid value for transa')

    if l != k:
        raise ValueError('objects are not aligned')

    if N != N2:
        raise ValueError('batch sizes are not the same')

    if transb == 'n':
        lda = max(1, m)
    else:
        lda = max(1, k)

    if transa == 'n':
        ldb = max(1, k)
    else:
        ldb = max(1, n)

    ldc = max(1, m)

    # construct pointer arrays needed for cublasCgemmBatched
    bx_arr = bptrs(bx_gpu)
    by_arr = bptrs(by_gpu)
    bc_arr = bptrs(bc_gpu)

    cublas.cublasCgemmBatched(handle, transb, transa, m, n, k, alpha, by_arr.gpudata,
                lda, bx_arr.gpudata, ldb, beta, bc_arr.gpudata, ldc, N)






class BatchedComplexDotOp(CuFFTOpBase):
    def make_node(self, inp1, inp2):
        inp1 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp1))
        inp2 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 4 # (batch, a, b, real/imag)
        assert inp2.ndim == 4

        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [ storage_map[v] for v in node.inputs]
        outputs = [ storage_map[v] for v in node.outputs]

        def thunk():
            bx = inputs[0]
            by = inputs[1]

            input_shape_x = bx[0].shape # (batch, a, b, 2)
            input_shape_y = by[0].shape # (batch, b, c, 2)

            output_shape = (input_shape_x[0], input_shape_x[1], input_shape_y[2], 2) # (batch, a, c, 2)

            bz = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if bz[0] is None or bz[0].shape != output_shape:
                bz[0] = cuda.CudaNdarray.zeros(output_shape)

            input_bx_pycuda = to_complex_gpuarray(bx[0])
            input_by_pycuda = to_complex_gpuarray(by[0])
            output_b_pycuda = to_complex_gpuarray(bz[0])

            # we want to write the results to one big contiguous array, so we can't
            # use linalg.dot here (it creates a new array and returns it)

            for i in xrange(input_shape_x[0]): # batch iter
                input_x_pycuda = input_bx_pycuda[i]
                input_y_pycuda = input_by_pycuda[i]
                output_pycuda = output_b_pycuda[i]

                sc_complex_dot(input_x_pycuda, input_y_pycuda, output_pycuda)

  
        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk



class NativeBatchedComplexDotOp(CuFFTOpBase):
    """
    This version uses cublasCgemmBatched under the hood, instead of 
    doing multiple cublasCgemm calls.
    """
    def make_node(self, inp1, inp2):
        inp1 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp1))
        inp2 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 4 # (batch, a, b, real/imag)
        assert inp2.ndim == 4

        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [ storage_map[v] for v in node.inputs]
        outputs = [ storage_map[v] for v in node.outputs]

        def thunk():
            bx = inputs[0]
            by = inputs[1]

            input_shape_x = bx[0].shape # (batch, a, b, 2)
            input_shape_y = by[0].shape # (batch, b, c, 2)

            output_shape = (input_shape_x[0], input_shape_x[1], input_shape_y[2], 2) # (batch, a, c, 2)

            bz = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if bz[0] is None or bz[0].shape != output_shape:
                bz[0] = cuda.CudaNdarray.zeros(output_shape)

            input_bx_pycuda = to_complex_gpuarray(bx[0])
            input_by_pycuda = to_complex_gpuarray(by[0])
            output_b_pycuda = to_complex_gpuarray(bz[0])

            # fancy native batched version
            sc_complex_dot_batched(input_bx_pycuda, input_by_pycuda, output_b_pycuda)



  
        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk














cufft = CuFFTOp()
cuifft = CuIFFTOp()
# complex_dot = ComplexDotOp()
complex_dot = MultiStreamComplexDotOp()
batched_complex_dot = BatchedComplexDotOp()
native_batched_complex_dot = NativeBatchedComplexDotOp()









def complex_elemwise_mult(x, y, no_concatenate=False):
    """
    This function computes the elemwise product of two arrays x and y,
    assuming that the last dimension is length 2 and represents the
    real and imaginary parts of the complex numbers.

    This is not the same as just x * y!

    no_concatenate: enable to return two separate tensors, one for the real part and one for the imaginary part.
    concatenation is expensive!
    """
    # can't do y[..., ::-1] in theano
    index_flip = [slice(None) for _ in xrange(y.ndim - 1)]
    index_flip += [slice(None, None, -1)]
    index_flip = tuple(index_flip)

    index_0 = [slice(None) for _ in xrange(y.ndim - 1)]
    index_0 += [0]
    index_0 = tuple(index_0)

    index_1 = [slice(None) for _ in xrange(y.ndim - 1)]
    index_1 += [1]
    index_1 = tuple(index_1)

    cis = x * y # for the real part
    trans = x * y[index_flip] # for the imaginary part - need to flip real and imag on y.

    real_part = cis[index_0] - cis[index_1]
    imag_part = trans[index_0] + trans[index_1]

    if no_concatenate:
        return real_part, imag_part
    else:
        return T.concatenate([T.shape_padright(real_part), T.shape_padright(imag_part)], axis=(y.ndim - 1))


def mult_and_reduce_basic(input_fft_u, filters_fft_u):
    # elementwise product (broadcasting among b and oc dimensions)
    output_fft_u = complex_elemwise_mult(input_fft_u, filters_fft_u) # (b, oc, ic, i0, i1//2 + 1, 2)

    # sum over the input channels
    output_fft_s = output_fft_u.sum(axis=2) # (b, oc, i0, i1//2 + 1, 2)

    return output_fft_s


def mult_and_reduce_late_concatenation(input_fft_u, filters_fft_u):
    """
    This version reduces across the ic dimension before concatenation, to reduce the amount of data that needs to be copied.
    """
    output_fft_u_real, output_fft_u_imag = complex_elemwise_mult(input_fft_u, filters_fft_u, no_concatenate=True)
    real_part = output_fft_u_real.sum(axis=2)
    imag_part = output_fft_u_imag.sum(axis=2)

    return T.concatenate([T.shape_padright(real_part), T.shape_padright(imag_part)], axis=real_part.ndim)






def _flip_last_dim(x):
    """
    Helper function because Theano does not support the ... operator

    This flips the last dimension of the input tensor.
    """
    index_flip = [slice(None) for _ in xrange(x.ndim - 1)]
    index_flip += [slice(None, None, -1)]
    index_flip = tuple(index_flip)
    return x[index_flip]

def _index_last_dim(x, i):
    """
    Helper function because Theano does not support the ... operator

    This indexes the last dimension of the input tensor with i.
    """
    index_i = [slice(None) for _ in xrange(x.ndim - 1)]
    index_i += [i]
    index_i = tuple(index_i)
    return x[index_i]


def _batched_dot_part(input_fft_v, filters_fft_v):    
    """
    input_fft_v is (b, ic, i0, i1//2 + 1, 2)
    filters_fft_v is (oc, ic, i0, i1//2 + 1, 2)
    """
    b, ic, i0, i1_f, _ = input_fft_v.shape
    oc = filters_fft_v.shape[0]

    # reshape to flatten the dimensions that are multiplied elementwise
    input_r = input_fft_v.reshape((b, ic, i0 * i1_f * 2))
    filters_r = filters_fft_v.reshape((oc, ic, i0 * i1_f * 2))

    # shuffle for batched_dot
    input_s = input_r.dimshuffle(2, 0, 1)
    filters_s = filters_r.dimshuffle(2, 1, 0)

    output_s = T.batched_dot(input_s, filters_s) # (i0 * i1_f * 2, b, oc)

    # shuffle again
    output_r = output_s.dimshuffle(1, 2, 0)

    # reshape to unflatten
    output = output_r.reshape((b, oc, i0, i1_f, 2))

    return output

def mult_and_reduce_batched_dot(input_fft_v, filters_fft_v):
    """
    IMPORTANT: this requires input where the b and oc axes HAVE NOT BEEN SEPARATED.

    This version uses theano.tensor.batched_dot to do the multiplication and reduction in one go.
    If b, ic and oc are large enough, this should be fast - but it does two dot products for each
    pixel in the input image! That might be painful.

    input_fft_v is (b, ic, i0, i1//2 + 1, 2)
    filters_fft_v is (oc, ic, i0, i1//2 + 1, 2)
    """

    cis = _batched_dot_part(input_fft_v, filters_fft_v)
    trans = _batched_dot_part(input_fft_v, _flip_last_dim(filters_fft_v))

    real_part = _index_last_dim(cis, 0) - _index_last_dim(cis, 1)
    imag_part = _index_last_dim(trans, 0) + _index_last_dim(trans, 1)

    return T.concatenate([T.shape_padright(real_part), T.shape_padright(imag_part)], axis=real_part.ndim)


def mult_and_reduce_scan(input_fft_u, filters_fft_u):
    """
    This version uses scan across the ic dimension to accumulate all the parts.
    """

    b, _, ic, i0, i1_f, _ = input_fft_u.shape
    oc = filters_fft_u.shape[1]

    # input_fft_u is     (b, 1, ic, i0, i1//2 + 1, 2)
    # filterS_fft_u is   (1, oc, ic, i0, i1//2 + 1, 2)

    input_fft_icfirst = input_fft_u.dimshuffle(2, 0, 1, 3, 4, 5)
    filters_fft_icfirst = filters_fft_u.dimshuffle(2, 0, 1, 3, 4, 5)

    def fn(input_part, filters_part, prev):
        prod = complex_elemwise_mult(input_part, filters_part)
        return prev + prod

    outputs, updates = theano.scan(fn=fn,
            outputs_info=T.zeros((b, oc, i0, i1_f, 2)),
            sequences=[input_fft_icfirst, filters_fft_icfirst]) # , profile=True)

    assert len(updates) == 0

    return outputs[-1] # (b, oc, i0, i1//2 + 1, 2)



def mult_and_reduce_scan_late_concat(input_fft_u, filters_fft_u):
    """
    This version uses scan across the ic dimension to accumulate all the parts.
    """

    b, _, ic, i0, i1_f, _ = input_fft_u.shape
    oc = filters_fft_u.shape[1]

    # input_fft_u is     (b, 1, ic, i0, i1//2 + 1, 2)
    # filterS_fft_u is   (1, oc, ic, i0, i1//2 + 1, 2)

    input_fft_icfirst = input_fft_u.dimshuffle(2, 0, 1, 3, 4, 5)
    filters_fft_icfirst = filters_fft_u.dimshuffle(2, 0, 1, 3, 4, 5)

    def fn(input_part, filters_part, prev_real, prev_imag):
        prod_real, prod_imag = complex_elemwise_mult(input_part, filters_part, no_concatenate=True)
        return prev_real + prod_real, prev_imag + prod_imag

    (outputs_real, outputs_imag), updates = theano.scan(fn=fn,
            outputs_info=[T.zeros((b, oc, i0, i1_f)), T.zeros((b, oc, i0, i1_f))],
            sequences=[input_fft_icfirst, filters_fft_icfirst]) # , profile=True)

    assert len(updates) == 0

    real_part = outputs_real[-1]
    imag_part = outputs_imag[-1]

    return T.concatenate([T.shape_padright(real_part), T.shape_padright(imag_part)], axis=real_part.ndim)
    # (b, oc, i0, i1//2 + 1, 2)



def mult_and_reduce_batched_complex_dot(input_fft_v, filters_fft_v):
    """
    IMPORTANT: this requires input where the b and oc axes HAVE NOT BEEN SEPARATED.

    This version uses a custom ComplexDot op together with scan.

    input_fft_v is (b, ic, i0, i1//2 + 1, 2)
    filters_fft_v is (oc, ic, i0, i1//2 + 1, 2)
    """

    b, ic, i0, i1_f, _ = input_fft_v.shape 
    oc = filters_fft_v.shape[0]

    # reshape to flatten the dimensions that are multiplied elemwise
    input_r = input_fft_v.reshape((b, ic, i0 * i1_f, 2))
    filters_r = filters_fft_v.reshape((oc, ic, i0 * i1_f, 2))

    # shuffle for batched dot product
    input_s = input_r.dimshuffle(2, 0, 1, 3) # (i0 * i1_f, b, ic, 2)
    filters_s = filters_r.dimshuffle(2, 1, 0, 3) # (i0 * i1_f, ic, oc, 2)

    def fn(input_part, filters_part):
        return complex_dot(input_part, filters_part)

    output_s, updates = theano.scan(fn=fn,
        outputs_info=None,
        sequences=[input_s, filters_s],
        non_sequences=None)
    # output_s is (i0 * i1_f, b, oc, 2)

    assert len(updates) == 0

    # shuffle again
    output_r = output_s.dimshuffle(1, 2, 0, 3)

    # reshape to unflatten
    output = output_r.reshape((b, oc, i0, i1_f, 2))

    return output



def mult_and_reduce_standalone_batched_complex_dot(input_fft_v, filters_fft_v, input_shape=None, filter_shape=None):
    """
    IMPORTANT: this requires input where the b and oc axes HAVE NOT BEEN SEPARATED.

    This version uses a custom BatchedComplexDot op (no scan) and multiple streams.

    input_fft_v is (b, ic, i0, i1//2 + 1, 2)
    filters_fft_v is (oc, ic, i0, i1//2 + 1, 2)
    """

    if input_shape is None:
        input_shape =  input_fft_v.shape # symbolic

    if filter_shape is None:
        filter_shape = filters_fft_v.shape # symbolic

    b, ic, i0, i1_f, _ = input_shape
    oc = filter_shape[0]

    # reshape to flatten the dimensions that are multiplied elemwise
    input_r = input_fft_v.reshape((b, ic, i0 * i1_f, 2))
    filters_r = filters_fft_v.reshape((oc, ic, i0 * i1_f, 2))

    # shuffle for batched dot product
    input_s = input_r.dimshuffle(2, 0, 1, 3) # (i0 * i1_f, b, ic, 2)
    filters_s = filters_r.dimshuffle(2, 1, 0, 3) # (i0 * i1_f, ic, oc, 2)

    # output_s = batched_complex_dot(input_s, filters_s)
    output_s = native_batched_complex_dot(input_s, filters_s)

    # shuffle again
    output_r = output_s.dimshuffle(1, 2, 0, 3)

    # reshape to unflatten
    output = output_r.reshape((b, oc, i0, i1_f, 2))

    return output













# mult_and_reduce = mult_and_reduce_basic
# mult_and_reduce = mult_and_reduce_late_concatenation
# mult_and_reduce = mult_and_reduce_batched_dot


def conv2d_fft(input, filters, image_shape=None, filter_shape=None):
    """
    expects bc01 input
    performs a valid convolution

    input: (b, ic, i0, i1)
    filters: (oc, ic, f0, f1)
    """

    # use symbolic shapes to compute shape info at runtime if not specified
    if image_shape is None:
        image_shape = input.shape

    if filter_shape is None:
        filter_shape = filters.shape

    b, ic, i0, i1 = image_shape # batch size, input channels, input dim 0, input dim 1
    oc, ic_, f0, f1 = filter_shape # output channels, input channels, filter dim 0, filter dim 1

    # assert ic == ic_ # same number of input channels
    # assert f0 <= i0 # filter fits within input
    # assert f1 <= i1 # filter fits within input

    # pad filters to input shape
    filters_padded = T.zeros((oc, ic, i0, i1))
    filters_padded = T.set_subtensor(filters_padded[:, :, :f0, :f1], filters)

    # reshape for FFT
    input_flat = input.reshape((b * ic, i0, i1))
    filters_flat = filters_padded.reshape((oc * ic, i0, i1))

    # perform FFT
    input_fft_flat = cufft(input_flat) # (b * ic, i0, i1//2 + 1, 2)
    filters_fft_flat = cufft(filters_flat) # (oc * ic, i0, i1//2 + 1, 2)

    # unfold ic dimension, separate b and oc
    input_fft_u = input_fft_flat.reshape((b, 1, ic, i0, i1//2 + 1, 2))
    filters_fft_u = filters_fft_flat.reshape((1, oc, ic, i0, i1//2 + 1, 2))

    # without separate b and oc
    input_fft_v_shape = (b, ic, i0, i1//2 + 1, 2)
    filters_fft_v_shape = (oc, ic, i0, i1//2 + 1, 2)
    input_fft_v = input_fft_flat.reshape(input_fft_v_shape)
    filters_fft_v = filters_fft_flat.reshape(filters_fft_v_shape)

    # elementwise product (broadcasting among b and oc dimensions) + sum along ic axis
    # output_fft_s = mult_and_reduce_late_concatenation(input_fft_u, filters_fft_u) # (b, oc, i0, i1//2 + 1, 2)
    # output_fft_s = mult_and_reduce_batched_dot(input_fft_v, filters_fft_v) # (b, oc, i0, i1//2 + 1, 2)
    # output_fft_s = mult_and_reduce_scan(input_fft_u, filters_fft_u)
    # output_fft_s = mult_and_reduce_scan_late_concat(input_fft_u, filters_fft_u)
    # output_fft_s = mult_and_reduce_batched_complex_dot(input_fft_v, filters_fft_v) # (b, oc, i0, i1//2 + 1, 2)
    output_fft_s = mult_and_reduce_standalone_batched_complex_dot(input_fft_v, filters_fft_v,
                            input_shape=input_fft_v_shape, filter_shape=filters_fft_v.shape) # (b, oc, i0, i1//2 + 1, 2)

    # reshape for IFFT
    output_fft_flat = output_fft_s.reshape((b * oc, i0, i1//2 + 1, 2))

    # perform IFFT
    output_flat = cuifft(output_fft_flat) # (b * oc, i0, i1)
    
    # reshape
    output_circ = output_flat.reshape((b, oc, i0, i1)) # circular!

    # slice because the convolution was circular, we need it to be valid
    output = output_circ[:, :, f0 - 1:, f1 - 1:]

    # rescale manually
    output = (1.0 / T.cast(i0 * i1, theano.config.floatX)) * output # allow for the scale factor to move to the gpu

    # output should now be the result of a batched valid convolution of the input with the filters.
    return output





if __name__ == '__main__':

    # ### Basic CuFFTOp functionality test

    # import time
    # import theano.tensor as T
    # from theano.sandbox.cuda.basic_ops import host_from_gpu


    # x = T.tensor3('x')

    # dbl = host_from_gpu(CuFFTOp()(x))
    # # dbl = CuFFTOp()(x)

    # f = theano.function([x], dbl)

    # a = np.random.randn(256, 512, 512).astype('float32')

    # print "GPU"
    # start_time = time.time()
    # b = f(a)
    # print "%.4f" % (time.time() - start_time)

    # print "GPU2"
    # start_time = time.time()
    # b = f(a)
    # print "%.4f" % (time.time() - start_time)

    # print "GPU3"
    # start_time = time.time()
    # b = f(a)
    # print "%.4f" % (time.time() - start_time)


    # b_complex = b[..., 0] + 1j * b[..., 1]

    # print "CPU"
    # start_time = time.time()
    # b_verify = np.fft.rfftn(a, axes=(1,2))
    # print "%.4f" % (time.time() - start_time)

    # # print b_complex - b_verify

    # # print "allclose:"
    # # print np.allclose(b_verify, b_complex)


    # ### Test inverse FFT

    # import time
    # import theano.tensor as T
    # from theano.sandbox.cuda.basic_ops import host_from_gpu


    # x = T.tensor3('x')

    # x_f = CuFFTOp()(x)
    # x_if = CuIFFTOp()(x_f)
    # out = host_from_gpu(x_if)


    # f = theano.function([x], out)

    # a = np.random.randn(256, 512, 512).astype('float32')


    # print "GPU"
    # start_time = time.time()
    # b = f(a)
    # print "%.4f" % (time.time() - start_time)

    # print "allclose:"
    # print np.allclose(a, b, atol=1e-5, rtol=1e-3)


    ### Test complex elemwise multiplication

    # import time
    # import theano.tensor as T
    # from theano.sandbox.cuda.basic_ops import host_from_gpu

    # a_real = np.random.randn(32, 64).astype('float32')
    # a_imag = np.random.randn(32, 64).astype('float32')
    # a_complex = a_real + a_imag * 1j

    # a_stack = np.concatenate([a_real[..., None], a_imag[..., None]], axis=-1)

    # b_real = np.random.randn(32, 64).astype('float32')
    # b_imag = np.random.randn(32, 64).astype('float32')
    # b_complex = b_real + b_imag * 1j

    # b_stack = np.concatenate([b_real[..., None], b_imag[..., None]], axis=-1)

    # c_complex = a_complex * b_complex


    # x = T.tensor3('x')
    # y = T.tensor3('y')

    # z = complex_elemwise_mult(x, y)

    # f = theano.function([x, y], z)

    # c_stack = f(a_stack, b_stack)

    # c_complex2 = c_stack[..., 0] + c_stack[..., 1] * 1j


    ### Test conv2d

    import time
    import theano.tensor as T
    from theano.sandbox.cuda.basic_ops import host_from_gpu
    from theano.tensor.nnet import conv

    x_shape = (64, 128, 32, 32)
    w_shape = (64, 128, 8, 8)

    # x_shape = (128, 32, 54, 54)
    # w_shape = (64, 32, 6, 6)

    # x_shape = (128, 128, 16, 16)
    # w_shape = (128, 128, 8, 8)

    # x_shape = (64, 3, 128, 128)
    # w_shape = (128, 3, 16, 16)

    # x_shape = (128, 1024, 32, 32)
    # w_shape = (128, 1024, 4, 4)


    x = theano.shared(np.random.randn(*x_shape).astype('float32'))
    w = theano.shared(np.random.randn(*w_shape).astype('float32'))

    y = conv.conv2d(x, w)

    y_fft = conv2d_fft(x, w, image_shape=x_shape, filter_shape=w_shape)

    print "compiling default theano conv"
    f = theano.function([], y)

    print "compiling fft conv"
    f_fft = theano.function([], y_fft)

    # print "running default theano conv"
    # start_time = time.time()
    # out = f()
    # print "%.5f s" % (time.time() - start_time)

    # print "running default theano conv (2)"
    # start_time = time.time()
    # out = f()
    # print "%.5f s" % (time.time() - start_time)

    # print "running fft conv"
    # start_time = time.time()
    # out_fft = f_fft()
    # print "%.5f s" % (time.time() - start_time)

    # print "running fft conv (2)"
    # start_time = time.time()
    # out_fft = f_fft()
    # print "%.5f s" % (time.time() - start_time)

    # print "running fft conv (3)"
    # start_time = time.time()
    # out_fft = f_fft()
    # print "%.5f s" % (time.time() - start_time)

    start_time = time.time()
    for k in xrange(10):
        f_fft()
    print "took %.5f seconds" % (time.time() - start_time)






