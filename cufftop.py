import numpy as np 
import theano

import theano.misc.pycuda_init
import theano.sandbox.cuda as cuda
from theano.misc.pycuda_utils import to_gpuarray, to_cudandarray

from scikits.cuda import fft


# TODO: implement __eq__ and __hash__ correctly
# TODO: Find out if scikits.cuda.fft.fft is destructive - if so we need to specify a destroy_map
# TODO: pycuda might provide a faster way to do elementwise multiplication of complex arrays.


# TODO: the elementwise product gets too big very quickly. fix this by doing the product + summing in batches.

# TODO: investigate FFTW compatibility modes. Can probably set this to the fastest setting.
# TODO: investigate the effect of enabling fastmath on FFT performance.


# TODO: implement a ComplexElemwiseMultOp, this might be a lot quicker than the current approach.
# it can also be made descructive (destroying its second input) which is nice.



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



cufft = CuFFTOp()
cuifft = CuIFFTOp()


def complex_elemwise_mult(x, y):
    """
    This function computes the elemwise product of two arrays x and y,
    assuming that the last dimension is length 2 and represents the
    real and imaginary parts of the complex numbers.

    This is not the same as just x * y!
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

    return T.concatenate([T.shape_padright(real_part), T.shape_padright(imag_part)], axis=(y.ndim - 1))




def conv2d_fft(input, filters):
    """
    expects bc01 input
    performs a valid convolution

    input: (b, ic, i0, i1)
    filters: (oc, ic, f0, f1)
    """

    input_shape = input.shape
    filter_shape = filters.shape

    b, ic, i0, i1 = input_shape # batch size, input channels, input dim 0, input dim 1
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

    # elementwise product (broadcasting among b and oc dimensions)
    output_fft_u = complex_elemwise_mult(input_fft_u, filters_fft_u) # (b, oc, ic, i0, i1//2 + 1, 2)

    # sum over the input channels
    output_fft_s = output_fft_u.sum(axis=2) # (b, oc, i0, i1//2 + 1, 2)

    # reshape for IFFT
    output_fft_flat = output_fft_s.reshape((b * oc, i0, i1//2 + 1, 2))

    # perform IFFT
    output_flat = cuifft(output_fft_flat) # (b * oc, i0, i1)

    # rescale manually
    output_rescaled = output_flat / (i0 * i1) 
    
    # reshape
    output_circ = output_rescaled.reshape((b, oc, i0, i1)) # circular!

    # slice because the convolution was circular, we need it to be valid
    output = output_circ[:, :, f0 - 1:, f1 - 1:]

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

    x = theano.shared(np.random.randn(32, 64, 16, 16).astype('float32'))
    w = theano.shared(np.random.randn(64, 64, 8, 8).astype('float32'))

    y = conv.conv2d(x, w)

    y_fft = conv2d_fft(x, w)

    print "compiling default theano conv"
    f = theano.function([], y)

    print "compiling fft conv"
    f_fft = theano.function([], y_fft)

    # print "running default theano conv"
    # start_time = time.time()
    # out = f()
    # print "%.5f s" % (time.time() - start_time)

    # print "running fft conv"
    # start_time = time.time()
    # out_fft = f_fft()
    # print "%.5f s" % (time.time() - start_time)

    for k in xrange(10):
        f_fft()









# class PycudaElemwiseSourceModuleMakeThunkOp(Op):
#     nin = property(lambda self: self.scalar_op.nin)
#     nout = property(lambda self: self.scalar_op.nout)

#     def __init__(self, scalar_op, inplace_pattern=None, name=None):
#         if inplace_pattern is None:
#             inplace_pattern = {}
#         self.name = name
#         self.scalar_op = scalar_op
#         self.inplace_pattern = inplace_pattern

#     def __str__(self):
#         if self.name is None:
#             if self.inplace_pattern:
#                 items = self.inplace_pattern.items()
#                 items.sort()
#                 return self.__class__.__name__ + "{%s}%s" % (self.scalar_op,
#                                                              str(items))
#             else:
#                 return self.__class__.__name__ + "{%s}" % (self.scalar_op)
#         else:
#             return self.name

#     def make_node(self, *inputs):
#         assert self.nout == 1
#         assert len(inputs) == 2  # TODO remove
#         _inputs = [gpu_contiguous(as_cuda_ndarray_variable(i)) for i in inputs]
#         if self.nin > 0 and len(_inputs) != self.nin:
#             raise TypeError('Wrong argument count', (self.nin, len(_inputs)))
#         for i in _inputs[1:]:
#             if i.type.ndim != inputs[0].type.ndim:
#                 raise TypeError('different ranks among inputs')

#         if any([any(i.type.broadcastable) for i in inputs]):
#             raise Exception("pycuda don't support broadcasted dimensions")

#         otype = CudaNdarrayType(broadcastable=[False] * _inputs[0].type.ndim)
#         out_node = Apply(self, _inputs, [otype() for o in xrange(self.nout)])
#         return out_node

#     def make_thunk(self, node, storage_map, _, _2):
#         #TODO support broadcast!
#         #TODO assert all input have the same shape
#         fct_name = "pycuda_elemwise_%s" % str(self.scalar_op)
#         in_name = ["i" + str(id) for id in range(len(node.inputs))]
#         out_name = ["o" + str(id) for id in range(self.nout)]

#         c_code = self.scalar_op.c_code(node, "some_name",
#                                        tuple([n + "[i]" for n in in_name]),
#                                        tuple(n + "[i]" for n in out_name), {})
#         c_code_param = ", ".join([_replace_npy_types(var.type.dtype_specs()[1]) + " *" + name
#                                   for var, name in
#                                   zip(node.inputs, in_name) +
#                                   zip(node.outputs, out_name)] + ["int size"])
#         mod = SourceModule("""
#   __global__ void %s(%s)
#   {
#     int i = (blockIdx.x+blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y);
#     i += threadIdx.x + threadIdx.y*blockDim.x;
#     if(i<size){
#         %s
#     }
#   }
#   """ % (fct_name, c_code_param, c_code))
#         pycuda_fct = mod.get_function(fct_name)
#         inputs = [storage_map[v] for v in node.inputs]
#         outputs = [storage_map[v] for v in node.outputs]

#         def thunk():
#             z = outputs[0]
#             if (z[0] is None or
#                 z[0].shape != inputs[0][0].shape or
#                 not z[0].is_c_contiguous()):
#                 z[0] = theano.sandbox.cuda.CudaNdarray.zeros(
#                     inputs[0][0].shape)
#             if inputs[0][0].shape != inputs[1][0].shape:
#                 raise TypeError("PycudaElemwiseSourceModuleMakeThunkOp:"
#                                 " inputs don't have the same shape!")

#             if inputs[0][0].size > 512:
#                 grid = (int(numpy.ceil(inputs[0][0].size / 512.)), 1)
#                 block = (512, 1, 1)
#             else:
#                 grid = (1, 1)
#                 block = (inputs[0][0].shape[0], inputs[0][0].shape[1], 1)
#             out = pycuda_fct(inputs[0][0], inputs[1][0], z[0],
#                              numpy.intc(inputs[1][0].size), block=block,
#                              grid=grid)
#         thunk.inputs = inputs
#         thunk.outputs = outputs
#         thunk.lazy = False

#         return thunk













