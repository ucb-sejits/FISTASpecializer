"""
specializer FISTASpecializer
"""

from ctree.jit import LazySpecializedFunction


# importations
import numpy as np
import pycl as cl
import ast
import ctypes as ct

# special functions
from ctree.types import get_c_type_from_numpy_dtype


'''
	Parameters
    ----------
    A : array like (numpy.ndarray, numpy.matrix, LinearOperator)
        2D array representing the linear operator. Note: does not have to
        be square

    b : array like (numpy.ndarray, numpy.matrix, LinearOperator)
        Vector of the received data (the result from the forward
        transformation).

    pL : function R^n -> R^n
        See the notes section.

    residual_diff : float, optional
        Specifies when to stop looping, (when the change in the residuals is
        less than this value).

    max_iter : int, optional
        The maximum number of iterations in the algorithm. If the residual
        difference is not yet below the specified threshold, then the
        solution at the last iteration is returned.

    monotone : bool, optional
        Whether to use the monotone version of the gradient descent or not
        (the residuals are monotonic).

    F : function R^n -> R
        The objective function to be minimized. The monotone flag requires
        this parameter, which is by default None. Therefore, in order to
        use the monotone version, this parameter also must be explicitly
        specified.

        A common function is :math:`F = x \mapsto \|A \cdot x - b\|^2`.


    Returns
    -------
    x* : numpy.ndarray
        The optimal solution to within a certain accuracy.

    residual : numpy.ndarray of float
        The residuals accrued during the looping. This also contains a
        record of how many iterations were performed (its length).

'''


class FISTAConcrete(ConcreteSpecializedFunction):
	def __init__(self):
		# TODO: OpenCL context and queues go here
		raise NotImplementedError()

	# Not sure if we actually have to implement this
	def finalize(self):
		# TODO: Compilation goes here
		raise NotImplementedError()

	def __call__(self, b, pL, residual_diff=0.0001, max_iter=1000, monoton=False, F=None):
		# TODO: create your output buffer(s) here
		# TODO: call the generated C function here

		### Preprocessing steps in python ###

		# Turn (N,1) arrays into vectors.
		b = np.squeeze(b)
		

	    # Initial guess is empty
	    y = zeros(A.shape[1])
	    x1 = y

	    t1 = 1

    	# Initialize residual using empty guess
    	# TODO: this is probably something we want to specialize eventually
    	residual = [norm(A.dot(x1) - b) / norm(b)]

    	i = 0
    	while True:

	        # Calculate the next optimal solution.

	        z = pL(y) 	# TODO: Call the C version
	        i += 1

	        # Change ratio of how much of the last two solutions to mix. The
	        # ratio decreases over time (the solution calculated from pL is
	        # more accurate).
	        t2 = 0.5 * (1 + sqrt(1 + 4 * t1**2))

	        if monotone:
	            x2 = min((z, x1), key=F)
	        else:
	            x2 = z

	        # Used to reduce the number of calculations from the algorithms
	        # given in the gradient chapter. Use 'is' for constant time lookup.
	        if x2 is x1:
	            mix = (t1 / t2) * (z - x2)
	        else:  # x2 is z
	            mix = ((t1 - 1) / t2) * (x2 - x1)

	        # Create next mixed iteration vector
	        y = x2 + mix

	        # Calculate the norm residuals of this new result
	        # TODO: this is probably something we want to specialize eventually
	        residual.append(norm(A.dot(x2) - b) / norm(b))

	        # Break here since we don't need a new y if the change in residual
	        # is below the desired cutoff.
	        if (residual[-2] - residual[-1]) < residual_diff or i >= max_iter:
	            break

	        # Set up for next iteration.
	        x1 = x2
	        t1 = t2

	    # z is returned because it is the last calculated value that adheres
	    # to any constraints inside pL (positivity).
	    return z, residual


class FISTA(LazySpecializedFunction):

    def transform(self, tree, program_config):

    	# Extract the data type of the arguments and returned data
    	args_data = program_config[0]
    	data_type = get_c_type_from_numpy_dtype(A.dtype)

        # Finding the primary kernel
        # TODO: this might return multiple if there are many functions passed in (e.g. the F function in FISTA)
		pL = PyBasicConversions().visit(tree).find(FunctionDecl) 
		pL.name = "pL"

		apply_one.params[0].type = data_type()
		pL.return_type = data_type()

		# TODO: set pL return type
		# TODO: set pL param types (should be one parameter with type ndarray)

		# FIXME: this is where the codegeneration goes.

		# def pL(y):
		# 	z = y - alpha*(A.T*(A*y - b))
	 	# 	z[z<0] = 0
	 	# 	return z


        # Adding this in later
        # F = PyBasicConversions().visit(tree.body[1])


if __name__ == '__main__':
    pass

