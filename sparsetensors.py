import numpy as np
import scipy.sparse as sp
import functools
import itertools
import numbers
import logging
import copy
from collections import Counter

class SparseTensor:
    """
    A simple implementation of Sparse n-dimensional tensors.
    """
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.coords = []
        self.values = []
        self.sorted = False
        
    def __getitem__(self, keys):
        """
        Returns smaller SparseTensor based on keys indexing.
        """
        return NotImplemented
      
    def _itemization_(self, keys, value, func):
        """
        Helper function to run a function on the specific coordinates 
        of the sparse tensor and their corresponding value. Used in
        __setitem__ and add.
        parameter func should take a coord and a number.
        """
        if 0 in self.shape:
            return
        if not hasattr(keys, '__len__'):
            keys = (keys,)
        else:
            keys = tuple(keys)
        
        def smart_key(key, size):
            if isinstance(key, slice):
                return slice(0 if key.start is None else key.start,
                             size if key.stop is None else key.stop,
                             1 if key.step is None else key.step)
            else:
                return key
        
        keys = [smart_key(key, size) for key, size in zip(keys+tuple([slice(None, None, None)]*(len(self.shape)-len(keys))), self.shape)]
        
        for i in range(len(keys)):
            if isinstance(keys[i], slice):
                assert isinstance(keys[i].start, int), "What?"
                assert isinstance(keys[i].stop, int), "What?"
                assert isinstance(keys[i].step, int), "What?"
                assert keys[i].start >= 0, "Only positive slices are accepted."
                assert keys[i].stop >= 0, "Only positive slices are accepted."
                assert keys[i].start < self.shape[i], "Slice goes beyond the tensor."
                assert keys[i].stop <= self.shape[i], "Slice goes beyond the tensor."
            else:
                assert isinstance(keys[i], int) or isinstance(keys[i], np.integer)
                assert keys[i] >= 0, "Only positive indices are accepted."
                assert keys[i] < self.shape[i], "Index goes beyond the tensor."
        
        def shape_helper(k):
            if isinstance(k, slice):
                return int((k.stop-k.start)//k.step)
            else:
                return 1
        slice_shape = tuple(functools.reduce(lambda x,y: x+[shape_helper(y)], keys, []))
        if isinstance(value, np.ndarray):# or isinstance(value, SparseTensor):
            assert slice_shape == value.shape, "Non-matching shapes: "+str(slice_shape)+" and "+str(value.shape)
        elif hasattr(value, '__len__'):
            assert functools.reduce(lambda x, y: x*y, slice_shape) == len(value), "Non-matching shapes: "+str(slice_shape)+" and "+str((len(value),))
        elif isinstance(value, SparseTensor):
            assert slice_shape==value.shape, "Non-matching shapes: "+str(slice_shape)+" and "+str(value.shape)
        else:
            assert isinstance(value, numbers.Number), "Only numpy arrays, python arrays, and numbers are allowed."
        
        def key_iter_helper(k):
            if isinstance(k, slice):
                return np.arange(k.start, k.stop, k.step)
            else:
                return [k]
        if isinstance(value, np.ndarray):
            i = 0
            key_product = list(itertools.product(*[key_iter_helper(k) for k in keys]))
            for _, v in np.ndenumerate(value):
                if v!=0:
                    coord = key_product[i]
                    func(coord, v)
                i += 1
        elif hasattr(value, '__len__'):
            i = 0
            key_product = list(itertools.product(*[key_iter_helper(k) for k in keys]))
            for _, v in enumerate(value):
                if v!=0:
                    coord = key_product[i]
                    func(coord, v)
                i += 1
        elif isinstance(value, SparseTensor):
            for c, v in zip(value.coords, value.values):
                coord = []
                for i in range(len(keys)):
                    if isinstance(keys[i], slice):
                        coord.append(keys[i].start + c[i] * keys[i].step)
                    else:
                        coord.append(keys[i])
                func(tuple(coord), v)
        else:
            for coord in itertools.product(*[key_iter_helper(k) for k in keys]):
                func(coord, value)
        
    def __setitem__(self, keys, value):
        """
        Sets values in the sparse array. Similar to numpy
        array broadcasting but far less powerful.
        """
        def set_item_function(coord, value):
            for i in range(len(coord)):
                assert coord[i]>=0
                assert coord[i]<self.shape[i]
            if coord in self.coords:
                logging.warning("Values are being overwritten!")
                self.values[self.coords.index(coord)] = value
            else:
                self.coords.append(coord)
                self.values.append(value)
        self._itemization_(keys, value, set_item_function)
        self.sorted = False
        
    def add_pivot(self, raw_pivot, other):
        """
        Helper function for extra-dimensional-projection.
        Injects other (another SparseTensor) into self
        offsetting every coordinate in other base don raw_pivot.
        """
        if 0 in self.shape:
            return
        if not hasattr(raw_pivot, '__len__'):
            pivot = (raw_pivot,)
        else:
            pivot = raw_pivot
        for i in range(len(pivot)):
            assert isinstance(pivot[i], int)
            assert pivot[i] >= 0
            assert pivot[i] < self.shape[i]
        
        assert len(self.shape)==len(other.shape)
        for i in range(len(self.shape)):
            assert self.shape[i]-pivot[i] >= other.shape[i]
        
        for i in range(len(other.coords)):
            self[tuple([pivot[j]+other.coords[i][j] for j in range(len(self.shape))])] = other.values[i]
        self.sorted = False
        
    def add(self, keys, value):
        """
        Adds values in the sparse array. Similar to numpy
        array broadcasting but far less powerful.
        """
        def add_item_function(coord, value):
            if coord in self.coords:
                self.values[self.coords.index(coord)] += value
            else:
                self.coords.append(coord)
                self.values.append(value)
        self._itemization_(keys, value, add_item_function)
        self.sorted = False
        
    def to_dense(self):
        """
        Returns the corresponding dense (np.ndarray) tensor.
        """
        dense = np.zeros(self.shape)
        for i in range(len(self.coords)):
            try:
                dense[self.coords[i]] = self.values[i]
            except:
                print(self.coords[i])
                raise IndexError(i)
        return dense

    def flattened(self, order): #depreciated
        """
        Returns the corresponding 2-D sparse tensor given the ordering
        The first dimension is always the same and all other dimensions should have equal size
        'order' param should be a function that given a coord returns a single integer for the location in the second dimension
        """
        assert callable(order), "order must be a function"
        new_tensor = SparseTensor((self.shape[0], np.prod(self.shape[1:])))
        for coord, value in zip(self.coords, self.values):
            new_tensor[coord[0], order(coord[1:])] = value
        return new_tensor

    def from_dense(self, dense, force_shape=False):
        """
        Creates a SparseTensor from a dense (np.ndarray) tensor.
        """
        logging.debug("from_dense is being called on a sparse tensor. This WILL overwrite ANY values in the tensor. Generally this function is made for testing.")
        if not force_shape:
            assert self.shape==dense.shape, "Call force shape if you want a new shape"
        else:
            self.shape = dense.shape
        self.coords = []
        self.values = []
        for keys, val in np.ndenumerate(dense):
            if val!=0:
                self[keys] = val
        self.sorted = False
        
    def add_rank(self, injection_point):
        """
        Augments this tensor with a new rank that becomes the
        injection_point(-th) rank.
        """
        new_shape = list(self.shape)
        new_shape.insert(injection_point, 1)
        new_tensor = SparseTensor(tuple(new_shape))
        for i in range(len(self.coords)):
            new_coord = list(self.coords[i])
            new_coord.insert(injection_point, 0)
            new_tensor.coords.append(tuple(new_coord))
            new_tensor.values.append(self.values[i])
        return new_tensor
    
    def __mul__(self, other):
        """
        Multiplication with a scalar.
        """
        new_tensor = SparseTensor(self.shape)
        for coord, value in zip(self.coords, self.values):
            new_tensor[coord] = value * other
        return new_tensor
    
    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        """
        Addition between two tensors of the same exact shape
        """
        assert self.shape==other.shape, "Must be same shape for addition"

        new_tensor = SparseTensor(self.shape)
        for coord, value in zip(self.coords, self.values):
            new_tensor.add(coord, value)
        for coord, value in zip(other.coords, other.values):
            new_tensor.add(coord, value)
        
        return new_tensor

    #here follows a bunch of fun methods only for 2-Ds
    def first_nonzero(self, row): #depreciated
        """
        Returns the first (left to right) nonzero coord and value in a row, 2-D only for the time being
        """
        assert len(self.shape)==2, "first_nonzero only has 2-D support"
        assert self.shape[0]>row, "Out of bounds"
        if len(self.coords) == 0:
            return -1, 0 #no nonzeros
        filt = [c[1]==row for c in self.coords]
        coords = [c for c, f in zip(self.coords, filt) if f]
        values = [c for c, f in zip(self.values, filt) if f]
        c, v = list(sorted(zip(coords, values), key = lambda x: x[1]))[0]
        return c[1], v

    def reorder_rows(self, new_rows): #depreciated
        """
        Returns a new tensor that has the rows changed based on the new_rows.
        new_rows may either be a tuple (pair of values) this indicates a flip,
        or it may be an int array the size of shape[0] for a complete reordering
        """
        if len(new_rows)==2:
            for i in range(len(self.coords)):
                if self.coords[i][0]==new_rows[0]:
                    self.coords[i][0] = -1
                if self.coords[i][0]==new_rows[1]:
                    self.coords[i][0] = -2
            for i in range(len(self.coords)):
                if self.coords[i][0]==-1:
                    self.coords[i][0] = new_rows[1]
                if self.coords[i][0]==-2:
                    self.coords[i][0] = new_rows[0]
        else:
            assert len(new_rows)==self.shape[0], "Incorrect number of values for total reordering"
            assert Counter(new_rows)==Counter(range(self.shape[0])), "Not a permutation" #we only tend to do this branch once so the check can be slow
            self.coords = [tuple([new_rows[coord[0]]]+list(coord[1:])) for coord in self.coords]
        return self
    
    def multiply_row(self, row, coeff): #depreciated
        """
        Multiplies a row by a coefficient
        """
        for coord, value in zip(self.coords, self.values):
            if coord[0]==row:
                value *= coeff
        return self

    def copy_from(self, other, offset=None):
        """
        Copies the values from another SparseTensor.
        Only use for debugging.
        """
        logging.debug("Use of copy_from is not recommended as it lacks many protections.")
        assert isinstance(other, SparseTensor), "Cannot copy from non-SparseTensor."
        if offset is None:
            offset = [0]*len(self.coords)
        for s,o,f in zip(self.shape, other.shape, offset):
            assert s>=(o+f), "Shape out of range."
        
        for coord, value in zip(other.coords, other.values):
            new_coord = tuple([0+(coord[i] if len(coord)>i else 0)+(offset[i] if len(offset)>i else 0) for i in range(len(self.shape))])
            self[new_coord] = value
        self.sorted = False

    def copy(self):
        new_tensor = SparseTensor(self.shape)
        for coord, value in zip(self.coords, self.values):
            new_tensor[coord] = value
        return new_tensor

    def __eq__(self, other):
        if not len(self.coords) == len(other.coords):
            return False
        for coord, value in zip(self.coords, self.values):
            if other[coord] != value:
                return False
        return True

    def row_stripping(self, rows):
        """
        Returns a new Sparse tensor with the specified rows (cannonically first rank) removed
        """
        new_tensor = SparseTensor([self.shape[0]-len(rows)]+list(self.shape[1:]))

        mapping = np.full(self.shape[0], -1)
        j = 0
        for i in range(self.shape[0]):
            if not i in rows:
                mapping[i] = j
                j += 1
        
        for coord, value in zip(self.coords, self.values):
            if mapping[coord[0]] != -1:
                new_tensor[tuple([mapping[coord[0]]]+list(coord[1:]))] = value

        return new_tensor
    
    def column_stripping(self, columns):
        """
        Returns a new Sparse tensor with the specified columns (cannonically second rank) removed
        """
        new_tensor = SparseTensor([self.shape[0]]+[self.shape[1]-len(columns)])

        mapping = np.full(self.shape[1], -1)
        j = 0
        for i in range(self.shape[1]):
            if not i in columns:
                mapping[i] = j
                j += 1
        
        for coord, value in zip(self.coords, self.values):
            if mapping[coord[1]] != -1:
                new_tensor[tuple([coord[0]]+[mapping[coord[1]]])] = value

        return new_tensor

    def sort(self):
        res = sorted(zip(self.coords, self.values), key = lambda x: x[0])
        raise NotImplemented
    
    def __repr__(self):
        return "Coordinates: "+str(self.coords)+"\nValues: "+str(self.values)

def extra_dimensional_projection_numpy(tensors, preserved_dimensions=[]):
    """
    Computes the extra dimensional projection for a list of numpy ndarrays.
    
    Parameters
    ----------
    tensors : list of np.ndarrays
        Arrays to compute the extra dimensional projection of.
    preserved_dimensions : list of ints
        List representing what ranks should have constant dimensionality and
        therfor be contracted.
    
    Returns
    -------
    new_tensor : np.ndarray
        The tensor created by the extra dimensional projection of the input tensors.
    """
    for t in tensors:
        assert isinstance(t, np.ndarray), "extra_dimensional_projection_numpy only takes np.ndarrays"
    perserved_dimensions_values = [tensors[0].shape[i] for i in preserved_dimensions]
    for i in range(len(preserved_dimensions)):
        for t in tensors:
            assert t.shape[preserved_dimensions[i]] == perserved_dimensions_values[i], "Misshaped dimension on extra dimensional projection "+str(perserved_dimensions_values[i])+"'s index. Check input tensor shapes."
    
    expanded_shape = []
    starting_points = [[] for i in range(len(tensors))]
    for i in range(len(tensors[0].shape)):
        if i in preserved_dimensions:
            assert tensors[0].shape[i]==perserved_dimensions_values[preserved_dimensions.index(i)]
            expanded_shape.append(tensors[0].shape[i])
            for j in range(len(tensors)):
                starting_points[j].append(0)
        else:
            s = 0
            for j in range(len(tensors)):
                starting_points[j].append(s)
                s += tensors[j].shape[i]
            expanded_shape.append(s)
    
    new_tensor = np.zeros(expanded_shape, dtype=tensors[0].dtype)
    for i in range(len(tensors)):
        indexing_mesh = np.ix_(*[np.arange(starting_points[i][j],starting_points[i][j]+tensors[i].shape[j]) for j in range(len(tensors[0].shape))])
        assert new_tensor[indexing_mesh].shape == tensors[i].shape, "new_tensor indexing is somehow off"
        new_tensor[indexing_mesh] += tensors[i]
    
    return new_tensor

def extra_dimensional_projection(tensors, preserved_dimensions=[]): 
    """
    Computes the extra dimensional projection for a list of SparseTensors.
    
    Parameters
    ----------
    tensors : list of SparseTensors
        Arrays to compute the extra dimensional projection of.
    preserved_dimensions : list of ints
        List representing what ranks should have constant dimensionality and
        therfor be contracted.
    
    Returns
    -------
    new_tensor : SparseTensor
        The tensor created by the extra dimensional projection of the input tensors.
    """
    for t in tensors:
        assert isinstance(t, SparseTensor), type(t)
    perserved_dimensions_values = [tensors[0].shape[i] for i in preserved_dimensions]
    for i in range(len(preserved_dimensions)):
        for t in tensors:
            assert t.shape[preserved_dimensions[i]] == perserved_dimensions_values[i], "Misshaped dimension on extra dimensional projection "+str(perserved_dimensions_values[i])+"'s index. Check input tensor shapes."
    
    expanded_shape = []
    starting_points = [[] for i in range(len(tensors))]
    for i in range(len(tensors[0].shape)):
        if i in preserved_dimensions:
            assert tensors[0].shape[i]==perserved_dimensions_values[preserved_dimensions.index(i)]
            expanded_shape.append(tensors[0].shape[i])
            for j in range(len(tensors)):
                starting_points[j].append(0)
        else:
            s = 0
            for j in range(len(tensors)):
                starting_points[j].append(s)
                s += tensors[j].shape[i]
            expanded_shape.append(s)

    new_tensor = SparseTensor(expanded_shape)
    for i in range(len(tensors)):
        new_tensor.add_pivot(tuple(starting_points[i]), tensors[i])
    
    return new_tensor


def einstein_summation(subscripts, *operands, **kwargs):
    """
    Generalizer of 'np.einsum'. Will call 'np.einsum' with inputs if
    there are no SparseTensor operands. Otherwise will run a 
    sparse einstein summation via 'einstein_summation_sparse'
    
    See np.einsum and einstein_summation_sparse for details on kwargs.
    """
    check_einsum_inputs(subscripts, *operands)
    sparse_huh = False
    for op in operands:
        if isinstance(op, SparseTensor):
            sparse_huh = True
    if len(operands)==0:
        raise ValueError("Todo: remove me")
        return operands[0]
    if sparse_huh:
        #logging.debug("Running a sparse einstein summation.")
        return einstein_summation_sparse(subscripts, *operands, **kwargs)
    else:
        #logging.debug("Running a dense einstein summation.")
        return np.einsum(subscripts, *operands, **kwargs)

def check_einsum_inputs(subscripts, *operands):
    """
    Verifies reasonablness of subscripts and operands .shape(s)
    for completing an einstein_summation
    """
    if '->' in subscripts:
        for op in operands:
            assert not isinstance(op, SparseTensor)
    else:
        op_subs = subscripts.split(",")
        assert len(op_subs)==len(operands), "Incorrect subscript count given."
        for sub, op in zip(op_subs, operands):
            assert len(sub)==len(op.shape), "Incorrected length subscript given."

def einsum_output_subscript(input_subscripts):
    """
    Caculates the expected einstein summation output subscripts
    following notational standards.
    """
    output_subscript = []
    for sub in input_subscripts:
        for l in sub:
            if l in output_subscript:
                output_subscript.remove(l)
            else:
                output_subscript.append(l)
    return "".join(output_subscript)

def einsum_output_shape(op_subs, *operands):
    """
    Caculates the expected einstein summation output shape.
    Use dict comprehension?
    """
    lengths = {}
    for subs, op in zip(op_subs, operands):
        for l, op_shape in zip(subs, op.shape):
            if l in lengths.keys():
                assert lengths[l] == op_shape, "Dimensions don't match between similarly subscripted indicies."
            else:
                lengths.update({l: op_shape})
    
    return tuple([lengths[l] for l in einsum_output_subscript(op_subs)])

def sparse_match(op_subs, *coords):
    """
    Calculates if a tuple of sparse tensors match at coordinates.
    If they don't match returns None, otherwise return the 
    bindings to constituting the match.
    """
    match = {}
    for subs, coord in zip(op_subs, coords):
        for l, v in zip(subs, coord):
            if l in match.keys():
                if v!=match[l]:
                    return None
            else:
                match.update({l: v})
    return match
    
def einstein_summation_sparse(subscripts, *operands, stay_sparse=False):
    """
    Calculates a einstein summation containing atleast one SparseTensor.
    Similar operation to 'np.einsum'. 'stay_sparse' keyword argument
    indicates if function should return a sparse or a dense tensor.
    """
    op_subs = subscripts.split(",")
    sparse_operands = [(sub, op) for sub, op in zip(op_subs, operands) if isinstance(op, SparseTensor)]
    dense_operands = [(sub, op) for sub, op in zip(op_subs, operands) if isinstance(op, np.ndarray)]
    output_shape = einsum_output_shape(op_subs, *operands)
    if len(output_shape)>0:
        output = SparseTensor(output_shape)
        output_subscript = einsum_output_subscript(op_subs)
    else:
        output = 0
    for prod in itertools.product(*[zip(op[1].coords, op[1].values) for op in sparse_operands]): #absolutely cursed line, product containing a point of each 
        coord_prod = [z[0] for z in prod]
        value_prod = [z[1] for z in prod]
        match = sparse_match([e[0] for e in sparse_operands], *coord_prod)
        if match:
            if isinstance(output, SparseTensor):
                output.add(tuple([match[l] for l in output_subscript]), np.prod(value_prod)*np.prod([op[tuple([match[l] for l in sub])] for sub, op in dense_operands]))
            else:
                output += np.prod(value_prod)*np.prod([op[tuple([match[l] for l in sub])] for sub, op in dense_operands])
    if isinstance(output, SparseTensor) and not stay_sparse:
        return output.to_dense()
    else:
        return output
    

def concatenate_sparse_tensors(tensors, injection_rank):
    """
    Concatenate a list of sparse tensors with the same shape together, making a new sparse tensor.
    """
    shape = copy.deepcopy(tensors[0].shape)
    for i in range(len(tensors)):
        tensor = tensors[i]
        assert tensor.shape == shape, i
    shape = list(shape)
    shape.insert(injection_rank, len(tensors))
    shape = tuple(shape)
    new_tensor = SparseTensor(shape)

    for i in range(len(tensors)):
        tensor = tensors[i]
        for coord, value in zip(tensor.coords, tensor.values):
            new_coord = copy.deepcopy(list(coord))
            new_coord.insert(injection_rank, i)
            new_tensor[tuple(new_coord)] = value
    return new_tensor