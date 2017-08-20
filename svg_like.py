import numpy


class Marker(object):
    '''
    object stores array of 1 elementary type of objects
    '''

    _elements = {'line': 2, 'quad': 4, 'triangle': 3, 'point': 1}

    def __init__(self, array, element='line'):
        assert element in self._elements # known elementary shape
        self._array = numpy.array(array)
        n_lines, n_points, point_dim = self._array.shape
        assert n_points == self._elements[element] and point_dim == 3
        self.element  = element
        self.n_points = n_points

    def flatten_for_gl(self):
        #return self.n_points, len(self._array), self._array.reshape(-1, self._array.shape[-1])
        return self._array.reshape(-1, self._array.shape[-1])

    def __repr__(self):
        return repr(self._array)

    def __str__(self):
        return str(self._array)

    def __len__(self):
        return len(self._array)

'''
need: 1 object holding everything to draw, consumable by gl's draw function
-- function to construct from a list?

nested/composed Markers

MarkerObjects = {'QUADS': Marker(...)}
'''

def constr_gl(objs):
    '''
    return array of all points and colors (random)
    and the encoding of drawing elements in the arrays -- list(('element', N_points, len))
    '''

    points = numpy.concatenate((obj.flatten_for_gl() for obj in objs))
    colors = numpy.random.rand(len(points), 3)
    #elements = [(obj.element, obj.n_points, len(obj)) for obj in objs]
    elements = [(obj.element, len(obj)) for obj in objs] # n_points is redundant
    return points, colors, elements


