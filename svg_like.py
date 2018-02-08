import pickle
import logging
import numpy
from gler import known_elements, gl_elements, GlElement, GlObjects


logging.basicConfig(level=logging.INFO)

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

MarkerObjects = {'QUADS': Marker(...)}


nested/composed Markers

[marker, same, markers...]

marker = [('element_type', N these elements in the marker), ('another element type', N),...]
like in
elements = [('quad', 1), ('point', 2)]
-- describes 1 composed element

then transpose the [markers] array
to get [points of element 1, points of element 2, ...]
and set elements = [(element 1, N markers * n elements 1), etc]
'''

class Markers(object):
    '''
    object stores array of "simple description 1" objects
    '''

    def __init__(self, array, elements):
        '''Markers(markers_points, marker_elements)

        --> let's make elements a dictionary <-- need SortedDict
        let's stick to tuples..
        '''
        assert all(e in known_elements for e, _ in elements)
        # the description of elements in the marker
        self.elements = elements
        # N points per 1 marker
        self.n_points = sum(n*known_elements[e][1] for e, n in elements)

        #ar = numpy.array(array)
        #assert ar.shape[1] = n_points # let's do nested arrays of points for now
        # for simplicity let's flatten marker points right away
        self._array = numpy.array(array).reshape(-1, 3*self.n_points)

    def flatten_for_gl(self):
        '''
        rearrange points to cluster separate elements
        return array[[x,y,z point of element1], [point of element 1], ... element 2 ... element 3]
        and elements dict('element': N_elements)
        '''
        #return self.n_points, len(self._array), self._array.reshape(-1, self._array.shape[-1])
        #return self._array.reshape(-1, self.n_points)
        # TODO: not sure how to do it with nupy stuff
        # ar[:,:3*3].reshape((-1,3)) -- example, get 1 triangle from begining
        points_per_elements = {}
        cur_index = 0
        for e, n in self.elements:
            # 3 coords for a point * N points per element * N elements in the marker
            n_coords = 3*known_elements[e][1]*n
            points_per_elements[e] = self._array[:,cur_index:cur_index+n_coords].reshape((-1,3))
            cur_index += n_coords

        flat_elements = [(e, n*len(self._array)) for e, n in self.elements]

        return numpy.row_stack((points for _, points in points_per_elements.items())), flat_elements


    def __repr__(self):
        return repr(self._array)

    def __str__(self):
        return str(self._array)

    def __len__(self):
        return len(self._array)


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


def random_points(N_points):
    pointverts = 2*(numpy.random.rand(N_points, 3) - 0.5)
    pointcolor = numpy.random.rand(N_points, 3)
    return pointverts, pointcolor


def test_triangle_strip(N_points):

    vertices, colours = random_points(N_points)

    mrks = Markers(vertices, [('triangle_strip', 1)]) # 1 element in marker, but at the end gl will get all points for this..

    flat_points, flat_elements = mrks.flatten_for_gl()

    #print(repr(flat_points))
    #print(repr(colours))
    #print(repr(flat_elements))

    import gler
    gler.pointdata, gler.pointcolor, gler.pointelements = flat_points, colours, flat_elements
    #gler.glutPostRedisplay()
    if not gler.glThread.isAlive():
        gler.glThread.start()


if __name__ == '__main__':
    N_points = 2
    print("testing the %d circles" % N_points)
    test_triangle_strip(N_points)

