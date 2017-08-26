import pickle
import logging
import numpy
from gler import known_elements


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

marker = [marker points], [elements description]
like in
elements = [('quad', 1), ('point', 2)]

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
        self.elements = elements
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


def count_Y(record_tuple):
    #y, m, d, h = record_tuple
    y, m, w, wd, h = record_tuple
    y -= 2015
    #return ((y*400 + m*33 + d)*4 + h)
    return ((y*650 + m*50 + w*8 + wd)*4 + h)

def count_X(record_tuple2):
    y, m, w, wd = record_tuple2
    y -= 2015
    return (y*650 + m*50 + w*8 + wd)*4

def normy(c):
    #return (c - 3500) / 2000.
    return (c - 5500) / 4000.

def normx(c):
    #return (c - 5500) / 4000.
    return (c - 1500) / 2000.

def norm(v_px):
    return (v_px - 5500) / 4000.

rgb_colors = {0: (0,0,0), 1: (76, 153, 0), 2: (220, 220, 0), 3: (255, 128, 0), 4: (255, 51, 51), 5: (153, 0, 0)}
def col(a):
    return rgb_colors[a]


def test_markers_on_cern_hostel_data():
    with open('data2.p', 'rb') as f:
        hostel_data = pickle.load(f)
    logging.info(len(hostel_data))

    p = numpy.array([record_tuple + day + col(load) # this is info on 1 point of the quad
                          for record_tuple, record in hostel_data.items()
                          for day, load in record.items()])

    logging.info(p.shape)

    # record_tuple is p[:,:5] etc
    point_color = p[:,-3:] / 255.
    #return ((y*650 + m*50 + w*8 + wd)*4 + h)
    point_y1 = norm(((p[:,0] - 2015)*650 + p[:,1]*50 + p[:,2]*9 + p[:,3])*4 + p[:,4])
    point_y2 = norm(((p[:,0] - 2015)*650 + p[:,1]*50 + p[:,2]*9 + p[:,3])*4 + p[:,4] + 1)
    point_x1 = norm(((p[:,5] - 2015)*650 + p[:,6]*50 + p[:,7]*8 + p[:,8])*4)
    point_x2 = norm(((p[:,5] - 2015)*650 + p[:,6]*50 + p[:,7]*8 + p[:,8])*4 + 4)

    flat_quad_points = numpy.column_stack((point_x1, point_y1, numpy.zeros_like(point_x1),
                                      point_x2, point_y1, numpy.zeros_like(point_x1),
                                      point_x2, point_y2, numpy.zeros_like(point_x1),
                                      point_x1, point_y2, numpy.zeros_like(point_x1)))

    return Markers(flat_quad_points, [('quad', 1)])

def main():
    with open('data2.p', 'rb') as f:
        hostel_data = pickle.load(f)
    logging.info(len(hostel_data))

    # this is kind of "Quads" marker
    #   the SVG-like description of the plot, in pixels of the plot, not in some GG measure
    #   no info on original data, just plot
    #   (maybe it would be useful to still keep the meaning some way -- will see)
    #numpy.array([[x, y, z, dx, dy, c]
    # nope, it's just the top-left point of the quad
    # numpy.stack is available only in numpy 1.10 which is not in pip3 now
    # thus dealing with point arrays via reshaping
    p = numpy.array([record_tuple + day + col(load) # this is info on 1 point of the quad
                          for record_tuple, record in hostel_data.items()
                          for day, load in record.items()])

    logging.info(p.shape)

    # record_tuple is p[:,:5] etc
    point_color = p[:,-3:] / 255.
    #return ((y*650 + m*50 + w*8 + wd)*4 + h)
    point_y1 = norm(((p[:,0] - 2015)*650 + p[:,1]*50 + p[:,2]*9 + p[:,3])*4 + p[:,4])
    point_y2 = norm(((p[:,0] - 2015)*650 + p[:,1]*50 + p[:,2]*9 + p[:,3])*4 + p[:,4] + 1)
    point_x1 = norm(((p[:,5] - 2015)*650 + p[:,6]*50 + p[:,7]*8 + p[:,8])*4)
    point_x2 = norm(((p[:,5] - 2015)*650 + p[:,6]*50 + p[:,7]*8 + p[:,8])*4 + 4)

    # beh..
    # annoying SVG-like-ness.. let's do directly to GL for now
    flat_quad_points = numpy.column_stack((point_x1, point_y1, numpy.zeros_like(point_x1),
                                      point_x2, point_y1, numpy.zeros_like(point_x1),
                                      point_x2, point_y2, numpy.zeros_like(point_x1),
                                      point_x1, point_y2, numpy.zeros_like(point_x1)))

    mrks = Markers(flat_quad_points, [('quad', 1)])

    flat_quad_point_colors = numpy.column_stack((point_color, point_color, point_color, point_color))

    #elements = [('quad', len(point_x1))] # old
    #elements = {'quad': len(point_x1)} # need SortedDict, let's stick to tuples
    flat_points, elements = mrks.flatten_for_gl() #flat_quad_points.reshape(-1, 3)
    logging.info((flat_points == flat_quad_points.reshape(-1,3)).all())
    flat_colors = flat_quad_point_colors.reshape(-1, 3)

    logging.info(flat_points.shape)
    logging.info(flat_colors.shape)
    logging.info(elements)

    # select the points into quads with colors for all 4 points
    # p[:,0] -- x
    # p[:,1] -- y
    # z, color
    import gler
    gler.glThread.start()
    gler.pointdata, gler.pointcolor, gler.pointelements = flat_points, flat_colors, elements


if __name__ == '__main__':
    main()

