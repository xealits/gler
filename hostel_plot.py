from svg_like import *


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

    #mrks = Markers(flat_quad_points, [('quad', 1)])
    logging.info('reshape: %s' % repr(flat_quad_points.reshape(-1,3)))
    #flat_points = flat_quad_points.reshape(-1,3)
    #elements = [GlElements(gl_elements['quad'], flat_points)]
    #drawing_spec = GlObjects([(GlElement(GL_LINES), lines_vtx), (GlElement(GL_POINTS), points_vtx)])
    drawing_spec = GlObjects([(GlElement(gl_elements['quad']), flat_quad_points)])

    flat_quad_point_colors = numpy.column_stack((point_color, point_color, point_color, point_color))

    #elements = [('quad', len(point_x1))] # old
    #elements = {'quad': len(point_x1)} # need SortedDict, let's stick to tuples
    #flat_points, elements = mrks.flatten_for_gl() #flat_quad_points.reshape(-1, 3)

    #logging.info((flat_points == flat_quad_points.reshape(-1,3)).all())
    flat_colors = flat_quad_point_colors.reshape(-1, 3)

    #logging.info(flat_points.shape)
    logging.info(flat_colors.shape)
    logging.info(drawing_spec)
    logging.info(drawing_spec.elements_spec)

    # select the points into quads with colors for all 4 points
    # p[:,0] -- x
    # p[:,1] -- y
    # z, color
    import gler
    gler.glThread.start()
    #gler.pointdata, gler.pointcolor, gler.pointelements = flat_points, flat_colors, elements
    gler.pointdata, gler.pointcolor, gler.pointelements = drawing_spec.elements_vtx, flat_colors, drawing_spec.elements_spec


if __name__ == '__main__':
    main()


