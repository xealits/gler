import logging, pickle, numpy
np = numpy

from gl_elements import GlElement
# TODO: for GL_TRIANGLE_FAN etc types
from OpenGL.GL import *
from OpenGL.GLUT import glutMainLoop

from gler2 import compile_and_load_program, std_flat_shader, elements

program = compile_and_load_program(std_flat_shader)


norm_scale = 1 / 200.

def norm(v_px):
    return (v_px - 5500) * norm_scale

rgb_colors = {0: (0,0,0), 1: (76, 153, 0), 2: (220, 220, 0), 3: (255, 128, 0), 4: (255, 51, 51), 5: (153, 0, 0)}
def col(a):
    return rgb_colors[a]

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
    # (record_tuple) = (2017, 9, 4, 0, 0) year month week day day-quarter
    # {record} = {(day): load_number, ...}
    # (day) = (2018, 10, 3, 0) year month week day
    p = numpy.array([record_tuple + day + col(load) # this is info on 1 point of the quad
                          for record_tuple, record in hostel_data.items()
                          for day, load in record.items()])

    logging.info(p.shape)

    # record_tuple is p[:,:5] etc
    point_color = p[:,-3:] / 255.
    #return ((y*650 + m*50 + w*8 + wd)*4 + h)
    point_y1 = norm(((p[:,0] - 2015)*650 + p[:,1]*50 + p[:,2]*9 + p[:,3])*4 + p[:,4])
    #point_y2 = norm(((p[:,0] - 2015)*650 + p[:,1]*50 + p[:,2]*9 + p[:,3])*4 + p[:,4] + 1)
    point_x1 = norm(((p[:,5] - 2015)*650 + p[:,6]*50 + p[:,7]*8 + p[:,8])*4)
    #point_x2 = norm(((p[:,5] - 2015)*650 + p[:,6]*50 + p[:,7]*8 + p[:,8])*4 + 4)

    # beh..
    # annoying SVG-like-ness.. let's do directly to GL for now
    flat_quad_points = numpy.column_stack((point_x1, point_y1, numpy.zeros_like(point_x1)))
                                      #point_x2, point_y1, numpy.zeros_like(point_x1),
                                      #point_x2, point_y2, numpy.zeros_like(point_x1),
                                      #point_x1, point_y2, numpy.zeros_like(point_x1)))


    a_pixel = np.array([[ 0.0,  0.0, -0.5],
               [ 0.0,  1.0, -0.5],
               [ 4.0,  1.0, -0.5],
               [ 4.0,  0.0, -0.5]])
    data = numpy.zeros(len(a_pixel), dtype = [("position", np.float32, 3)])
    data['position'] = a_pixel * norm_scale

    assert len(point_color) == len(flat_quad_points)
    instances = np.zeros(len(flat_quad_points), dtype = [("instance_position", np.float32, 3),
                                                       ("color", np.float32, 3)])
    instances['instance_position'] = flat_quad_points
    instances['color']             = point_color


    #logging.info('vtx shape %s' % repr(drawing_spec.elements_vtx.shape))
    #logging.info('elements  %s' % repr(drawing_spec.elements_spec))

    rectangles = GlElement(program, GL_QUADS, [data], [instances])
    #elements.extend([random_circles_triangles_instances(N_tetra)])
    elements.extend([rectangles])
    glutMainLoop()

if __name__ == '__main__':
    from threading import Thread
    def graphic_thread():
        ## trying to init threads in X
        #libpath = "/usr/lib/x86_64-linux-gnu/libX11.so"
        #cdll.LoadLibrary( libpath )
        #lib = CDLL( libpath )
        #lib.XInitThreads()
        ## does not help

        # and the glut
        #main()
        glutMainLoop()

    #glThread = Thread(target=main)
    #glThread.start()

    main()

