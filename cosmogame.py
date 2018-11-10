'''
simple prototype of general visual look
'''

from gl_elements import GlElement
# TODO: for GL_TRIANGLE_FAN etc types
from OpenGL.GL import *
# TODO: for the hack with the broken threaded game loop
#       manual trigerring of the glut loop
from OpenGL.GLUT import glutMainLoopEvent

from gler2 import compile_and_load_program, hyperbolic_shader, draw, elements
from gler2 import camera_position_x, camera_position_y

from time import sleep
import logging
import numpy
np = numpy

import ctypes
from ctypes import CDLL, cdll

program = compile_and_load_program(hyperbolic_shader)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # add some circles ontop
    canonical_circle_n = 50
    canonical_circle_x = np.cos([(np.pi*2*i)/canonical_circle_n for i in range(canonical_circle_n+1)])
    canonical_circle_y = np.sin([(np.pi*2*i)/canonical_circle_n for i in range(canonical_circle_n+1)])

    canonical_circle_fan = np.array([pt for i in range(1, canonical_circle_n+1) for pt in
       ([0,0,0],
        [canonical_circle_x[i],   canonical_circle_y[i],   0],
        [canonical_circle_x[i-1], canonical_circle_y[i-1], 0])])

    def random_circles_triangles_instances(N_circles, r_size=0.3):
        # generate positions and colors of 1 circle
        # radius is random within given value
        radius = numpy.random.rand(1)*r_size 

        # position = 0., 0., 0. shift added to the fan
        circle_vertices = numpy.row_stack(radius*canonical_circle_fan + [0.,0.,0.])

        # color is random for the cirlce, not for the instance
        #circle_colors  = [[r,g,b] for r,g,b in numpy.random.rand(1, 3) for _ in range(canonical_circle_n*3)]

        data = numpy.zeros(len(circle_vertices), dtype = [("position", np.float32, 3)] )
        #data = numpy.zeros(len(circle_vertices), dtype = [("position", np.float32, 3),
        #                                                  ("color", np.float32, 3)] )
        data['position'] = circle_vertices
        #data['color']    = circle_colors
        logging.debug(data)

        # and positions for N_circles instances of this same circle
        circle_centers = (numpy.random.rand(N_circles, 3) - [0.5, 0.5, 0.])# * 2
        circle_colors  = numpy.random.rand(N_circles, 3)

        #instances = numpy.zeros(len(circle_centers), dtype = [("instance_position", np.float32, 3),
        #                                         ("color",    np.float32, 3)] )
        ##instances = numpy.zeros(len(circle_centers), dtype = [("instance_position", np.float32, 3)] )
        #instances['color']    = circle_colors
        #instances['instance_position'] = circle_centers

        # for fast updating of the position color and position are separate vbos
        instances_position = numpy.zeros(len(circle_centers), dtype = [("instance_position", np.float32, 3)])
        instances_color    = numpy.zeros(len(circle_centers), dtype = [("color", np.float32, 3)])
        instances_position['instance_position'] = circle_colors
        instances_color['color']                = circle_centers

        return GlElement(program, GL_TRIANGLE_FAN, [data], [instances_position, instances_color])



    # the ship
    a = [-0.7,   0.0,  0.0]
    b = [ 0.0,  -0.2,  0.0]
    c = [ 0.0,   0.2,  0.0]
    d = [-0.1,   0.0, -0.2]

    canonical_ship = np.array([a, b, c,
                               a, b, d,
                               a, c, d,
                               b, c, d])

    ship_size = 1.0
    ship_z = -0.5
    ship_vertices = numpy.zeros(len(canonical_ship), dtype = [("position", np.float32, 3),
                                                            ("color", np.float32, 3)] )
    ship_vertices['position'] = canonical_ship*ship_size

    ship_colors = []
    for color in [[1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0]][:int(len(canonical_ship)/3)]:
        ship_colors.append(color)
        ship_colors.append(color)
        ship_colors.append(color)
    ship_vertices['color']    = ship_colors

    ship_instances_position = numpy.zeros(1, dtype = [("instance_position", np.float32, 3)])
    ship_instances_position['instance_position'] = [camera_position_x, camera_position_y, ship_z]

    #return (rads, angles, zis),
    ship_element = GlElement(program, GL_TRIANGLES, [ship_vertices], [ship_instances_position])

    ''' Now there is a problem with handling user inputs
    I want to change the default behavior:
    * I need to move the camera as usual
    * and also update the ship's position.

    Or in future I will need:
    * update the ship's position and speed,
    * update the camera to track the ship's position.

    For now I want ot do this by passing a custom function to handle the user's input.
    But then I need all kinds of internal parameters of gler to program it (camera_position_x etc).
    Which is not very clean.
    Let's simplify, clarify these parameters.

    One of the pains is to have to change the full function for all keys, not just couple keys.
    Let's implement this with couple handler-functions for now.

    So later I'll have to do it the other way around:
    * the game loop calculates the position of the ship, updates it
    * and updates the camera's position
    * the user's inputs only change some parameters in ship's movement
    '''

    import gler2 

    # ad-hoc custom key handling functions
    def custom_handle_key_up():
        #glTranslate(0., -0.02, 0.)
        gler2.camera_position_y += gler2.shift_val
        glUniform1f(gler2.PARAM_center_y, gler2.camera_position_y)
        #glUniform1f(PARAM_center_y, camera_position_y+camera_tilt_y)

        # update the ship
        ship_element['instance_position'] = [-gler2.camera_position_x, -gler2.camera_position_y, ship_z]

    def custom_handle_key_down():
        #glTranslate()
        #glTranslate(0., 0.02, 0.)
        gler2.camera_position_y -= gler2.shift_val
        glUniform1f(gler2.PARAM_center_y, gler2.camera_position_y)
        #glUniform1f(PARAM_center_y, camera_position_y+camera_tilt_y)

        # update the ship
        ship_element['instance_position'] = [-gler2.camera_position_x, -gler2.camera_position_y, ship_z]

    def custom_handle_key_left():
        #glTranslate(0.02, 0., 0.)
        gler2.camera_position_x += gler2.shift_val
        glUniform1f(gler2.PARAM_center_x, gler2.camera_position_x)
        #glUniform1f(PARAM_center_x, camera_position_x+camera_tilt_x)

        # update the ship
        ship_element['instance_position'] = [-gler2.camera_position_x, -gler2.camera_position_y, ship_z]

    def custom_handle_key_right():
        #glTranslate(-0.02, 0., 0.)
        gler2.camera_position_x -= gler2.shift_val
        glUniform1f(gler2.PARAM_center_x, gler2.camera_position_x)
        #glUniform1f(PARAM_center_x, camera_position_x+camera_tilt_x)

        # update the ship
        ship_element['instance_position'] = [-gler2.camera_position_x, -gler2.camera_position_y, ship_z]

    gler2.handle_key_up    = custom_handle_key_up
    gler2.handle_key_down  = custom_handle_key_down
    gler2.handle_key_left  = custom_handle_key_left
    gler2.handle_key_right = custom_handle_key_right


    # tetrahedrons for asteroids etc
    a = [-0.5,  0.0,  0.0]
    b = [ 0.0,  0.0,  0.0]
    c = [ 0.0,  0.5,  0.0]
    d = [ 0.0,  0.0,  0.5]

    canonical_tetrahedron = np.array([a, b, c,
                                      a, b, d,
                                      a, c, d,
                                      b, c, d])

    def random_tetraheders(N_instances, r_size=0.3,
            random_position=False,
            spread=[1,1,0.1],
            central_position=[0, -15, 0],
            radius = 5.):
        data = numpy.zeros(len(canonical_tetrahedron), dtype = [("position", np.float32, 3),
                                                                ("color", np.float32, 3)] )
        data['position'] = canonical_tetrahedron*r_size
        colors = []
        #for color in numpy.random.rand(4, 3):
        #    colors.append(color)
        #    colors.append(color)
        #    colors.append(color)
        for color in [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0]][:int(len(canonical_tetrahedron)/3)]:
            colors.append(color)
            colors.append(color)
            colors.append(color)
        data['color']    = colors

        # for fast updating of the position color and position are separate vbos
        instances_position = numpy.zeros(N_instances, dtype = [("instance_position", np.float32, 3)])
        if random_position:
            #instances_position['instance_position'] = (numpy.random.rand(N_instances, 3) - [0.5, 0.5, 0]) * spread
            # less span in z axis
            # circular distr
            # numpy.column_stack((numpy.random.rand(5), numpy.ones(5)))
            zis = numpy.random.rand(N_instances) * spread[2]
            angles = (np.random.random(N_instances) - 0.5) * 2 * np.pi
            rads   = np.random.random(N_instances)*0.1 + radius
            xis = np.cos(angles)
            yis = np.sin(angles)
            positions = np.column_stack((xis*rads, yis*rads, zis))
            instances_position['instance_position'] = positions + central_position

        #return GlElement(program, GL_TRIANGLE_STRIP, [data], [instances_position])
        return (rads, angles, zis), GlElement(program, GL_TRIANGLES, [data], [instances_position])


    # testing updates


    block_a   = [-4.0, -2.0, -0.5]
    block_b   = [-4.0,  2.0, -0.5]
    block_c   = [ 4.0,  2.0, -0.5]
    block_d   = [ 4.0, -2.0, -0.5]

    block_a_t = [-4.0, -2.0,  0.5]
    block_b_t = [-4.0,  2.0,  0.5]
    block_c_t = [ 4.0,  2.0,  0.5]
    block_d_t = [ 4.0, -2.0,  0.5]

    # indicies are better for this
    canonical_paral = np.array([block_a,   block_b,   block_c,   block_d,
                                block_a_t, block_b_t, block_c_t, block_d_t,
                                block_a,   block_b,   block_b_t, block_a_t,
                                block_b,   block_c,   block_c_t, block_b_t,
                                block_c,   block_d,   block_d_t, block_c_t,
                                block_d,   block_a,   block_a_t, block_d_t,
                                      ])

    block_a   = [-0.5, -0.5, -0.5]
    block_b   = [-0.5,  0.5, -0.5]
    block_c   = [ 0.5,  0.5, -0.5]
    block_d   = [ 0.5, -0.5, -0.5]

    block_a_t = [-0.5, -0.5,  0.5]
    block_b_t = [-0.5,  0.5,  0.5]
    block_c_t = [ 0.5,  0.5,  0.5]
    block_d_t = [ 0.5, -0.5,  0.5]

    # indicies are better for this
    canonical_cube = np.array([block_a,   block_b,   block_c,   block_d,
                               block_a_t, block_b_t, block_c_t, block_d_t,
                               block_a,   block_b,   block_b_t, block_a_t,
                               block_b,   block_c,   block_c_t, block_b_t,
                               block_c,   block_d,   block_d_t, block_c_t,
                               block_d,   block_a,   block_a_t, block_d_t,
                                     ])

    def random_blocks(N_instances, r_size=0.5, spread=8., central_position=[0., 0., 0.], canonical_block=canonical_paral):
        data = numpy.zeros(len(canonical_block), dtype = [("position", np.float32, 3),
                                                          ("color",    np.float32, 3)] )
        data['position'] = canonical_block*r_size
        colors = []
        #for color in numpy.random.rand(4, 3):
        #    colors.append(color)
        #    colors.append(color)
        #    colors.append(color)
        for color in [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]][:int(len(canonical_block)/4)]:
            colors.append(color)
            colors.append(color)
            colors.append(color)
            colors.append(color)
        data['color']    = colors

        instances_position = numpy.zeros(N_instances, dtype = [("instance_position", np.float32, 3)])
        xis = (np.random.random(N_instances) - 0.5) * spread
        yis = numpy.random.rand(N_instances) * 0.2
        zis = numpy.random.rand(N_instances) * 0.1
        positions = np.column_stack((xis, yis, zis))
        instances_position['instance_position'] = positions + central_position

        return GlElement(program, GL_QUADS, [data], [instances_position])

    N_tetra = 100
    planet_center = [-150., -15., 0.]
    (satelites_rads, satelites_angles, satelites_zis), satelites = random_tetraheders(N_tetra, 0.5, random_position=True, spread=[100,1,0.1], central_position=planet_center)

    system_center = [30., -300., 0.]
    (asteroids_rads, asteroids_angles, asteroids_zis), asteroids = random_tetraheders(300, 0.5, random_position=True, spread=[100,1,0.1], central_position=system_center, radius = 310.)

    #elements.extend([random_circles_triangles_instances(N_tetra)])
    elements.extend([satelites,
                random_blocks(1, 5, 0.1, central_position=planet_center, canonical_block=canonical_cube),
                asteroids,
                random_blocks(75, central_position=[0., 5., 0.], spread=30.),
                ship_element])

    # Запускаем основной цикл
    from threading import Thread
    def graphic_thread():
        # trying to init threads in X
        libpath = "/usr/lib/x86_64-linux-gnu/libX11.so"
        cdll.LoadLibrary( libpath )
        lib = CDLL( libpath )
        lib.XInitThreads()
        # does not help

        # and the glut
        glutMainLoop()

    glThread = Thread(target=graphic_thread)
    #glThread = Thread(target=glutMainLoop)
    # threads crash from time to time with:
    #[xcb] Unknown request in queue while dequeuing
    #[xcb] Most likely this is a multi-threaded client and XInitThreads has not been called
    #[xcb] Aborting, sorry about that.
    # people say X11 is not thread safe and there is some special option to turn on to make it safe

    ## trying multiprocessing (the memory must be shared)
    #from multiprocessing import Process
    #glThread = Process(target=glutMainLoop)
    ## got this
    ##XIO:  fatal IO error 11 (Resource temporarily unavailable) on X server ":0"
    ##      after 57 requests (57 known processed) with 5 events remaining.
    ##XIO:  fatal IO error 11 (Resource temporarily unavailable) on X server ":0"
    ##      after 59 requests (59 known processed) with 0 events remaining.
    ##DRM_IOCTL_I915_GEM_CONTEXT_DESTROY failed: No such file or directory
    # this is why apple and microsoft rule desktop

    logging.debug(repr(satelites_angles))
    logging.debug(repr(satelites_rads))

    def game():
        global satelites_angles, satelites_rads, satelites_zis
        global asteroids_angles, asteroids_rads, asteroids_zis

        sleep(2.)
        while True:
            sleep(0.01)
            #print('foo')
            # sim
            #circle_centers = (numpy.random.rand(N_instances, 3) - [0.5, 0.5, 0.])# * 2
            #instance_centers = (numpy.random.rand(N_instances, 3) - [0.5, 0.5, 0.])# * 2
            # increment angle
            satelites_angles += 0.005
            logging.debug(repr(satelites_angles))
            xis = np.cos(satelites_angles)
            yis = np.sin(satelites_angles)
            positions = np.column_stack((xis*satelites_rads, yis*satelites_rads, satelites_zis))
            satelites_centers = positions + planet_center
            # update
            #elements[0]['instance_position'] = satelites_centers
            satelites['instance_position'] = satelites_centers

            asteroids_angles += 0.0001
            xis = np.cos(asteroids_angles)
            yis = np.sin(asteroids_angles)
            positions = np.column_stack((xis*asteroids_rads, yis*asteroids_rads, asteroids_zis))
            asteroids_centers = positions + system_center
            # update
            asteroids['instance_position'] = asteroids_centers

            # draw
            draw()
            glutMainLoopEvent() # should substitute the mainloop

    # thre graphics thread
    #glThread.start()

    ## try this fro sim part
    #libpath = "/usr/lib/x86_64-linux-gnu/libX11.so"
    #cdll.LoadLibrary( libpath )
    #lib = CDLL( libpath )
    #lib.XInitThreads()
    ## does not help

    game()

    ## does not work
    #gameThread = Thread(target=game)
    #gameThread.start()
    #glutMainLoop()





