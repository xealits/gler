# -*- coding: utf-8 -*-
# from 
# Импортируем все необходимые библиотеки:
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

import logging
import numpy
np = numpy


class GlElement(object):

    def __init__(self, program, gl_primitive, vertices, instances=None):
        '''
        create one vao and vbos for element data (marker and instances if needed)
        and set rules for drawing in the vao

        the rules must be:
           names from input numpy arrays are found in program and bound there
        '''

        self.program = program
        self.primitive = gl_primitive

        # инициализация на ГПУ
        # просто загрузим буферы, правила отрисовки -- при рендеринге
        # TODO: I assume OpenGL is already setup, need to add the setup rutienes here, make a complete svg_like module

        # the vao of the full element
        self.element_vao = glGenVertexArrays(1)
        glBindVertexArray(self.element_vao)

        # the vertices buffer
        self.vertices = vertices
        self.vertices_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_buffer)

        # Upload data
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

        # set the attribute layout in the vbo
        # this description is saved in the bound vao

        # attaching vertex buffer attributes to the program, setting parsing rules
        stride = self.vertices.strides[0]
        offset = 0
        for name in self.vertices.dtype.names:
            print('attaching ' + name)
            # prepare offset for opengl parsing rule and shift it for the next item
            gl_offset = ctypes.c_void_p(offset)
            offset += self.vertices.dtype[name].itemsize

            # attach to the location in program
            loc = glGetAttribLocation(program, name)
            print('at ' + str(loc))
            glEnableVertexAttribArray(loc)
            glBindBuffer(GL_ARRAY_BUFFER, self.vertices_buffer) # already bound, not sure if I need to repeate it
            # probably it has something to do with binding all these buffers and instances to the same target
            # probably Enable shifts the target of the target
            #glVertexAttribPointer(loc, 3, GL_FLOAT, False, stride, offset)
            #print(name, self.vertices[name].shape, self.vertices[name].shape[1])
            n_coords = self.vertices[name].shape[1] # n coords in this attribute of the buffer
            glVertexAttribPointer(loc, n_coords, GL_FLOAT, False, stride, gl_offset)

            # in khronos tutorial (vaos and vbos) they run glEnable after here

        # same for instance buffers

        if instances is not None:
            # Request a buffer slot from GPU
            instance_buffer = glGenBuffers(1)

            glBindBuffer(GL_ARRAY_BUFFER, instance_buffer)

            # upload
            glBufferData(GL_ARRAY_BUFFER, instances.nbytes, instances, GL_DYNAMIC_DRAW)

            self.instances = instances
            self.instances_buffer = instance_buffer

            glBindBuffer(GL_ARRAY_BUFFER, self.instances_buffer)

            # attaching vertex buffer attributes to the program, setting parsing rules
            stride = self.instances.strides[0]
            offset = 0
            for name in self.instances.dtype.names:
                print('attaching ' + name)
                # same procedure
                gl_offset = ctypes.c_void_p(offset)
                offset += self.instances.dtype[name].itemsize

                # attach to the location in program
                loc = glGetAttribLocation(program, name)
                print('at ' + str(loc))
                glEnableVertexAttribArray(loc)
                glBindBuffer(GL_ARRAY_BUFFER, self.instances_buffer)
                #glVertexAttribPointer(loc, 3, GL_FLOAT, False, stride, offset)
                n_coords = self.instances[name].shape[1] # n coords in this attribute of the buffer
                glVertexAttribPointer(loc, n_coords, GL_FLOAT, False, stride, gl_offset)

                # and the step for instances
                glVertexAttribDivisor(loc, 1)
                # so it means step of 1 of n_coord floats of the insance attribute
                # TODO: need to add other types than floats

        else:
            self.instances = None

        # and unbind the vbo target
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        # khronos tutorial says I can delete vbo buffers
        # they are just pointers
        # since vbos are tracked in vao they won't be cleaned up

        # and the vao
        glBindVertexArray(0)

    def __repr__(self):
        if self.instances:
            return 'GlElement(%s, %s, #%d vert, #%d inst)' % (repr(self.program), repr(self.primitive), len(self.vertices), len(self.instances))
        else:
            return 'GlElement(%s, %s, #%d vert)' % (repr(self.program), repr(self.primitive), len(self.vertices))

    def glDraw(self, vertices_name="position"):
        '''
        draw this element in OpenGL
        the element is bunch of vertices and instances if given
        find all names of the element vertices and instances in the shader program and attach them
        then draw according to primitive and instancing rules
        '''

        #all_attribute_array_locations = []
        #instance_attribute_array_locations = []

        # all is set in vao
        # bind it
        glBindVertexArray(self.element_vao)

        # now drawing rules
        # here I assume 1 attribute of the shader -- the vertices of the primitives
        # which are in "vertices" buffer
        # they are called "position" by default
        # we need to know how man vertices there are
        # other attributes must be pulled automatically

        # N primitives and N vertices per each
        n_primitives = self.vertices['position'].shape[0]
        # = self.vertices.shape[0]
        n_vertices   = self.vertices['position'].shape[1]
        n_vertices_to_draw = n_primitives * n_vertices

        if self.instances is not None:
            n_instances = self.instances.shape[0]
            print('drawing instances', n_instances, n_vertices_to_draw)
            glDrawArraysInstanced(self.primitive, 0, n_vertices_to_draw, n_instances)
            # just drawing all vertices for all instances

        else:
            print('drawing primitives', n_vertices_to_draw)
            glDrawArrays(self.primitive, 0, n_vertices_to_draw)


        # supposedly I don't need these anymore
        ## let's try unseting divisors
        #for loc in instance_attribute_array_locations:
        #    glVertexAttribDivisor(loc, 0) # works!

        ## and try disabling all attribute arrays
        #for loc in all_attribute_array_locations:
        #    glDisableVertexAttribArray(loc)

        # unbind
        #glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return 0



if __name__ == '__main__':
    global g_Width
    global g_Height
    g_Width  = 300
    g_Height = 300

    # Процедура подготовки шейдера (тип шейдера, текст шейдера)
    def create_shader(shader_type, source):
        # Создаем пустой объект шейдера
        shader = glCreateShader(shader_type)
        # Привязываем текст шейдера к пустому объекту шейдера
        glShaderSource(shader, source)
        # Компилируем шейдер
        glCompileShader(shader)
        # Возвращаем созданный шейдер
        return shader

    def reshape(width, height):
        global g_Width, g_Height
        g_Width = width
        g_Height = height
        glViewport(0, 0, g_Width, g_Height)

    def draw():
        glClear(GL_COLOR_BUFFER_BIT)

        for element in elements:
            print(element.glDraw())

        glutSwapBuffers()

    # Использовать двойную буферезацию и цвета в формате RGB (Красный Синий Зеленый)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    # Указываем начальный размер окна (ширина, высота)
    glutInitWindowSize(g_Width, g_Height)
    # Указываем начальное
    # положение окна относительно левого верхнего угла экрана
    glutInitWindowPosition(50, 50)
    # Инициализация OpenGl
    glutInit(sys.argv)
    # Создаем окно с заголовком
    glutCreateWindow(b"gl_elements!")

    # reshape function -- reacts to window reshape
    glutReshapeFunc(reshape)
    # Определяем процедуру, отвечающую за перерисовку
    glutDisplayFunc(draw) # когда она запускается?
    # мне нужно 1 раз загрузить буферы с вершинами и всё, дальше только перерисовывать зум и движение по графику

    ## Определяем процедуру, отвечающую за обработку клавиш
    #glutSpecialFunc(specialkeys)

    # Задаем серый цвет для очистки экрана
    glClearColor(0.2, 0.2, 0.2, 1)

    # новый шейдер
    # с атрибутами вершин
    # не знаю как оно отличается от предыдущего -- там не явно както всё передаётся
    # тут в атрибуты явно загонится буфер с вершинами для отрисовки
    #
    # только для зума нужно таки использовать ту проджекшн матрикс
    #  uniform float scale;
    #  uniform float shift_x, shift_y;
    #        Vertex.x += shift_x;
    #        Vertex.y += shift_y;
    #        Vertex.x *= scale;
    #        Vertex.y *= scale;
    #        Vertex.z *= scale;
    #    vec4 Vertex = vec4(position, 1.0);

    vertex = create_shader(GL_VERTEX_SHADER,"""
      attribute vec3 position;
      attribute vec3 color;
      attribute vec3 instance_position;
      varying vec3 vertex_color;
      void main()
      {
        vec4 Vertex = vec4(instance_position + position, 1.0);
        gl_Position = gl_ModelViewProjectionMatrix * Vertex;
        vertex_color = color;
      } """)

    # Создаем фрагментный шейдер:
    # Определяет цвет каждого фрагмента как "смешанный" цвет его вершин
    fragment = create_shader(GL_FRAGMENT_SHADER, """
    varying vec3 vertex_color;
    void main() {
                   gl_FragColor = vec4(vertex_color, 1.0);
    }""")

    # Создаем пустой объект шейдерной программы
    program = glCreateProgram()
    # Приcоединяем вершинный шейдер к программе
    glAttachShader(program, vertex)
    # Присоединяем фрагментный шейдер к программе
    glAttachShader(program, fragment)
    # "Собираем" шейдерную программу
    glLinkProgram(program)

    # are these required?
    glDetachShader(program, vertex)
    glDetachShader(program, fragment)

    # Сообщаем OpenGL о необходимости использовать данную шейдерну программу при отрисовке объектов
    glUseProgram(program)

    #pointdata = [[0, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.4, -0.1, 0], [0., 0.7, 0], [-0.2, 0.5, 0], [0.1,0.1,0], [0.2,0.3,0], [-0.1,0.1,0]]
    #pointcolor = [[1, 1, 0], [0, 1, 1], [1, 0, 1], [0., 0., 1], [0., 1., 0], [1., 0., 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]]

    pointdata  = [[0, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]]
    pointcolor = [  [1, 1, 0],       [0, 1, 1],      [1, 0, 1]]
    #pointcolor = [  [0, 0, 1],       [0, 1, 1],      [1, 0, 1]]

    data = numpy.zeros(len(pointdata), dtype = [ ("position", np.float32, 3),
                                                 ("color",    np.float32, 3)] )

    data['color']    = pointcolor
    data['position'] = pointdata

    #print(data)
    # each is numpy array of 3-coordinate vectors

    linepoints = [[-0.7, -0.5, 0], [-0.5, 0.7, 0], [-0.2, 0.5, 0], [0.1,0.1,0], [0.2,0.3,0], [-0.1,0.1,0]]
    linecolor  = [    [0., 0., 1],  [0., 1., 0],    [1., 0., 0],   [1, 1, 0],   [0, 1, 1],    [1, 0, 1]]
    #linecolor  = [    [1., 1., 0],  [0., 1., 0],    [1., 0., 0],   [1, 1, 0],   [0, 1, 1],    [1, 0, 1]]

    linedata = numpy.zeros(len(linepoints), dtype = [("position", np.float32, 3),
                                                 ("color",    np.float32, 3)] )

    linedata['color']    = linecolor
    linedata['position'] = linepoints
    # each is numpy array of 3-coordinate vectors

    print(linedata)

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
        print(data)

        # and positions for N_circles instances of this same circle
        circle_centers = (numpy.random.rand(N_circles, 3) - [0.5, 0.5, 0.])# * 2
        circle_colors  = numpy.random.rand(N_circles, 3)

        instances = numpy.zeros(len(circle_centers), dtype = [("instance_position", np.float32, 3),
                                                 ("color",    np.float32, 3)] )
        #instances = numpy.zeros(len(circle_centers), dtype = [("instance_position", np.float32, 3)] )
        instances['color']    = circle_colors
        instances['instance_position'] = circle_centers

        return GlElement(program, GL_TRIANGLE_FAN, data, instances)

    #elements = [GlElement(program, GL_TRIANGLES, data)]
    #elements = [GlElement(program, GL_LINES, linedata)]
    #elements = [GlElement(program, GL_TRIANGLES, data), GlElement(program, GL_LINES, linedata)]
    #elements = [random_circles_triangles_instances(3)]
    #elements = [GlElement(program, GL_TRIANGLES, data), GlElement(program, GL_LINES, linedata), random_circles_triangles_instances(3)]

    zero_instance = numpy.zeros(1, dtype = [("instance_position", np.float32, 3)] )
    #elements = [GlElement(program, GL_TRIANGLES, data, zero_instance), GlElement(program, GL_LINES, linedata, zero_instance), random_circles_triangles_instances(3)]

    # Tafte-like linear box-plot
    zero_instance_white = numpy.zeros(1, dtype = [("instance_position", np.float32, 3),
                                                  ("color", np.float32, 3)] )
    zero_instance_white['color'] = [[1,1,1]]

    #averages = np.array([[0.7, -0.7, 0], [0.5, -0.5,  0], [0.2, -0.2,  0], [0.1,-0.1, 0], [-0.1, 0.0, 0], [-0.2, 0.2, 0]])
    #q_u1     = np.array([[0.0,  0.1, 0], [0.0,  0.05, 0], [0.0,  0.02, 0], [0.0, 0.1, 0], [ 0.0, 0.05,0], [ 0.0, 0.1, 0]])
    #q_u2     = np.array([[0.0,  0.3, 0], [0.0,  0.2,  0], [0.0,  0.2,  0], [0.0, 0.1, 0], [ 0.0, 0.25,0], [ 0.0, 0.2, 0]])
    #q_d1     = np.array([[0.0,  0.1, 0], [0.0,  0.05, 0], [0.0,  0.02, 0], [0.0, 0.1, 0], [ 0.0, 0.05,0], [ 0.0, 0.1, 0]])
    #q_d2     = np.array([[0.0,  0.3, 0], [0.0,  0.2,  0], [0.0,  0.2,  0], [0.0, 0.1, 0], [ 0.0, 0.25,0], [ 0.0, 0.2, 0]])

    N_data = 20
    averages = np.stack(((np.random.random(N_data) - 0.5)*2, (np.random.random(N_data) - 0.5) * 0.5, np.zeros(N_data)), axis=-1)
    q_u1     = np.stack((np.zeros(N_data), np.random.random(N_data), np.zeros(N_data)), axis=-1) * 0.2
    q_u2     = np.stack((np.zeros(N_data), np.random.random(N_data), np.zeros(N_data)), axis=-1) * 0.2
    q_d1     = np.stack((np.zeros(N_data), np.random.random(N_data), np.zeros(N_data)), axis=-1) * 0.2
    q_d2     = np.stack((np.zeros(N_data), np.random.random(N_data), np.zeros(N_data)), axis=-1) * 0.2

    # recalc actual coordinates around the average
    q_u1 = averages + q_u1
    q_u2 = q_u1 + q_u2

    q_d1 = averages - q_d1
    q_d2 = q_d1 - q_d2

    q_pairs_u = []
    q_pairs_d = []
    for i in range(len(averages)):
        q_pairs_u.append(q_u1[i])
        q_pairs_u.append(q_u2[i])
        q_pairs_d.append(q_d1[i])
        q_pairs_d.append(q_d2[i])

    #averages = [[0.5, -0.7, 0]]

    box_points = numpy.zeros(len(averages), dtype = [("position", np.float32, 3)])
    box_points['position'] = averages

    box_lines_u = numpy.zeros(len(q_pairs_u), dtype = [("position", np.float32, 3)])
    box_lines_u['position'] = q_pairs_u

    box_lines_d = numpy.zeros(len(q_pairs_d), dtype = [("position", np.float32, 3)])
    box_lines_d['position'] = q_pairs_d

    #averages = [[0.5, -0.7, 0]]
    #colors   = [[1.0,  1.0, 1]]
    #box_points = numpy.zeros(len(averages), dtype = [("position", np.float32, 3),
    #                                                 ("color", np.float32, 3)])
    #box_points['position'] = averages
    #box_points['color']    = colors
    #elements = [GlElement(program, GL_POINTS, box_points)] # works

    print(averages)
    print(box_points)
    print(zero_instance)
    print(zero_instance_white)

    #elements = [GlElement(program, GL_POINTS, box_points, zero_instance)] # and this works
    #elements = [GlElement(program, GL_POINTS, box_points, zero_instance_white)]
    #elements = [GlElement(program, GL_LINES, box_lines_u, zero_instance_white)]
    #elements = [GlElement(program, GL_LINES, box_lines_d, zero_instance_white)]
    elements = [GlElement(program, GL_POINTS, box_points, zero_instance_white),
                GlElement(program, GL_LINES, box_lines_u, zero_instance_white),
                GlElement(program, GL_LINES, box_lines_d, zero_instance_white)]

    # so now an additional point is drawn in the center of the window
    # there is no such point in drawing primitives

    #elements = [random_circles_triangles_instances(1)] # this works

    # not clear why the point at center is added, inctanced lines and circles are ok

    # Запускаем основной цикл
    glutMainLoop()


