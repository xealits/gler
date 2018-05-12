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

    def __init__(self, gl_primitive, vertices, instances=None):
        '''
        create vbo for element data (marker and instances if needed)
        and set rules for drawing

        the rules must be:
           names from input numpy arrays are found in program and bound there
        '''

        self.primitive = gl_primitive

        # инициализация на ГПУ
        # просто загрузим буферы, правила отрисовки -- при рендеринге
        # TODO: I assume OpenGL is already setup, need to add the setup rutienes here, make a complete svg_like module
        self.vertices = vertices
        self.vertices_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_buffer)

        # Upload data
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

        if instances is not None:
            # Request a buffer slot from GPU
            instance_buffer = glGenBuffers(1)

            glBindBuffer(GL_ARRAY_BUFFER, instance_buffer)

            # upload
            glBufferData(GL_ARRAY_BUFFER, instances.nbytes, instances, GL_DYNAMIC_DRAW)

            self.instances = instances
            self.instances_buffer = instance_buffer
        else:
            self.instances = None

        # and unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def __repr__(self):
        if self.instances:
            return 'GlElement(%s, #%d vert, #%d inst)' % (repr(self.primitive), len(self.vertices), len(self.instances))
        else:
            return 'GlElement(%s, #%d vert)' % (repr(self.primitive), len(self.vertices))

    def glDraw(self, program, vertices_name="position"):
        '''
        draw this element in OpenGL
        the element is bunch of vertices and instances if given
        find all names of the element vertices and instances in the shader program and attach them
        then draw according to primitive and instancing rules
        '''

        # processing the vertex buffer

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

        # now instances
        # TODO: weirdly they are bound to the same target, I don't know how it works

        if self.instances is not None:
            print('instancing')
            glBindBuffer(GL_ARRAY_BUFFER, self.instances_buffer)

            # attaching vertex buffer attributes to the program, setting parsing rules
            stride = self.instances.strides[0]
            offset = 0
            for name in self.instances.dtype.names:
                # same procedure
                gl_offset = ctypes.c_void_p(offset)
                offset += self.instances.dtype[name].itemsize

                # attach to the location in program
                loc = glGetAttribLocation(program, name)
                glEnableVertexAttribArray(loc)
                glBindBuffer(GL_ARRAY_BUFFER, self.instances_buffer)
                #glVertexAttribPointer(loc, 3, GL_FLOAT, False, stride, offset)
                n_coords = self.instances[name].shape[1] # n coords in this attribute of the buffer
                glVertexAttribPointer(loc, n_coords, GL_FLOAT, False, stride, gl_offset)

                # and the step for instances
                glVertexAttribDivisor(loc, 1)
                # so it means step of 1 of n_coord floats of the insance attribute
                # TODO: need to add other types than floats

        '''
        if gl_element in self._const_elements:
            if n_vertices:
                assert n_vertices == self._const_elements[gl_element]
            else:
                n_vertices = self._const_elements[gl_element]

        self.element = gl_element
        self.n_vertices  = n_vertices  # per 1 element
        self.n_instances = n_instances
        '''

        # all set,
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
            print('drawing instances')
            n_instances = self.instances.shape[0]
            return glDrawArraysInstanced(self.primitive, 0, n_vertices_to_draw, n_instances)
            # just drawing all vertices for all instances

        else:
            print('drawing primitives')
            return glDrawArrays(self.primitive, 0, n_vertices_to_draw)



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
            print(element.glDraw(program))

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

    data = numpy.zeros(len(pointdata), dtype = [ ("position", np.float32, 3),
                                                 ("color",    np.float32, 3)] )

    data['color']    = pointcolor
    data['position'] = pointdata

    #print(data)
    # each is numpy array of 3-coordinate vectors

    linepoints = [[-0.7, -0.5, 0], [-0.5, 0.7, 0], [-0.2, 0.5, 0], [0.1,0.1,0], [0.2,0.3,0], [-0.1,0.1,0]]
    linecolor  = [    [0., 0., 1],  [0., 1., 0],    [1., 0., 0],   [1, 1, 0],   [0, 1, 1],    [1, 0, 1]]

    linedata = numpy.zeros(len(linepoints), dtype = [("position", np.float32, 3),
                                                 ("color",    np.float32, 3)] )

    linedata['color']    = linecolor
    linedata['position'] = linepoints
    # each is numpy array of 3-coordinate vectors

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

        return GlElement(GL_TRIANGLE_FAN, data, instances)

    #elements = [GlElement(GL_TRIANGLES, data), GlElement(GL_LINES, linedata)]
    #elements = [GlElement(GL_LINES, linedata)]
    #elements = [random_circles_triangles_instances(3)]
    elements = [GlElement(GL_TRIANGLES, data), GlElement(GL_LINES, linedata), random_circles_triangles_instances(3)]

    # Запускаем основной цикл
    glutMainLoop()

