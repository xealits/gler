# -*- coding: utf-8 -*-
# from 
# Импортируем все необходимые библиотеки:
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

#import sys
# Из модуля random импортируем одноименную функцию random
from random import random
import logging
import numpy
np = numpy
import ctypes

# объявляем массив pointcolor глобальным (будет доступен во всей программе)
global pointcolor
# что он содержит?



zoom = 1.
shift_x = 0.
shift_y = 0.

zoom_step = 1.1
shift_step = 0.1

g_fViewDistance = 9.
g_Width  = 300
g_Height = 300
g_nearPlane = -1000.
g_farPlane = 1000.


# Процедура обработки специальных клавиш
def specialkeys(key, x, y):
    '''
    вызывает функцию glRotatef(градус поворода, ось_Х, ось_У, ось_З)
    '''
    # Сообщаем о необходимости использовать глобального массива pointcolor
    global pointcolor, zoom, shift_x, shift_y
    # Обработчики специальных клавиш
    # zoom
    if key == GLUT_KEY_PAGE_UP: # zoom out
        zoom /= zoom_step
        glUniform1f(PARAM_scale, zoom)
        #glTranslatef(0., 0., 0.02) # может это бы сработало? движение по Z
        # нет, оно ничего не зумит вид из камеры
    if key == GLUT_KEY_PAGE_DOWN: # zoom in
        zoom *= zoom_step
        glUniform1f(PARAM_scale, zoom)
        #glTranslatef(0., 0., -0.02)

    # move around
    if key == GLUT_KEY_UP:          # Клавиша вверх
        #glRotatef(5, 1, 0, 0)       # Вращаем на 5 градусов по оси X
        #glTranslate(0., -0.02, 0.)
        shift_y -= shift_step / zoom
        glUniform1f(PARAM_shift_Y, shift_y)
    if key == GLUT_KEY_DOWN:        # Клавиша вниз
        #glRotatef(-5, 1, 0, 0)      # Вращаем на -5 градусов по оси X
        #glTranslate()
        #glTranslate(0., 0.02, 0.)
        shift_y += shift_step / zoom
        glUniform1f(PARAM_shift_Y, shift_y)
    if key == GLUT_KEY_LEFT:        # Клавиша влево
        #glRotatef(5, 0, 1, 0)       # Вращаем на 5 градусов по оси Y
        #glTranslate(0.02, 0., 0.)
        shift_x += shift_step / zoom
        glUniform1f(PARAM_shift_X, shift_x)
    if key == GLUT_KEY_RIGHT:       # Клавиша вправо
        #glRotatef(-5, 0, 1, 0)      # Вращаем на -5 градусов по оси Y
        #glTranslate(-0.02, 0., 0.)
        shift_x -= shift_step / zoom
        glUniform1f(PARAM_shift_X, shift_x)

    #if key == GLUT_KEY_END:         # Клавиша END
    #    # Заполняем массив pointcolor случайными числами в диапазоне 0-1
    #    pointcolor = [[random(), random(), random()], [random(), random(), random()], [random(), random(), random()]]

    # asynchorous command to redraw
    glutPostRedisplay()

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

class BoolSwitch:
    states = []
    current_state = None
    def __init__(self, states):
        self.states = set(states)
        self.current_state = states[0]
    def __getattr__(self, test_state):
        #if test_state in self.states:
            return test_state == self.current_state
    def set(self, state):
        #assert state in self.states
        self.current_state = state
    def __str__(self):
        return '%s/%s' % (self.current_state, self.states)
    def __repr__(self):
        return 'BoolSwitch(%s)' % ([self.current_state] + list(self.states - {self.current_state}))

draw_shape = BoolSwitch(states=['lines', 'quads', 'triangles'])

# element = N points/vertices to draw + GL_ELEMENT
# some GL_ELEMENTs have fixed amount of vertices -- triangle, quad etc -- others don't: triangle_strip, line_strip etc
# -- drawing either in 1 DrawArrays call
# or in for loop of DrawArrays
# pack this functionality in 1 object?

known_elements = {'triangle_fan': (GL_TRIANGLE_FAN, 1),
    'triangle_strip': (GL_TRIANGLE_STRIP, 1), # actually triangle strip consumes all vertices you give it
    'quad_strip': (GL_QUAD_STRIP, 1),
    'line_strip': (GL_LINE_STRIP, 1),
    # need to figure out how to work with there ones ^
    'triangle': (GL_TRIANGLES, 3),
    'quad': (GL_QUADS, 4),
    'line': (GL_LINES, 2),
    'point': (GL_POINTS, 1)}

varied_elements = {'triangle_fan': (GL_TRIANGLE_FAN, 1),
    'triangle_strip': (GL_TRIANGLE_STRIP, 1), # actually triangle strip consumes all vertices you give it
    'quad_strip': (GL_QUAD_STRIP, 1),
    'line_strip': (GL_LINE_STRIP, 1),
    # need to figure out how to work with there ones ^
    }

const_elements ={
    'triangle': (GL_TRIANGLES, 3),
    'quad': (GL_QUADS, 4),
    'line': (GL_LINES, 2),
    'point': (GL_POINTS, 1)}


gl_elements = {'triangle_fan': GL_TRIANGLE_FAN,
    'triangle_strip': GL_TRIANGLE_STRIP,
    'quad_strip': GL_QUAD_STRIP,
    'line_strip': GL_LINE_STRIP,
    'triangle': GL_TRIANGLES,
    'quad': GL_QUADS,
    'line': GL_LINES,
    'point': GL_POINTS}

class GlElement(object):
    _const_elements = {
        GL_TRIANGLES: 3,
        GL_QUADS: 4,
        GL_LINES: 2,
        GL_POINTS: 1}

    def __init__(self, gl_element, n_vertices=None):
        if gl_element in self._const_elements:
            if n_vertices:
                assert n_vertices == self._const_elements[gl_element]
            else:
                n_vertices = self._const_elements[gl_element]
        self.element = gl_element
        self.n_vertices = n_vertices # per 1 element

    def __repr__(self):
        return 'GlElements(%s, %d)' % (repr(self.element), self.n_vertices)

    def glDraw(self, starting_array_index, n_elements):
        if self.element in self._const_elements:
            # GL_TYPE, index in the common vertices array, N vertices to draw in this command
            n_vertices_to_draw = n_elements * self.n_vertices
            glDrawArrays(self.element, starting_array_index, n_vertices_to_draw)
            starting_array_index += n_vertices_to_draw
        else:
            # if it's element of varied amount of vertices
            # draw in loop
            for _ in range(n_elements):
                n_vertices_to_draw = self.n_vertices
                glDrawArrays(self.element, starting_array_index, n_vertices_to_draw)
                starting_array_index += n_vertices_to_draw

            # with glDrawRangeElements
            #  glDrawRangeElements( GLenum ( mode ) , GLuint ( start ) , GLuint ( end ) , GLsizei ( count ) , GLenum ( type ) , const GLvoid * ( indices ) )-> void 
            #  type Specifies the type of the values in indices . Must be one of GL_UNSIGNED_BYTE , GL_UNSIGNED_SHORT , or GL_UNSIGNED_INT . 
            # from https://stackoverflow.com/questions/36508068/using-pythons-opengl-api-how-do-i-perform-gldrawelements-on-a-subset-of-the-el
            # do this:
            #print(n_elements, self.n_vertices)
            #glDrawRangeElements(self.element, starting_array_index, starting_array_index + n_elements*self.n_vertices,
            #    3*n_elements*self.n_vertices, GL_UNSIGNED_SHORT, ctypes.c_void_p(starting_array_index*2))
            ##glDrawRangeElements(self.element, starting_array_index, starting_array_index + n_elements*self.n_vertices,
            ##    3*n_elements*self.n_vertices, GL_UNSIGNED_SHORT, indices)
            #starting_array_index += n_elements*self.n_vertices

        return starting_array_index

#TODO: a class of an inhomogeneous array of elements
# т.е. нужен список элементов, который трансформируется в 1 вектор вершин и список элементов для рисовательной команды
class GlObjects(object):
    def __init__(self, elements_list):
        # elements_descr = [(GlElement, vertices), ...]
        # -- list of homogeneous elements with their vertices
        # elements_spec is [(el, n_elements), ...]
        elements_vtx  = [(el, numpy.array(vtx).reshape(-1, 3)) for el, vtx in elements_list]
        assert all(len(vtx) % el.n_vertices == 0 for el, vtx in elements_vtx) # integer number of elements
        self.elements_spec = [(el, len(vtx) // el.n_vertices) for el, vtx in elements_vtx]
        self.elements_vtx  = numpy.row_stack(vtx for _, vtx in elements_vtx)
        print(len(self.elements_vtx), sum(el.n_vertices * n_el for el, n_el in self.elements_spec))

    def flatten_gl(self):
        #return array of vertices and list of element descriptions
        # plus color
        return self.elements_vtx, self.elements_spec


canonical_circle_n = 50
canonical_circle_x = np.cos([(np.pi*2*i)/canonical_circle_n for i in range(canonical_circle_n)])
canonical_circle_y = np.sin([(np.pi*2*i)/canonical_circle_n for i in range(canonical_circle_n)])
canonical_circle_fan = np.row_stack(([0,0,0],
    np.column_stack((canonical_circle_x, canonical_circle_y, np.zeros(canonical_circle_n))),
    [canonical_circle_x[0], canonical_circle_y[0], 0]))

canonical_circle_n += 2

N_circles = 25
#circle * 5 + np.array([1,1,0])
ones_circles_drawing_spec = GlObjects([(GlElement(GL_TRIANGLE_FAN, canonical_circle_n),
    numpy.row_stack(r*canonical_circle_fan + [x,y,0] for x,y,r in zip(numpy.zeros(N_circles),
         numpy.zeros(N_circles),
         numpy.ones(N_circles)) ))])

'''
random_circles_drawing_spec = GlObjects([(GlElement(GL_TRIANGLE_FAN, canonical_circle_n),
    numpy.row_stack(r*canonical_circle_fan + [x,y,0] for x,y,r in zip((numpy.random.rand(N_circles)-0.5)*2,
        (numpy.random.rand(N_circles) - 0.5)*2,
         numpy.random.rand(N_circles)*0.3) ))])
'''

# trianlges-circle

canonical_circle_n = 50
canonical_circle_x = np.cos([(np.pi*2*i)/canonical_circle_n for i in range(canonical_circle_n+1)])
canonical_circle_y = np.sin([(np.pi*2*i)/canonical_circle_n for i in range(canonical_circle_n+1)])
canonical_circle_fan = np.array([pt for i in range(1, canonical_circle_n+1) for pt in
   ([0,0,0],
    [canonical_circle_x[i],   canonical_circle_y[i],   0],
    [canonical_circle_x[i-1], canonical_circle_y[i-1], 0])])


#print(canonical_circle_fan)

#np.row_stack(([0,0,0],
#    np.column_stack((canonical_circle_x, canonical_circle_y, np.zeros(canonical_circle_n))),
#    [canonical_circle_x[0], canonical_circle_y[0], 0]))

circle_colors = [[r,g,b] for r,g,b in numpy.random.rand(N_circles, 3) for _ in range(canonical_circle_n)]

def random_circles(N_circles, r_size=0.3):
    x_y_r = zip((numpy.random.rand(N_circles)-0.5)*2,
        (numpy.random.rand(N_circles) - 0.5)*2,
         numpy.random.rand(N_circles)*r_size)
    circle_colors = [[r,g,b] for r,g,b in numpy.random.rand(N_circles, 3) for _ in range(canonical_circle_n*3)]
    return GlObjects([(GlElement(GL_TRIANGLES),
        numpy.row_stack(r*canonical_circle_fan + [x,y,0] for x,y,r in x_y_r))]), circle_colors

random_circles_drawing_spec, circle_colors = random_circles(20, 0.1)




# Процедура перерисовки
def draw():
    '''
    т.е. это вызывается Н раз в секунду?
    не ясно как здесь с производительностью? про переинициализируется? какие стркутуры данных?
    например "включаем использование массива вершин" -- исполняется Н раз в секунду?
    можно ли и стоит ли исполнять его 1 раз перед началом программы?
    '''
    glClear(GL_COLOR_BUFFER_BIT)                    # Очищаем экран и заливаем серым цветом
    #gluPerspective(zoom, float(g_Width)/float(g_Height), g_nearPlane, g_farPlane)
    # when to run it?
    #glLoadIdentity()
    #gluLookAt(0, 0, -g_fViewDistance, 0, 0, 0, -.1, 0, 0)   #-.1,0,0
    #glMatrixMode(GL_PROJECTION)
    #glLoadIdentity()
    #gluPerspective(zoom, float(g_Width)/float(g_Height), g_nearPlane, g_farPlane)
    #glMatrixMode(GL_MODELVIEW)
    # the shader is not drawn....

    glEnableClientState(GL_VERTEX_ARRAY)            # Включаем использование массива вершин
    glEnableClientState(GL_COLOR_ARRAY)             # Включаем использование массива цветов
    # Указываем, где взять массив верши:
    # Первый параметр - сколько используется координат на одну вершину
    # Второй параметр - определяем тип данных для каждой координаты вершины
    # Третий парметр - определяет смещение между вершинами в массиве
    # Если вершины идут одна за другой, то смещение 0
    # Четвертый параметр - указатель на первую координату первой вершины в массиве
    glVertexPointer(3, GL_FLOAT, 0, pointdata)

    indices = vbo.VBO(np.array(range(len(pointdata)), dtype=numpy.uint16), target=GL_ELEMENT_ARRAY_BUFFER)
    indices.bind()

    # Указываем, где взять массив цветов:
    # Параметры аналогичны, но указывается массив цветов
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    # Рисуем данные массивов за один проход:
    # Первый параметр - какой тип примитивов использовать (треугольники, точки, линии и др.)
    # Второй параметр - начальный индекс в указанных массивах
    # Третий параметр - количество рисуемых объектов (в нашем случае это 3 вершины - 9 координат)

    #if draw_shape.triangles:
    #    glDrawArrays(GL_TRIANGLES, 0, 3*(len(pointdata) // 3))
    #elif draw_shape.lines:
    #    glDrawArrays(GL_LINES, 0, 2*(len(pointdata) // 2))
    #elif draw_shape.quads:
    #    glDrawArrays(GL_QUADS, 0, 4*(len(pointdata) // 4))
    #else:
    #    glDrawArrays(GL_POINTS, 0, len(pointdata))

    #glPointSize(5)

    initial_point = 0
    '''
    for element, l in pointelements:
        gl_object, n_points = known_elements[element] # catch keyerror exception?
        n_vertices_to_draw = n_points*l
        glDrawArrays(gl_object, initial_point, n_vertices_to_draw)
        initial_point += n_vertices_to_draw
        '''
    for element, n_elements in pointelements:
        initial_point += element.glDraw(initial_point, n_elements)

    glDisableClientState(GL_VERTEX_ARRAY)           # Отключаем использование массива вершин
    glDisableClientState(GL_COLOR_ARRAY)            # Отключаем использование массива цветов

    indices.unbind()

    glutSwapBuffers()                               # Выводим все нарисованное в памяти на экран
    #glutPostRedisplay() # this should redraw the window
    # it doesn't and then it makes blinky lines..

def reshape(width, height):
    global g_Width, g_Height
    g_Width = width
    g_Height = height
    glViewport(0, 0, g_Width, g_Height)


timer_tick = 50 # ms

def glutTimer(value):
    ''' the 2 second timer will update the window as in the example:
    glutTimerFunc(1, glutTimer, 1);
    void glutTimer(int value)
    {
    glutPostRedisplay();
    glutTimerFunc(1, glutTimer, 1);
    }
    '''
    glutPostRedisplay();
    glutTimerFunc(timer_tick, glutTimer, 1);

from threading import Thread
# ok, maybe use
#glThread = threading.Thread(target=runGl)
# -- yep, that's it




# т.е. делаем в этом модуле такой С-объект

# Определяем массив вершин (три вершины по три координаты)
pointdata = [[0, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.4, -0.1, 0], [0., 0.7, 0], [-0.2, 0.5, 0], [0.1,0.1,0], [0.2,0.3,0], [-0.1,0.1,0]]
# Определяем массив цветов (по одному цвету для каждой вершины)
pointcolor = [[1, 1, 0], [0, 1, 1], [1, 0, 1], [0., 0., 1], [0., 1., 0], [1., 0., 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
pointelements = [('line', 3), ('point', 3)]
#pointelements = {'line': 3, 'point': 3} # need sorted dict, let's stick to just tuples

lines_vtx = [[0, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.4, -0.1, 0], [0., 0.7, 0], [-0.2, 0.5, 0]]
#lines  = GlElements(GL_LINES, lines_vtx)
points_vtx = [[0.1,0.1,0], [0.2,0.3,0], [-0.1,0.1,0]]
#points = GlElements(GL_POINTS, points_vtx)

drawing_spec = GlObjects([(GlElement(GL_LINES), lines_vtx), (GlElement(GL_POINTS), points_vtx)])

# circles
drawing_spec = ones_circles_drawing_spec # random_circles_drawing_spec
drawing_spec = random_circles_drawing_spec
pointcolor = circle_colors

#print(repr(drawing_spec.elements_vtx))
print(drawing_spec.elements_vtx.shape)
print(drawing_spec.elements_spec)
pointdata, pointelements = drawing_spec.elements_vtx, drawing_spec.elements_spec

N_lines = 3
N_points = 3

# Здесь начинется выполнение программы
def gl_window_program():
    global PARAM_scale, PARAM_shift_Y, PARAM_shift_X

    # Использовать двойную буферезацию и цвета в формате RGB (Красный Синий Зеленый)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    # Указываем начальный размер окна (ширина, высота)
    glutInitWindowSize(g_Width, g_Height)
    # Указываем начальное
    # положение окна относительно левого верхнего угла экрана
    glutInitWindowPosition(50, 50)
    # Инициализация OpenGl
    glutInit(sys.argv)
    # Создаем окно с заголовком "Shaders!"
    glutCreateWindow(b"Shaders!")
    # reshape function -- reacts to window reshape
    glutReshapeFunc(reshape)
    # Определяем процедуру, отвечающую за перерисовку
    glutDisplayFunc(draw)
    # Определяем процедуру, выполняющуюся при "простое" программы
    #glutIdleFunc(draw)
    # the timer redraws the window every timer_tick ms
    # should be less consuming
    #glutTimerFunc(timer_tick, glutTimer, 1); # -- doesn't work at all, also blinks?
    # -- now the keys should asynchorously command to redraw
    # Определяем процедуру, отвечающую за обработку клавиш
    glutSpecialFunc(specialkeys)
    # Задаем серый цвет для очистки экрана
    glClearColor(0.2, 0.2, 0.2, 1)

    # НЕЯСНО: что тут происходит? фрагментные шейдер? что за структура у програм граф. карт?
    # Создаем вершинный шейдер:
    # Положение вершин не меняется
    # Цвет вершины - такой же как и в массиве цветов
    vertex_vanila = create_shader(GL_VERTEX_SHADER, """
    uniform float scale;
    varying vec4 vertex_color;
                void main(){
                    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                    gl_PointSize = 5.0;
                    vertex_color = gl_Color;
                }""")

    vertex = create_shader(GL_VERTEX_SHADER, """
    uniform float scale, shift_x, shift_y;
    varying vec4 vertex_color;
                void main(){
                    vec4 Vertex = gl_Vertex;
                        Vertex.x += shift_x;
                        Vertex.y += shift_y;
                        Vertex.x *= scale;
                        Vertex.y *= scale;
                        Vertex.z *= scale;
                    gl_Position = gl_ModelViewProjectionMatrix * Vertex;
                    gl_PointSize = 5.0;
                    vertex_color = gl_Color;
                }""")

    # Создаем фрагментный шейдер:
    # Определяет цвет каждого фрагмента как "смешанный" цвет его вершин
    fragment = create_shader(GL_FRAGMENT_SHADER, """
    varying vec4 vertex_color;
                void main() {
                    gl_FragColor = vertex_color;
    }""")

    # Создаем пустой объект шейдерной программы
    program = glCreateProgram()
    # Приcоединяем вершинный шейдер к программе
    glAttachShader(program, vertex)
    # Присоединяем фрагментный шейдер к программе
    glAttachShader(program, fragment)
    # "Собираем" шейдерную программу
    glLinkProgram(program)
    # Сообщаем OpenGL о необходимости использовать данную шейдерну программу при отрисовке объектов
    glUseProgram(program)


    # magic
    #index_buffer = vbo.VBO(numpy.array(range(len(pointdata)), dtype=numpy.uint16), target=GL_ELEMENT_ARRAY_BUFFER)
    #index_buffer.bind()
    #glEnableVertexAttribArray(0);
    #glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)


    # no interpreter process kill on window close
    #glut.glutSetOption(glut.GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION) 
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION) 

    #PARAM_scale = glGetUniformLocation(vertex2, 'scale')
    PARAM_scale = glGetUniformLocation(program, 'scale')
    PARAM_shift_X = glGetUniformLocation(program, 'shift_x')
    PARAM_shift_Y = glGetUniformLocation(program, 'shift_y')
    # set zoom scale parameter:
    glUniform1f(PARAM_scale, 1.0)
    glUniform1f(PARAM_shift_X, 0.)
    glUniform1f(PARAM_shift_Y, 0.)

    # Запускаем основной цикл
    glutMainLoop()

# перерисовка окна
#glutPostRedisplay()


glThread = Thread(target=gl_window_program)

from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')
    #p = Process(target=f, args=('bob',))
    #p.start()
    #p.join()
    glThread.start()


