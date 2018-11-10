'''
simple prototype of general visual look

it does:
* window
* user inputs
* sets up shader program and returns it for attaching vaos with primitives
'''

# -*- coding: utf-8 -*-
# from 
# Импортируем все необходимые библиотеки:
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

import logging
import numpy
np = numpy

#from time import sleep
#import ctypes
#from ctypes import CDLL, cdll

#from gl_elements import GlElement

g_Width  = 1000
g_Height = 800

scale = 1.

camera_position_x = 0.
camera_position_y = 0.
camera_tilt_x = 0.
camera_tilt_y = 0.

'''
camera position is defined by
shift, tilt and g_fViewDistance

shift and scale go into shader too
  uniform float scale;
  uniform float shift_x, shift_y;
the shader calculates the warp according to the camera's position

-- rename shift into camera_position
'''

scale_step = 1.1
shift_step = 1.0
shift_val  = shift_step / scale
tilt_step  = 0.01

g_fViewDistance = 100.
g_nearPlane = 1.
g_farPlane  = 10000.

zoom = -10.

# DRAWING PROCEDURES (COMMANDS FOR GL)

# Процедура обработки специальных клавиш

# separate key-handler functions, the hooks for custom ui handling
def handle_key_up():
    global camera_position_y
    #glTranslate(0., -0.02, 0.)
    camera_position_y += shift_val
    glUniform1f(PARAM_center_y, camera_position_y)
    #glUniform1f(PARAM_center_y, camera_position_y+camera_tilt_y)

    # update the ship
    #ship_element['instance_position'] = [-camera_position_x, -camera_position_y, ship_z]

def handle_key_down():
    global camera_position_y
    #glTranslate()
    #glTranslate(0., 0.02, 0.)
    camera_position_y -= shift_val
    glUniform1f(PARAM_center_y, camera_position_y)
    #glUniform1f(PARAM_center_y, camera_position_y+camera_tilt_y)

    # update the ship
    #ship_element['instance_position'] = [-camera_position_x, -camera_position_y, ship_z]

def handle_key_left():
    global camera_position_x
    #glTranslate(0.02, 0., 0.)
    camera_position_x += shift_val
    glUniform1f(PARAM_center_x, camera_position_x)
    #glUniform1f(PARAM_center_x, camera_position_x+camera_tilt_x)

    # update the ship
    #ship_element['instance_position'] = [-camera_position_x, -camera_position_y, ship_z]

def handle_key_right():
    global camera_position_x
    #glTranslate(-0.02, 0., 0.)
    camera_position_x -= shift_val
    glUniform1f(PARAM_center_x, camera_position_x)
    #glUniform1f(PARAM_center_x, camera_position_x+camera_tilt_x)

    # update the ship
    #ship_element['instance_position'] = [-camera_position_x, -camera_position_y, ship_z]


# TODO: what are 'x' and 'y' inputs?
def default_specialkeys(key, x, y):
    '''
    вызывает функцию glRotatef(градус поворода, ось_Х, ось_У, ось_З)
    '''
    #print(key)
    #print(glutGetModifiers() == GLUT_ACTIVE_ALT)
    mods = glutGetModifiers()

    # Сообщаем о необходимости использовать глобального массива pointcolor
    global pointcolor, scale, camera_position_x, camera_position_y, zoom, camera_tilt_x, camera_tilt_y, g_fViewDistance, shift_val
    # Обработчики специальных клавиш
    # scale
    if key == GLUT_KEY_PAGE_UP: # scale out
        #scale /= scale_step
        #glUniform1f(PARAM_scale, scale)
        #glTranslatef(0., 0., 0.02) # может это бы сработало? движение по Z
        # нет, оно ничего не зумит вид из камеры
        g_fViewDistance += 1
        zoom += 0.01
    if key == GLUT_KEY_PAGE_DOWN: # scale in
        #scale *= scale_step
        #glUniform1f(PARAM_scale, scale)
        #glTranslatef(0., 0., -0.02)
        g_fViewDistance -= 1
        zoom -= 0.01

    shift_val = shift_step / scale
    # move around
    # Z axis
    if key == GLUT_KEY_F5:
        #glRotatef(5, 0, 0, 1)
        zoom -= 1
    elif key == GLUT_KEY_F6:
        #glRotatef(-5, 0, 0, 1)
        zoom += 1
    elif key == GLUT_KEY_F7:
        scale /= scale_step
        glUniform1f(PARAM_scale, scale)
    elif key == GLUT_KEY_F8:
        scale *= scale_step
        glUniform1f(PARAM_scale, scale)

    if key == GLUT_KEY_UP and mods == GLUT_ACTIVE_ALT:
        #glRotatef(5, 1, 0, 0)       # Вращаем на 5 градусов по оси X
        camera_tilt_y += tilt_step
    elif key == GLUT_KEY_UP:        # Клавиша вверх
        handle_key_up()

    if key == GLUT_KEY_DOWN and mods == GLUT_ACTIVE_ALT:        # Клавиша вниз
        #glRotatef(-5, 1, 0, 0)      # Вращаем на -5 градусов по оси X
        camera_tilt_y -= tilt_step
    elif key == GLUT_KEY_DOWN:      # Клавиша вниз
        handle_key_down()

    if key == GLUT_KEY_LEFT and mods == GLUT_ACTIVE_ALT:        # Клавиша влево
        #glRotatef(5, 0, 1, 0)       # Вращаем на 5 градусов по оси Y
        camera_tilt_x += tilt_step
    elif key == GLUT_KEY_LEFT:        # Клавиша влево
        handle_key_left()

    if key == GLUT_KEY_RIGHT and mods == GLUT_ACTIVE_ALT:       # Клавиша вправо
        #glRotatef(-5, 0, 1, 0)      # Вращаем на -5 градусов по оси Y
        camera_tilt_x -= tilt_step
    elif key == GLUT_KEY_RIGHT:       # Клавиша вправо
        handle_key_right()

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

def reshape(width, height):
    global g_Width, g_Height
    g_Width = width
    g_Height = height
    glViewport(0, 0, g_Width, g_Height)

# the gl_elements to draw on the scene
elements = []

def draw():
    #glClear(GL_COLOR_BUFFER_BIT)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    # http://carloluchessa.blogspot.pt/2012/09/simple-viewer-in-pyopengl.html
    # Set up viewing transformation, looking down -Z axis
    glLoadIdentity()
    #gluLookAt(0, 0, -g_fViewDistance, 0, 0, 0, -.1, 0, 0)   #-.1,0,0
    # +Z
    #gluLookAt(0, 0, -g_fViewDistance, 0, 0, 0, .1, 0, 0)   #-.1,0,0
    #gluLookAt(0, 0, -g_fViewDistance, 0, 0, 0, 0, 1, 0)   #-.1,0,0
    # from opengl-tutorial:
    # frist 3  -- camera position
    # second 3 -- camera target, where it looks at -- the point/postition, where it looks at
    # last 3   -- up vector, it's upside down if inverted,
    #             and they suggest 0, 1, 0... -- which means "up is positive Y"
    #             the "up" is the top of the window where the scene is rendered

    # and moving the camera:
    #gluLookAt(-camera_position_x-camera_tilt_x, -camera_position_y-camera_tilt_y, -g_fViewDistance, -camera_position_x, -camera_position_y, 0, 0, 1, 0)   #-.1,0,0
    camera_x =   np.sin(camera_tilt_y) * np.sin(camera_tilt_x) * abs(g_fViewDistance)
    camera_y = - np.sin(camera_tilt_y) * np.cos(camera_tilt_x) * abs(g_fViewDistance)
    camera_z = - np.cos(camera_tilt_y) * abs(g_fViewDistance)
    gluLookAt(-camera_position_x+camera_x, -camera_position_y+camera_y, camera_z, -camera_position_x, -camera_position_y, 0, 0, 1, 0)   #-.1,0,0

    # Set perspective (also zoom)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(zoom, float(g_Width)/float(g_Height), g_nearPlane, g_farPlane)
    # this float(g_Width)/float(g_Height) sets the camera angle
    # and fixes the screen size/pixel coordinates issue
    glMatrixMode(GL_MODELVIEW)

    for element in elements:
        # if element.isOn?
        logging.debug(element.glDraw())

    glutSwapBuffers()


# WINDOW AND GL INITIALIZATION

# Использовать двойную буферезацию и цвета в формате RGB (Красный Синий Зеленый)
#glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB) #  overlaps are wrong
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)

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
glutSpecialFunc(default_specialkeys)

# Задаем серый цвет для очистки экрана
glClearColor(0.2, 0.2, 0.2, 1)
# and some more stuff on handling depth:
# https://paroj.github.io/gltut/Positioning/Tut05%20Overlap%20and%20Depth%20Buffering.html
glEnable(GL_DEPTH_TEST);
glDepthMask(GL_TRUE);
glDepthFunc(GL_LEQUAL);
glDepthRange(0., 1.);
glClearDepth(1.);
# and clear depth:
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

# новый шейдер
# с атрибутами вершин
# не знаю как оно отличается от предыдущего -- там не явно както всё передаётся
# тут в атрибуты явно загонится буфер с вершинами для отрисовки
#
# только для зума нужно таки использовать ту проджекшн матрикс
#  uniform float scale;
#  uniform float center_x, center_y;
#        Vertex.x += center_x;
#        Vertex.y += center_y;
#        Vertex.x *= scale;
#        Vertex.y *= scale;
#        Vertex.z *= scale;
#    vec4 Vertex = vec4(position, 1.0);

#         Vertex.x += center_x;
#         Vertex.y += center_y;
#         Vertex.z += sqrt((Vertex.x-center_x)*(Vertex.x-center_x) + (Vertex.y-center_y)*(Vertex.y-center_y));
#         Vertex.z += ((Vertex.x-center_x)*(Vertex.x-center_x) + (Vertex.y-center_y)*(Vertex.y-center_y));
#         Vertex.z += 2.0*sqrt((Vertex.x+center_x)*(Vertex.x+center_x) + (Vertex.y+center_y)*(Vertex.y+center_y));
#         Vertex.z += 0.1*((Vertex.x+center_x)*(Vertex.x+center_x) + (Vertex.y+center_y)*(Vertex.y+center_y));
#         Vertex.z += 100.*exp(-1.0/(0.001+sqrt((Vertex.x+center_x)*(Vertex.x+center_x) + (Vertex.y+center_y)*(Vertex.y+center_y))));

# hyperbolic
#         Vertex.z += 10.0*sqrt((instance_position.x+center_x)*(instance_position.x+center_x) + (instance_position.y+center_y)*(instance_position.y+center_y));
#         Vertex.z += 10.0*sqrt((Vertex.x+center_x)*(Vertex.x+center_x) + (Vertex.y+center_y)*(Vertex.y+center_y));

# rotational projection of the hyperbolic on the surface
# and also explicit hyperbolic shader, not conic

#    float init_x = instance_position.x+center_x;
#    float init_y = instance_position.y+center_y;
#    float rad2 = init_x*init_x + init_y*init_y;
#    float rad  = sqrt(rad2);
#    float hyperbolic_z   = 10.0*sqrt(5.0 + rad2);
#    float slope_tan  = rad/hyperbolic_z; // dz/dr = r/z
#    float slope_tan2 = slope_tan * slope_tan;
#    float slope_cos = 1 / sqrt(1+slope_tan2);
#    float slope_sin = slope_tan / sqrt(1+slope_tan2);
#    float rot_z = Vertex.z * slope_cos;
#    float rot_r = Vertex.z * slope_sin;
#    Vertex.z = hyperbolic_z - rot_z;
#    rot_x = rot_r * init_x / rad;
#    rot_y = rot_r * init_y / rad;
#    Vertex.x += rot_x;
#    Vertex.y += rot_y;


# angular
#    float dist2 = ((Vertex.x+center_x)*(Vertex.x+center_x) + (Vertex.y+center_y)*(Vertex.y+center_y));

#    float dist2 = ((instance_position.x+center_x)*(instance_position.x+center_x) + (instance_position.y+center_y)*(instance_position.y+center_y));
#    float dist = sqrt(dist2);
#    float alpha = 0.15*dist/sqrt(dist2 + 16.);
#         Vertex.z += dist/tan(alpha);

#    float dist2 = ((instance_position.x+center_x)*(instance_position.x+center_x) + (instance_position.y+center_y)*(instance_position.y+center_y));
#    float dist2 = ((Vertex.x+center_x)*(Vertex.x+center_x) + (Vertex.y+center_y)*(Vertex.y+center_y));


#         Vertex.z += sqrt(Vertex.x*Vertex.x + Vertex.y*Vertex.y);

#    float dist2 = ((Vertex.x)*(Vertex.x) + (Vertex.y)*(Vertex.y));
#    float dist  = sqrt(dist2);
#    float alpha = 0.15*dist/sqrt(dist2 + 16.);
#         Vertex.z += dist/tan(alpha);

#    float dist2 = ((Vertex.x+center_x)*(Vertex.x+center_x) + (Vertex.y+center_y)*(Vertex.y+center_y));
#    float dist = sqrt(dist2);
#    float alpha = 0.15*dist/sqrt(dist2 + 16.);
#         Vertex.z += dist/tan(alpha);

#         Vertex.z += 10.0*sqrt((Vertex.x+center_x)*(Vertex.x+center_x) + (Vertex.y+center_y)*(Vertex.y+center_y));
#         Vertex.z += 10.0*sqrt((Vertex.x)*(Vertex.x) + (Vertex.y)*(Vertex.y));

#         Vertex.x *= scale;
#         Vertex.y *= scale;
#         Vertex.z *= scale;

# MAKING SHADER PROGRAM

std_flat_shader = """
  uniform float scale;
  uniform float center_x, center_y;
  attribute vec3 position;
  attribute vec3 color;
  attribute vec3 instance_position;
  varying vec3 vertex_color;
  void main()
  {
    vec4 Vertex = vec4(instance_position + position, 1.0);

    gl_Position = gl_ModelViewProjectionMatrix * Vertex;
    vertex_color = color;
  }"""

hyperbolic_shader = """
  uniform float scale;
  uniform float center_x, center_y;
  attribute vec3 position;
  attribute vec3 color;
  attribute vec3 instance_position;
  varying vec3 vertex_color;
  void main()
  {
    vec4 Vertex = vec4(instance_position + position, 1.0);

    float init_x = instance_position.x + center_x + position.x;
    float init_y = instance_position.y + center_y + position.y;
    float rad2 = init_x*init_x + init_y*init_y;
    float rad  = sqrt(rad2);
    float hyperbolic_z = 6.0*sqrt(1.0 + rad2);

    float slope_tan  = rad / hyperbolic_z;
    float slope_tan2 = slope_tan * slope_tan;
    float slope_cos =       1.0 / sqrt(1.0 + slope_tan2);
    float slope_sin = slope_tan / sqrt(1.0 + slope_tan2);
    float rot_z = - ( Vertex.z) * slope_cos;
    float rot_r =   ( Vertex.z) * slope_sin;
    float rot_x = rot_r * init_x / rad;
    float rot_y = rot_r * init_y / rad;

    Vertex.x -= rot_x;
    Vertex.y -= rot_y;
    Vertex.z += hyperbolic_z - 3.0*rot_z;

    gl_Position = gl_ModelViewProjectionMatrix * Vertex;
    vertex_color = color;
  }"""

#    float init_x = instance_position.x+center_x;
#    float init_y = instance_position.y+center_y;
#    float slope_tan  = rad / hyperbolic_z;
#    float slope_tan2 = slope_tan * slope_tan;
#    float slope_cos = 1.0 / sqrt(1.0+slope_tan2);
#    float slope_sin = slope_tan / sqrt(1.0+slope_tan2);
#    float rot_z = Vertex.z * slope_cos;
#    float rot_r = Vertex.z * slope_sin;
#    float rot_x = rot_r * init_x / rad;
#    float rot_y = rot_r * init_y / rad;

#    Vertex.x += rot_x;
#    Vertex.y += rot_y;
#    Vertex.z += hyperbolic_z - rot_z;

std_fragment_shader = """
varying vec3 vertex_color;
void main() {
               gl_FragColor = vec4(vertex_color, 1.0);
}"""


def compile_and_load_program(shader_text=std_flat_shader, fragment_shader_text=std_fragment_shader):

    vertex = create_shader(GL_VERTEX_SHADER, shader_text)

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

    # attach user interface to corresponding parameters
    # zoom and shift
    #PARAM_scale = glGetUniformLocation(vertex2, 'scale')
    global PARAM_scale  
    global PARAM_center_x
    global PARAM_center_y
    PARAM_scale   = glGetUniformLocation(program, 'scale')
    PARAM_center_x = glGetUniformLocation(program, 'center_x')
    PARAM_center_y = glGetUniformLocation(program, 'center_y')
    # set zoom scale parameter:
    glUniform1f(PARAM_scale, 1.0)
    glUniform1f(PARAM_center_x, 0.)
    glUniform1f(PARAM_center_y, 0.)

    return program

