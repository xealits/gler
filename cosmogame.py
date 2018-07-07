'''
simple prototype of general visual look
'''

# -*- coding: utf-8 -*-
# from 
# Импортируем все необходимые библиотеки:
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

from time import sleep
import logging
import numpy
np = numpy

from gl_elements import GlElement


g_Width  = 500
g_Height = 500

scale = 1.
shift_x = 0.
shift_y = 0.
tilt_x = 0.
tilt_y = 0.

scale_step = 1.1
shift_step = 1.0
tilt_step  = 0.01

g_fViewDistance = 100.
g_nearPlane = 1.
g_farPlane  = 10000.

zoom = -10.

# Процедура обработки специальных клавиш
def specialkeys(key, x, y):
    '''
    вызывает функцию glRotatef(градус поворода, ось_Х, ось_У, ось_З)
    '''
    print(key)
    #print(glutGetModifiers() == GLUT_ACTIVE_ALT)
    mods = glutGetModifiers()

    # Сообщаем о необходимости использовать глобального массива pointcolor
    global pointcolor, scale, shift_x, shift_y, zoom, tilt_x, tilt_y, g_fViewDistance
    # Обработчики специальных клавиш
    # scale
    if key == GLUT_KEY_PAGE_UP: # scale out
        #scale /= scale_step
        #glUniform1f(PARAM_scale, scale)
        #glTranslatef(0., 0., 0.02) # может это бы сработало? движение по Z
        # нет, оно ничего не зумит вид из камеры
        g_fViewDistance += 1
    if key == GLUT_KEY_PAGE_DOWN: # scale in
        #scale *= scale_step
        #glUniform1f(PARAM_scale, scale)
        #glTranslatef(0., 0., -0.02)
        g_fViewDistance -= 1

    shift_val = shift_step / scale
    # move around
    # Z axis
    if key == GLUT_KEY_F5:
        #glRotatef(5, 0, 0, 1)
        zoom -= 1
    elif key == GLUT_KEY_F6:
        #glRotatef(-5, 0, 0, 1)
        zoom += 1

    if key == GLUT_KEY_UP and mods == GLUT_ACTIVE_ALT:
        #glRotatef(5, 1, 0, 0)       # Вращаем на 5 градусов по оси X
        tilt_y += tilt_step
    elif key == GLUT_KEY_UP:        # Клавиша вверх
        #glTranslate(0., -0.02, 0.)
        shift_y += shift_val
        glUniform1f(PARAM_shift_Y, shift_y)
        #glUniform1f(PARAM_shift_Y, shift_y+tilt_y)

    if key == GLUT_KEY_DOWN and mods == GLUT_ACTIVE_ALT:        # Клавиша вниз
        #glRotatef(-5, 1, 0, 0)      # Вращаем на -5 градусов по оси X
        tilt_y -= tilt_step
    elif key == GLUT_KEY_DOWN:      # Клавиша вниз
        #glTranslate()
        #glTranslate(0., 0.02, 0.)
        shift_y -= shift_val
        glUniform1f(PARAM_shift_Y, shift_y)
        #glUniform1f(PARAM_shift_Y, shift_y+tilt_y)

    if key == GLUT_KEY_LEFT and mods == GLUT_ACTIVE_ALT:        # Клавиша влево
        #glRotatef(5, 0, 1, 0)       # Вращаем на 5 градусов по оси Y
        tilt_x += tilt_step
    elif key == GLUT_KEY_LEFT:        # Клавиша влево
        #glTranslate(0.02, 0., 0.)
        shift_x += shift_val
        glUniform1f(PARAM_shift_X, shift_x)
        #glUniform1f(PARAM_shift_X, shift_x+tilt_x)

    if key == GLUT_KEY_RIGHT and mods == GLUT_ACTIVE_ALT:       # Клавиша вправо
        #glRotatef(-5, 0, 1, 0)      # Вращаем на -5 градусов по оси Y
        tilt_x -= tilt_step
    elif key == GLUT_KEY_RIGHT:       # Клавиша вправо
        #glTranslate(-0.02, 0., 0.)
        shift_x -= shift_val
        glUniform1f(PARAM_shift_X, shift_x)
        #glUniform1f(PARAM_shift_X, shift_x+tilt_x)

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
    #gluLookAt(-shift_x-tilt_x, -shift_y-tilt_y, -g_fViewDistance, -shift_x, -shift_y, 0, 0, 1, 0)   #-.1,0,0
    camera_x =   np.sin(tilt_y) * np.sin(tilt_x) * abs(g_fViewDistance)
    camera_y = - np.sin(tilt_y) * np.cos(tilt_x) * abs(g_fViewDistance)
    camera_z = - np.cos(tilt_y) * abs(g_fViewDistance)
    gluLookAt(-shift_x+camera_x, -shift_y+camera_y, camera_z, -shift_x, -shift_y, 0, 0, 1, 0)   #-.1,0,0

    # Set perspective (also zoom)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(zoom, float(g_Width)/float(g_Height), g_nearPlane, g_farPlane)
    # this float(g_Width)/float(g_Height) sets the camera angle
    # and fixes the screen size/pixel coordinates issue
    glMatrixMode(GL_MODELVIEW)

    for element in elements:
        logging.debug(element.glDraw())

    glutSwapBuffers()

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
glutSpecialFunc(specialkeys)

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
#  uniform float shift_x, shift_y;
#        Vertex.x += shift_x;
#        Vertex.y += shift_y;
#        Vertex.x *= scale;
#        Vertex.y *= scale;
#        Vertex.z *= scale;
#    vec4 Vertex = vec4(position, 1.0);

#         Vertex.x += shift_x;
#         Vertex.y += shift_y;
#         Vertex.z += sqrt((Vertex.x-shift_x)*(Vertex.x-shift_x) + (Vertex.y-shift_y)*(Vertex.y-shift_y));
#         Vertex.z += ((Vertex.x-shift_x)*(Vertex.x-shift_x) + (Vertex.y-shift_y)*(Vertex.y-shift_y));
#         Vertex.z += 2.0*sqrt((Vertex.x+shift_x)*(Vertex.x+shift_x) + (Vertex.y+shift_y)*(Vertex.y+shift_y));
#         Vertex.z += 0.1*((Vertex.x+shift_x)*(Vertex.x+shift_x) + (Vertex.y+shift_y)*(Vertex.y+shift_y));
#         Vertex.z += 100.*exp(-1.0/(0.001+sqrt((Vertex.x+shift_x)*(Vertex.x+shift_x) + (Vertex.y+shift_y)*(Vertex.y+shift_y))));

# hyperbolic
#         Vertex.z += 10.0*sqrt((instance_position.x+shift_x)*(instance_position.x+shift_x) + (instance_position.y+shift_y)*(instance_position.y+shift_y));
#         Vertex.z += 10.0*sqrt((Vertex.x+shift_x)*(Vertex.x+shift_x) + (Vertex.y+shift_y)*(Vertex.y+shift_y));

# rotational projection of the hyperbolic on the surface
# and also explicit hyperbolic shader, not conic

#    float init_x = instance_position.x+shift_x;
#    float init_y = instance_position.y+shift_y;
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
#    float dist2 = ((Vertex.x+shift_x)*(Vertex.x+shift_x) + (Vertex.y+shift_y)*(Vertex.y+shift_y));

#    float dist2 = ((instance_position.x+shift_x)*(instance_position.x+shift_x) + (instance_position.y+shift_y)*(instance_position.y+shift_y));
#    float dist = sqrt(dist2);
#    float alpha = 0.15*dist/sqrt(dist2 + 16.);
#         Vertex.z += dist/tan(alpha);

#    float dist2 = ((instance_position.x+shift_x)*(instance_position.x+shift_x) + (instance_position.y+shift_y)*(instance_position.y+shift_y));
#    float dist2 = ((Vertex.x+shift_x)*(Vertex.x+shift_x) + (Vertex.y+shift_y)*(Vertex.y+shift_y));


#         Vertex.z += sqrt(Vertex.x*Vertex.x + Vertex.y*Vertex.y);

#    float dist2 = ((Vertex.x)*(Vertex.x) + (Vertex.y)*(Vertex.y));
#    float dist  = sqrt(dist2);
#    float alpha = 0.15*dist/sqrt(dist2 + 16.);
#         Vertex.z += dist/tan(alpha);

#    float dist2 = ((Vertex.x+shift_x)*(Vertex.x+shift_x) + (Vertex.y+shift_y)*(Vertex.y+shift_y));
#    float dist = sqrt(dist2);
#    float alpha = 0.15*dist/sqrt(dist2 + 16.);
#         Vertex.z += dist/tan(alpha);

#         Vertex.z += 10.0*sqrt((Vertex.x+shift_x)*(Vertex.x+shift_x) + (Vertex.y+shift_y)*(Vertex.y+shift_y));
#         Vertex.z += 10.0*sqrt((Vertex.x)*(Vertex.x) + (Vertex.y)*(Vertex.y));

vertex = create_shader(GL_VERTEX_SHADER,"""
  uniform float scale;
  uniform float shift_x, shift_y;
  attribute vec3 position;
  attribute vec3 color;
  attribute vec3 instance_position;
  varying vec3 vertex_color;
  void main()
  {
    vec4 Vertex = vec4(instance_position + position, 1.0);
         Vertex.x *= scale;
         Vertex.y *= scale;
         Vertex.z *= scale;

    float init_x = instance_position.x + shift_x + position.x;
    float init_y = instance_position.y + shift_y + position.y;
    float rad2 = init_x*init_x + init_y*init_y;
    float rad  = sqrt(rad2);
    float hyperbolic_z = 5.0*sqrt(1.0 + rad2);

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
  }""")

#    float init_x = instance_position.x+shift_x;
#    float init_y = instance_position.y+shift_y;
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

# zoom and shift
#PARAM_scale = glGetUniformLocation(vertex2, 'scale')
PARAM_scale   = glGetUniformLocation(program, 'scale')
PARAM_shift_X = glGetUniformLocation(program, 'shift_x')
PARAM_shift_Y = glGetUniformLocation(program, 'shift_y')
# set zoom scale parameter:
glUniform1f(PARAM_scale, 1.0)
glUniform1f(PARAM_shift_X, 0.)
glUniform1f(PARAM_shift_Y, 0.)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

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



    a = [-0.5,  0.0,  0.0]
    b = [ 0.0,  0.0,  0.0]
    c = [ 0.0,  0.5,  0.0]
    d = [ 0.0,  0.0,  0.5]

    canonical_tetrahedron = np.array([a, b, c,
                                      a, b, d,
                                      a, c, d,
                                      b, c, d])

    #canonical_tetrahedron = np.array([a, b, c,
    #                                  a, b, d])

    def random_tetraheders(N_instances, r_size=0.3, random_position=False, spread=[1,1,0.1]):
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
            rads   = np.random.random(N_instances)*0.1 + 5
            xis = np.cos(angles)
            yis = np.sin(angles)
            positions = np.column_stack((xis*rads, yis*rads, zis))
            instances_position['instance_position'] = positions + [0, -15, 0]

        #return GlElement(program, GL_TRIANGLE_STRIP, [data], [instances_position])
        return GlElement(program, GL_TRIANGLES, [data], [instances_position])


    # testing updates
    N_tetra = 100


    block_a   = [-0.5, -0.5, -0.5]
    block_b   = [-0.5,  0.5, -0.5]
    block_c   = [ 0.5,  0.5, -0.5]
    block_d   = [ 0.5, -0.5, -0.5]

    block_a_t = [-0.5, -0.5,  0.5]
    block_b_t = [-0.5,  0.5,  0.5]
    block_c_t = [ 0.5,  0.5,  0.5]
    block_d_t = [ 0.5, -0.5,  0.5]

    # indicies are better for this
    canonical_block = np.array([block_a,   block_b,   block_c,   block_d,
                                block_a_t, block_b_t, block_c_t, block_d_t,
                                block_a,   block_b,   block_b_t, block_a_t,
                                block_b,   block_c,   block_c_t, block_b_t,
                                block_c,   block_d,   block_d_t, block_c_t,
                                block_d,   block_a,   block_a_t, block_d_t,
                                      ])

    def random_blocks(N_instances, r_size=0.5, spread=8.):
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
        instances_position['instance_position'] = positions

        return GlElement(program, GL_QUADS, [data], [instances_position])

    #elements = [random_circles_triangles_instances(N_tetra)]
    elements = [random_tetraheders(N_tetra, 0.5, random_position=True, spread=[100,1,0.1]),
                random_blocks(1, 5, 0.1)]

    # Запускаем основной цикл
    #glutMainLoop()
    from threading import Thread
    glThread = Thread(target=glutMainLoop)
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

    glThread.start()

    ##for _ in range(10):
    #while True:
    #    sleep(1)
    #    #print('foo')
    #    # sim
    #    #circle_centers = (numpy.random.rand(N_instances, 3) - [0.5, 0.5, 0.])# * 2
    #    instance_centers = (numpy.random.rand(N_instances, 3) - [0.5, 0.5, 0.])# * 2
    #    # update
    #    elements[0]['instance_position'] = instance_centers
    #    # draw
    #    draw()





