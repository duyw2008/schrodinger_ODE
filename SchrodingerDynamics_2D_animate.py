
# -*- coding: utf-8 -*-
'''
求解两维薛定谔方程
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from colorsys import hls_to_rgb
import scipy as sp
import scipy.sparse
import field,toml 

hbar,m = 0, 0
size, delta_t, N, step = 0, 0, 0, 0
k_x, k_y, a_x, a_y = 0, 0, 0, 0
x0, y0 = 0, 0
x_axis, y_axis, X, Y = None, None, None, None
flag_intensity = False
wall_potential = 1e10                   # 1e10  # 势垒高度
V_x, V_y = None, None
start_time = 0
wave_function = None			# 波函数
compteur = 0
LAPLACE_MATRIX = None			# 拉普拉斯矩阵
H1 = None
HX, HY = None, None

potential_boudnary = []			# 势垒边界


####################################################
def init():
    """
    参数初始化
    """
    global hbar, m, x0, y0, x_axis, y_axis, X, Y, size, wave_function, start_time, H1, HX, HY, V_x, V_y
    config_toml = toml.load("/home/duyw/Desktop/program/schrodinger_ODE/config.toml")

    FPS = int(config_toml["FPS"])
    duration = int(config_toml["DURATION"])

    size = int(config_toml["SIZE"])
    N = int(config_toml["N"])
    delta_t = float(config_toml["DELTA_T"])/FPS

    x0 = float(config_toml['x'])
    y0 = float(config_toml['y'])

    k_x = float(config_toml["Kx"])
    k_y = float(config_toml["Ky"])
    a_x = float(config_toml["Ax"])
    a_y = float(config_toml["Ay"])

    field.setPotential(config_toml["V"])
    field.setObstacle(config_toml["O"])

    #if len(sys.argv) >= 3 and  "--intensity" in sys.argv[2:]:
    #    flag_intensity = True

    step = size/N
    frame = duration * FPS    

    x_axis = np.linspace(-size/2, size/2, N)
    y_axis = np.linspace(-size/2, size/2, N)
    X, Y = np.meshgrid(x_axis, y_axis)
    
    '''
    波函数初始化
    '''
    wave_function=makeGaussian(X,Y,x0,y0,a_x,a_y,k_x,k_y,N,step)
    
    '''
    生成field
    '''
    LAPLACE_MATRIX = sp.sparse.lil_matrix(-2*sp.sparse.identity(N*N))
    print(LAPLACE_MATRIX.shape)
    for i in range(N):
        for j in range(N-1):
            k = i*N + j
            LAPLACE_MATRIX[k,k+1] = 1

    V_x = np.zeros(N*N, dtype='c16')
    for j in range(N):
        for i in range(N):
            xx = i
            yy = N*j
            if field.isObstacle(x_axis[j], y_axis[i]):
                V_x[xx+yy] = wall_potential
            else:
                V_x[xx+yy] = field.getPotential(x_axis[j], y_axis[i])
	
    V_y = np.zeros(N*N, dtype='c16')
    for j in range(N):
        for i in range(N):
            xx = j*N
            yy = i
            if field.isObstacle(x_axis[i], y_axis[j]):
                V_y[xx+yy] = wall_potential
            else:
                V_y[xx+yy] = field.getPotential(x_axis[i], y_axis[j])

    V_x_matrix = sp.sparse.diags([V_x], [0])
    V_y_matrix = sp.sparse.diags([V_y], [0])

    LAPLACE_MATRIX = LAPLACE_MATRIX/(step ** 2)

    H1 = (1*sp.sparse.identity(N*N) - 1j*(delta_t/2)*(LAPLACE_MATRIX))
    H1 = sp.sparse.dia_matrix(H1)

    HX = (1*sp.sparse.identity(N*N) - 1j*(delta_t/2)*(LAPLACE_MATRIX - V_x_matrix))
    HX = sp.sparse.dia_matrix(HX)

    HY = (1*sp.sparse.identity(N*N) - 1j*(delta_t/2)*(LAPLACE_MATRIX - V_y_matrix))
    HY = sp.sparse.dia_matrix(HY)

    for i in range(0, N):
        for j in range(0, N):
            if field.isObstacle(x_axis[j], y_axis[i]):
                adj = getAdjPos(i, j, N)
                for xx, yy in adj:
                    if xx >= 0 and yy >= 0 and xx < N and yy <N and not field.isObstacle(x_axis[yy], y_axis[xx]):
                        potential_boudnary.append((i, j))

####################################################
def integrate(MM, N, step):
    a = 0
    air = step*step/2
    for i in range(N-1):
        for j in range(N-1):
            AA, AB, BA, BB = MM[i][j], MM[i][j+1], MM[i+1][j], MM[i+1][j+1]
            a += air*(AA+AB+BA)/3
            a += air*(BB+AB+BA)/3
    return a
####################################################
def getAdjPos(x, y, N):
	res = []
	res.append((x-1,y))
	res.append((x+1,y))
	res.append((x, y - 1))
	res.append((x,y+1))
	res.append((x - 1,y+1))
	res.append((x - 1,y-1))
	res.append((x + 1,y+1))
	res.append((x+1, y+1))
	return res
####################################################
def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + 2*r**1.2)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    c = c.swapaxes(0,1)
    return c
####################################################
# 产生二维高斯波包
def makeGaussian(X,Y,x0,y0,a_x,a_y,k_x,K_y,N,step):
    """
    Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    phase = np.exp( 1j*(X*k_x + Y*k_y))
    px = np.exp(-((x0 - X)**2)/(4*a_x**2))
    py = np.exp(-((y0 - Y)**2)/(4*a_y**2))
    wave_func = phase*px*py
    norm = np.sqrt(integrate(np.abs(wave_func)**2, N, step))
    wave_func = wave_func/norm
    return wave_func

####################################################

init()
'''
plt.figure()
plt.imshow(ga)
plt.show()
'''
rgb_map = None
if flag_intensity:
    cmap = plt.cm.inferno
    data = np.abs(wave_function)**2
    norm = plt.Normalize(data.min(), data.max())
    rgb_map = cmap(norm(data))
    rgb_map = rgb_map[:, :, :3]
else:
    rgb_map = colorize(wave_function)

for i, j in potential_boudnary:
    rgb_map[i][j] = 1, 1, 1
plt.figure()
plt.imshow(rgb_map, interpolation='none', extent=[-size/2,size/2,-size/2,size/2])
plt.show()


