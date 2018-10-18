import numpy as np

potential_expr = None
obstacle_expr = None

############################################
def setPotential(expr):
    '''
    设置电势、势能，这里设为0
    '''
    global potential_expr
    potential_expr = expr
    test_pot_expr()

############################################
def setObstacle(expr):
    '''
    设置障碍，这里设为"(x > 0.5 and x < 1 and not ((y > 0.25 and y < 0.75) or (y < -0.25 and y > -0.75)))"
    '''
    global obstacle_expr
    obstacle_expr = expr
    test_obs_expr() 
    print("obstacle:",obstacle_expr)

############################################
def test_pot_expr():
    '''
    异常测试
    '''
    global potential_expr
    x = 0
    y = 0
    try:
        a = eval(potential_expr)
    except:
        print(potential_expr)
        print('潜在计算错误：默认设置为0')
        potential_expr = '0'
        input('按一个键继续')

############################################
def test_obs_expr():
    '''
    异常测试
    '''
    global obstacle_expr
    x = 0
    y = 0
    try:
        a = eval(obstacle_expr)
    except e:
        print('错误设置obsatcle：设置为默认值')
        obstacle_expr = 'False'
        input('按一个键继续')

############################################
def isObstacle(x, y):
    '''
    判断是否为障碍
    '''
    a = False
    try:
        a = eval(obstacle_expr)
    except:
        pass
    return a

############################################
def getPotential(x, y):
    '''
    获取电势、势能
    '''
    a = 0 + 0j
    try:
        a = eval(potential_expr)
    except:
        pass
    return a

