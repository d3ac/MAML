import math
import random

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def simulated_annealing(obj_func, init_temp, alpha, num_iter):
	# obj_func：目标函数
	# init_temp：目标函数
	# alpha：设顶参数
	# num_iter：迭代次数
    current_state = (random.uniform(-5, 5), random.uniform(-5, 5))
    current_energy = obj_func(*current_state)
    best_state = current_state
    best_energy = current_energy
    for i in range(num_iter):
        temperature = init_temp * math.exp(-alpha * i)
        new_state = (random.uniform(-5, 5), random.uniform(-5, 5))
        new_energy = obj_func(*new_state)
        delta_energy = new_energy - current_energy
        if delta_energy < 0:
            current_state = new_state
            current_energy = new_energy
        else:
            acceptance_prob = math.exp(-delta_energy / temperature)
            if random.random() < acceptance_prob:
                current_state = new_state
                current_energy = new_energy
        if current_energy < best_energy:
            best_state = current_state
            best_energy = current_energy

    return best_state, best_energy
# 这里调用了模拟退火算法
best_state, best_energy = simulated_annealing(rosenbrock, init_temp=100, alpha=0.01, num_iter=10000)
print(best_state)

def f(x,y):
    return (1-x)**2 + 100*(y-x**2)**2

print(f(*best_state))
print(f(1,4))
