#    Authors:    Chao Li, Handing Wang, Jun Zhang, Wen Yao, Tingsong Jiang
#    Xidian University, China
#    Defense Innovation Institute, Chinese Academy of Military Science, China
#    EMAIL:      lichaoedu@126.com, hdwang@xidian.edu.cn
#    DATE:       February 2022
# ------------------------------------------------------------------------
# This code is part of the program that produces the results in the following paper:
#
# Chao Li, Handing Wang, Jun Zhang, Wen Yao, Tingsong Jiang, An Approximated Gradient Sign Method Using Differential Evolution For Black-box Adversarial Attack, IEEE Transactions on Evolutionary Computation, 2022.
#
# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------


import random
import torch
import copy
import numpy as np
from VGG16_Model import vgg


population_size = 50
generations = 100
F = 0.5
CR = 0.6
xmin = -1
xmax = 1
eps = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg().to(device)
model.load_state_dict(torch.load(r'C:\Users\LC\Desktop\AGSM-DE\AGSM-DE-PythonCode\vgg16_params.pkl', map_location=torch.device('cuda')))
model.eval()

def init_population(num):
    population = np.zeros((population_size, num))
    for i in range(population_size):
        for j in range(num):
           rand_value = random.random()
           population[i, j] = xmin + rand_value * (xmax-xmin)
    return population



def calculate_fitness(taget_image, adversarial_images, population, num, second_label, first_labels):
    second_label = second_label
    taget_image = taget_image.cpu().detach().numpy()
    fitness = []
    function_value = np.zeros(population_size)
    sign_images = np.zeros((population_size, 3, 32, 32))

    for i in range(population_size):
        for j in range (0, num):
           sign_images[i, :, :, :] += population[i, j] * (adversarial_images[j, :, :, :] - taget_image[0, :, :, :])
        sign_images[i, :, :, :] = np.sign(sign_images[i, :, :, :])

    for b in range(population_size):
       attack_image = taget_image + eps * sign_images[b, :, :, :]
       attack_image = torch.from_numpy(attack_image)
       attack_image = attack_image.to(device)
       outputs = model(attack_image.float())
       outputs = outputs.cpu().detach().numpy()
       d = outputs[0, first_labels]
       c = np.min(outputs)
       outputs.itemset(first_labels, c)
       g = np.max(outputs)
       function_value[b] = d-g
       fitness.append(function_value[b])

    return fitness


def best_value(fitness, G, D, population):
    min_fitness = fitness[0]
    min_individual = population[0]
    G_value = G[0]
    D_value = D[0]
    for i in range(population_size):
        if fitness[i] < min_fitness:
            min_fitness = fitness[i]
            min_individual = population[i]
            G_value = G[i]
            D_value = D[i]
    return min_individual, min_fitness, G_value, D_value

def mutation(subpopulation, optimization_dim):

    Mpopulation=np.zeros((population_size, optimization_dim))
    for i in range(population_size):
        r1 = r2 = r3 = 0
        while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = subpopulation[r1] + F * (subpopulation[r2] - subpopulation[r3])

        for j in range(0,optimization_dim):
            if xmin <= Mpopulation[i, j] <= xmax:
                Mpopulation[i, j] = Mpopulation[i, j]
            else:
                Mpopulation[i, j] = xmin + random.random() * (xmax - xmin)
    return Mpopulation

def crossover(Mpopulation, subpopulation, Spopulation, index, optimization_dim):
  Cpopulation = np.zeros((population_size, optimization_dim))
  for i in range(population_size):
     for j in range(0, optimization_dim):
        rand_j = random.randint(0, optimization_dim - 1)
        rand_float = random.random()
        if rand_float <= CR or rand_j == j:
             Cpopulation[i, j] = Mpopulation[i, j]
        else:
             Cpopulation[i, j] = subpopulation[i, j]
  Spopulation[0:population_size, index] = Cpopulation
  return Spopulation

def selection(taget_image, adversarial_images, Spopulation, population,num, second_label, first_labels, pfitness):
    Cfitness = calculate_fitness(taget_image, adversarial_images, Spopulation, num, second_label, first_labels)
    for i in range(population_size):
        if Cfitness[i] < pfitness[i]:
            population[i] = Spopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i]
    return population, pfitness

def LDE(taget_image, adversarial_images, second_label, first_labels):

    num = np.size(adversarial_images, 0)
    population = init_population(num)
    fitness = calculate_fitness(taget_image, adversarial_images, population, num, second_label, first_labels)
    Best_indi_index = np.argmin(fitness)
    Best_indi = population[Best_indi_index, :]
    optimization_dim = 10
    if num <= optimization_dim:
        optimization_dim = num
        index = random.sample(range(0, num), optimization_dim)
    else:
        index = random.sample(range(0, num), optimization_dim)

    for step in range(generations):
        if min(fitness) < 0:
            break
        Spopulation = copy.deepcopy(population)
        subpopulation = copy.deepcopy(population[0:population_size, index])
        Mpopulation = mutation(subpopulation, optimization_dim)
        Spopulation = crossover(Mpopulation, subpopulation, Spopulation, index, optimization_dim)
        population, fitness = selection(taget_image, adversarial_images, Spopulation, population, num, second_label, first_labels, fitness)
        Best_indi_index = np.argmin(fitness)
        Best_indi = population[Best_indi_index, :]
        index = random.sample(range(0, num), optimization_dim)

    attack_sign = np.zeros((1, 3, 32, 32))
    taget_image = taget_image.cpu().detach().numpy()
    for j in range(0, num):
       attack_sign[0, :, :, :] += minindividual[j] * (adversarial_images[j, :, :, :] - taget_image[0, :, :, :])
    attack_direction = np.sign(attack_sign)
    final_image = taget_image + eps * attack_direction
    final_image = torch.from_numpy(final_image)
    final_image = final_image.float()
    final_image[0, 0, :, :] = torch.clamp(final_image[0, 0, :, :], -1, 1)
    final_image[0, 1, :, :] = torch.clamp(final_image[0, 1, :, :], -1, 1)
    final_image[0, 2, :, :] = torch.clamp(final_image[0, 2, :, :], -1, 1)
    return final_image
