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


def init_population(dim):
    population = np.zeros((population_size, dim))
    for i in range(population_size):
        for j in range(dim):
           rand_value = random.random()
           population[i,j] = xmin + rand_value * (xmax-xmin)
    return population



def calculate_fitness(taget_image, sample_adv_images, population, second_label, first_labels, dim):
    second_label = second_label
    taget_image = taget_image.cpu().detach().numpy()
    fitness = []
    function_value=np.zeros(population_size)
    attack_direction=np.zeros((population_size,3,32,32))
    for i in range(population_size):
        for j in range(0,dim):
             attack_direction[i,:,:,:] = attack_direction[i,:,:,:] + population[i,j] * ( sample_adv_images[j,:,:,:] - taget_image[0,:,:,:])
        attack_direction[i, :, :, :] = np.sign(attack_direction[i, :, :, :])

    model.eval()
    for b in range(population_size):
       attack_image = taget_image + eps * attack_direction[b, :, :, :]
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

def mutation(population, dim):

    Mpopulation = np.zeros((population_size, dim))
    for i in range(population_size):
        r1 = r2 = r3 = 0
        while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = population[r1] + F * (population[r2] - population[r3])

        for j in range(dim):
            if xmin <= Mpopulation[i, j] <= xmax:
                Mpopulation[i, j] = Mpopulation[i, j]
            else:
                Mpopulation[i,j] = xmin + random.random() * (xmax - xmin)
    return Mpopulation

def crossover(Mpopulation, population, dim):
  Cpopulation = np.zeros((population_size,dim))
  for i in range(population_size):
     for j in range(dim):
        rand_j = random.randint(0, dim - 1)
        rand_float = random.random()
        if rand_float <= CR or rand_j == j:
             Cpopulation[i, j] = Mpopulation[i, j]
        else:
             Cpopulation[i, j] = population[i, j]
  return Cpopulation

def selection(taget_image, sample_adv_images, Cpopulation, population,second_label, first_labels, dim, pfitness):
    Cfitness = calculate_fitness(taget_image, sample_adv_images,  Cpopulation,second_label, first_labels, dim)
    for i in range(population_size):
        if Cfitness[i] < pfitness[i]:
            population[i] = Cpopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i] 
    return population, pfitness

def FDE(taget_image, adversarial_images, second_label, first_labels):
    num = np.size(adversarial_images, 0)
    if num >= 10:
        dim = 10
        index = random.sample(range(0, num), dim)
        sample_adv_images = adversarial_images[index]
    else:
        dim = num
        sample_adv_images = adversarial_images

    population = init_population(dim)
    fitness = calculate_fitness(taget_image, sample_adv_images, population,second_label, first_labels, dim)
    Best_indi_index = np.argmin(fitness)
    Best_indi = population[Best_indi_index, :]
    for step in range(generations):
        if min(fitness) < 0:
           break
        Mpopulation = mutation(population, dim)
        Cpopulation = crossover(Mpopulation, population, dim)
        population, fitness = selection(taget_image, sample_adv_images, Cpopulation, population, second_label, first_labels, dim, fitness)
        Best_indi_index = np.argmin(fitness)
        Best_indi = population[Best_indi_index, :]

    Finalattack_sign = np.zeros((1, 3, 32, 32))
    taget_image = taget_image.cpu().detach().numpy()
    for j in range(0, dim):
       Finalattack_sign[0, :, :, :] = Finalattack_sign[0, :, :, :] + Best_indi[j] * (sample_adv_images[j, :, :, :] - taget_image[0, :, :, :])
    Final_direction = np.sign(Finalattack_sign)
    final_image = taget_image + eps * Final_direction
    final_image = torch.from_numpy(final_image)
    final_image = final_image.float()
    final_image[0, 0, :, :] = torch.clamp(final_image[0, 0, :, :], -1, 1)
    final_image[0, 1, :, :] = torch.clamp(final_image[0, 1, :, :], -1, 1)
    final_image[0, 2, :, :] = torch.clamp(final_image[0, 2, :, :], -1, 1)

    return final_image
