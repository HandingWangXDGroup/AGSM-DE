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


import numpy as np
import torch
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from VGG16_Model import vgg
from FNS_DE import FDE

batch_size = 1
test_dataset = dsets.CIFAR10(root='/home/amax/文档/LICHAO_code/lc_code_CIFAR10_Non_target/CIFAR_data',
                                download=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                            ]),
                                train=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg().to(device)
model.load_state_dict(torch.load(r'C:\Users\LC\Desktop\AGSM-DE\AGSM-DE-PythonCode\vgg16_params.pkl',map_location=torch.device('cuda')))

count = 0
total_count = 0
net_correct = 0

def select_second_labels_images(second_label, first_label):
  sum_images = []
  for images, labels in test_loader:
      if labels == second_label:
          images = images.to(device)
          outputs = model(images)
          outputs = outputs.cpu().detach().numpy()
          if np.argmax(outputs) == second_label:
             min_value=np.min(outputs)
             outputs.itemset(second_label, min_value)
             if np.argmax(outputs) == first_label:
                target_images = torch.tensor(images)
                target_images = np.array(target_images.cpu())
                sum_images.append(target_images)
  sum_images = np.array(sum_images)
  num_images = np.size(sum_images, 0)
  if num_images > 1:
     sum_images = sum_images.squeeze()
  else:
     sum_images = sum_images.squeeze()
     sum_images = torch.tensor(sum_images)
     sum_images = sum_images.unsqueeze(0)
     sum_images = sum_images.detach().numpy()
  return sum_images

model.eval()

for images, labels in test_loader:

      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, pre = torch.max(outputs.data, 1)
      total_count += 1
      if pre == labels:
           net_correct += 1
           outputs = outputs.cpu().detach().numpy()
           min_value = np.min(outputs)
           outputs.itemset(labels, min_value)
           second_label = np.argmax(outputs)
           sum_images = select_second_labels_images(second_label, labels)
           images = FDE(images,sum_images, second_label, labels)
           images = images.to(device)
           labels = labels.to(device)
           outputs = model(images)
           _, pre = torch.max(outputs.data, 1)
           if pre == labels:
                count += 1
      acctak_count = net_correct - count
      print(total_count, net_correct, count, acctak_count)
      if net_correct > 0:
          print('Accuracy of attack: %f %%' % (100 * float(acctak_count) / net_correct))
      if net_correct ==500:
         break