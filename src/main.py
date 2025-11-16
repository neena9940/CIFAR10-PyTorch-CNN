from pydoc import classname

import torch
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt

data_path = '/Users/neena/Desktop/CIFAR10/data'

cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=True,
                                  transform=transforms.ToTensor())

print(len(cifar10))
print(len(cifar10_val))

img , label = cifar10_val[99]
#print(classname[label])

plt.imshow(img)
#plt.show()

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Get item 99
img, label = cifar10[99]

# Print info
print("Image type:", type(img))        # <class 'PIL.Image.Image'>
print("Label:", label)                 # e.g., 1
print("Class name:", class_names[label])  # 'automobile'

# Display image
plt.imshow(img)
plt.title(f"Class: {class_names[label]}")
plt.axis('off')  # Hide axes
plt.show()

print(dir(transforms))

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
print(img_t.shape)

img_t , _ = tensor_cifar10[99]
print(type(img_t))

imgs = torch.stack([img_t for img_t , _ in tensor_cifar10 ], dim = 3)
print(imgs.shape)

print(imgs.view(3, -1).mean(dim = 1))
print(imgs.view(3, -1).std(dim = 1))

print(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))

transformed_cifra10 = datasets.CIFAR10(data_path, train=True, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                         (0.2470, 0.2435, 0.2616))
                                                                                        ] ))

img_t , _ = transformed_cifra10[0]
plt.imshow(img_t.permute(1, 2, 0))
#plt.show()

#Building the dataset
label_map = {0:0, 2:1}
class_names = {'airplane', 'bird'}

cifar2 = [(img, label_map[label])
          for img, label in cifar10
          if label in [0,2]]
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val
              if label in [0,2]]



# softmax
import torch.nn as nn

softmax = nn.Softmax(dim=1)
x = torch.tensor([[1.0 ,2.0 ,3.0],
                [1.0 ,2.0 ,3.0]])
print(softmax(x))

# Buidling the model


n_out = 2

model = nn.Sequential(
    nn.Linear(3072, 512,),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.Softmax(dim = 1)
)


img , _ = cifar2[0]
tensor_img = to_tensor(img)
plt.imshow(tensor_img.permute(1, 2, 0))
plt.show()

img_batch = tensor_img.view(-1).unsqueeze(0)

out = model(img_batch)

print(out.shape)


