import torch
import random
import math
input_layer = 2
hidden_layer = 16
output_layer = 2
torch.manual_seed(42)
random.seed(42)
x = torch.tensor([40, 120])
w_input = torch.rand(hidden_layer, input_layer)
w_output = torch.rand(hidden_layer, output_layer)
b_input = random.sample(range(-100,100), hidden_layer)
b_output = random.sample(range(-100, 100), output_layer)
hidden = []
outputt = []

#формула nn.Linear
#y = w*x + b

#input layer with relu(), relu() убирает отрицательные значения
for i in range(hidden_layer):
    y = 0
    for j in range(input_layer):
        y += w_input[i,j]*x[j]
    y += b_input[i]
    if y<0:
        hidden.append(0)
    else:
        hidden.append(y)
#output layer
for i in range(output_layer):
    res = 0
    for j in range(hidden_layer):
        res += w_output[j,i]*hidden[j]
    res += b_output[i]
    outputt.append(res)
print(outputt)

#tanh делает диапазон от -1 до 1
e = math.exp(1)
def tanh(x):
    return (e**x-e**(-x))/(e**x+e**(-x))
