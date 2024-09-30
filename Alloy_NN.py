#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd

# Get the current working directory
cwd = os.getcwd()

# Define the data path relative to the current directory
data_path = os.path.join(cwd, 'corrosion_data.xlsx')

# Load the Excel file
data1 = pd.read_excel(data_path)

data1.head()


# In[4]:


import pandas as pd

data = data1.iloc[1:227, [4, 5, 66, 67, 68, 69, 71,77]]


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


data.head(20)


# In[9]:


# `data` is your DataFrame
indices_to_drop = [9,10,11,17]
data = data.drop(indices_to_drop)


# In[10]:


data.head(20)


# In[11]:


# `data` is your DataFrame
indices_to_drop = [19]
data = data.drop(indices_to_drop)


# In[12]:


data.head(20)


# In[13]:


data.shape


# In[14]:


input_data = data.iloc[:,0:7]


# In[15]:


input_data.head()


# In[16]:


input_data.shape


# In[17]:


Output_data = data.iloc[:,-1]


# In[18]:


Output_data.head()


# In[19]:


Output_data =Output_data.values.reshape(-1, 1)
Output_data .shape


# # regular NN model

# In[20]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Function to split data for training, validation, and testing
def split_data(inputs, outputs, val_size=0.2, test_size=0.2):
    inputs_train, inputs_temp, outputs_train, outputs_temp = train_test_split(
        inputs, outputs, test_size=val_size+test_size, random_state=42)
    val_split = test_size / (val_size + test_size)
    inputs_val, inputs_test, outputs_val, outputs_test = train_test_split(
        inputs_temp, outputs_temp, test_size=val_split, random_state=42)
    return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test

materials_data = [
    split_data(input_data[0:36], Output_data[0:36]),
    split_data(input_data[36:174], Output_data[36:174]),
    split_data(input_data[174:221], Output_data[174:221])
]


materials_data = [
    (np.asarray(data[0]), np.asarray(data[1]), np.asarray(data[2]), 
     np.asarray(data[3]), np.asarray(data[4]), np.asarray(data[5])) 
    for data in materials_data
]

# Custom Dataset class
class MaterialDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# Neural network with separate parameters for each material
class MaterialSpecificModel(nn.Module):
    def __init__(self, num_materials=3, num_features=7, num_outputs=1):
        super(MaterialSpecificModel, self).__init__()
        self.common_layers = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(64, num_outputs)

    def forward(self, x, material_idx):
        x = self.common_layers(x)
        x = self.output_layer(x)
        return x, material_idx

# Custom loss function
def custom_loss_function(outputs, targets, model, inputs, material_idx):
    # Calculate MSE loss
    mse_loss = torch.mean((outputs - targets) ** 2)

    # Combine the losses
    combined_loss = mse_loss 
    return combined_loss

# Initialize the network and optimizer
model = MaterialSpecificModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoaders for training, validation, and testing
train_loaders = [DataLoader(MaterialDataset(data[0], data[3]), batch_size=8, shuffle=True) for data in materials_data]
val_loaders = [DataLoader(MaterialDataset(data[1], data[4]), batch_size=8) for data in materials_data]
test_loaders = [DataLoader(MaterialDataset(data[2], data[5]), batch_size=8) for data in materials_data]

# Training and validation loops
num_epochs = 250
training_losses = []
validation_losses = []

for epoch in range(num_epochs):
    total_train_loss = 0.0
    total_val_loss = 0.0

    model.train()
    for material_idx, train_loader in enumerate(train_loaders):
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear gradients for each batch
            outputs, _ = model(inputs, material_idx)
            loss = custom_loss_function(outputs, targets, model, inputs, material_idx)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
    model.eval()
    with torch.no_grad():
        for material_idx, val_loader in enumerate(val_loaders):
            for inputs, targets in val_loader:
                outputs, _ = model(inputs, material_idx)
                loss = custom_loss_function(outputs, targets, model, inputs, material_idx)
                total_val_loss += loss.item()

    avg_train_loss = total_train_loss / sum(len(loader) for loader in train_loaders)
    avg_val_loss = total_val_loss / sum(len(loader) for loader in val_loaders)
    training_losses.append(avg_train_loss)
    validation_losses.append(avg_val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# Plotting the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[21]:


# Testing phase with prediction collection
def test_model_with_predictions(model, test_loaders):
    model.eval()
    test_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for material_idx, test_loader in enumerate(test_loaders):
            total_test_loss = 0.0
            material_predictions = []
            material_targets = []
            
            for inputs, targets in test_loader:
                outputs, _ = model(inputs, material_idx)
                loss = custom_loss_function(outputs, targets, model, inputs, material_idx)
                total_test_loss += loss.item()
                
                material_predictions.extend(outputs.numpy().flatten())
                material_targets.extend(targets.numpy().flatten())
                
            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            all_predictions.append(material_predictions)
            all_targets.append(material_targets)
            print(f'Material {material_idx + 1} - Test Loss: {avg_test_loss:.4f}')
    
    return test_losses, all_predictions, all_targets

# Execute the testing phase
test_losses, all_predictions, all_targets = test_model_with_predictions(model, test_loaders)


# In[22]:


# Plotting the comparison between predicted and actual values
def plot_comparisons(all_predictions, all_targets):
    materials = len(all_predictions)
    plt.figure(figsize=(15, 5 * materials))

    for i in range(materials):
        plt.subplot(materials, 1, i + 1)
        plt.plot(all_targets[i], label='Actual')
        plt.plot(all_predictions[i], label='Predicted', alpha=0.7)
        plt.title(f'Material {i + 1} - Predicted vs Actual')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the comparisons
plot_comparisons(all_predictions, all_targets)


# In[23]:


import numpy as np
import torch

# Testing phase with prediction collection and inversion from log scale
def test_model_with_predictions_and_invert_scale(model, test_loaders):
    model.eval()
    test_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for material_idx, test_loader in enumerate(test_loaders):
            total_test_loss = 0.0
            material_predictions = []
            material_targets = []
            
            for inputs, targets in test_loader:
                outputs, _ = model(inputs, material_idx)
                # Calculate loss before inverting the scale
                loss = custom_loss_function(outputs, targets, model, inputs, material_idx)
                total_test_loss += loss.item()
                
                # Invert the scale of outputs from log scale back to original
                inverted_outputs = np.exp(-outputs.numpy().flatten())
                material_predictions.extend(inverted_outputs)

                # Invert the scale of targets from log scale back to original
                inverted_targets = np.exp(-targets.numpy().flatten())
                material_targets.extend(inverted_targets)
                
            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            all_predictions.append(material_predictions)
            all_targets.append(material_targets)
            print(f'Material {material_idx + 1} - Test Loss: {avg_test_loss:.4f}')
    
    return test_losses, all_predictions, all_targets

# Execute the testing phase
test_losses, all_predictions, all_targets = test_model_with_predictions_and_invert_scale(model, test_loaders)


# In[24]:


import numpy as np
import torch

def test_model_with_predictions_and_invert_scale(model, test_loaders):
    model.eval()
    test_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for material_idx, test_loader in enumerate(test_loaders):
            total_test_loss = 0.0
            material_predictions = []
            material_targets = []
            
            for inputs, targets in test_loader:
                outputs, _ = model(inputs, material_idx)
                # Calculate loss before inverting the scale
                loss = custom_loss_function(outputs, targets, model, inputs, material_idx)
                total_test_loss += loss.item()
                
                # Invert the scale of outputs from log scale back to original
                inverted_outputs = np.exp(-outputs.detach().cpu().numpy().flatten())
                material_predictions.extend(inverted_outputs)

                # Invert the scale of targets from log scale back to original
                inverted_targets = np.exp(-targets.detach().cpu().numpy().flatten())
                material_targets.extend(inverted_targets)
                
            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            all_predictions.append(material_predictions)
            all_targets.append(material_targets)
            print(f'Material {material_idx + 1} - Test Loss: {avg_test_loss:.4f}')
    
    return test_losses, all_predictions, all_targets

# Execute the testing phase
test_losses, all_predictions, all_targets = test_model_with_predictions_and_invert_scale(model, test_loaders)

# Printing all original predicted and target values
for i in range(len(all_predictions)):
    print(f"Material {i+1} Predictions: {all_predictions[i]}")
    print(f"Material {i+1} Targets: {all_targets[i]}")
    
import numpy as np

def calculate_mse(predictions, targets):
    """Calculate the mean squared error between predictions and targets."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.mean((predictions - targets) ** 2)

# Assuming all_predictions and all_targets are lists of lists where each sublist corresponds to a material type
mse_scores = []
for i in range(len(all_predictions)):
    mse = calculate_mse(all_predictions[i], all_targets[i])
    mse_scores.append(mse)
    print(f"Material {i+1} MSE: {mse:.9f}")

# If you want to also display the overall MSE for all materials combined
all_material_predictions = [pred for sublist in all_predictions for pred in sublist]
all_material_targets = [target for sublist in all_targets for target in sublist]
overall_mse = calculate_mse(all_material_predictions, all_material_targets)
print(f"Overall MSE: {overall_mse:.8f}")    
    


# In[73]:


import matplotlib.pyplot as plt

# Defining the data
all_predictions = [
    [0.00044404887, 0.0004808659, 0.00044056427, 0.00025237259, 0.0005806881, 0.0006510963, 9.546525e-05, 6.418152e-05],
   [0.00011568826, 2.7999577e-05, 1.52267785e-05, 1.4833863e-05, 0.0031858054, 0.0012513507, 3.1111886e-06, 5.4731468e-06, 4.2258147e-05, 3.7535253e-06, 3.647185e-06, 4.3606506e-05, 4.131525e-06, 0.00129566, 3.8988623e-05, 3.218703e-06, 4.109174e-06, 6.0059924e-06, 3.2753449e-06, 4.0232603e-06, 2.0862737e-05, 6.199256e-06, 7.782112e-05, 0.0013063886, 3.7535294e-06, 4.392534e-05, 6.759076e-05, 5.5115168e-05],
    [6.2251966e-05, 0.00031365952, 1.3705859e-05, 0.00054542435, 1.309075e-05, 4.9430128e-05, 0.00031369212, 0.00056111097, 1.8101176e-05, 0.0003282061]
]

all_targets = [
    [0.002127207, 0.0016795004, 0.001985226, 0.00021393207, 0.0020267735, 0.005239717, 0.001178148, 0.006337538],
    [6.2840336e-05, 5.746271e-06, 2.816898e-07, 6.1720784e-06, 0.0016111361, 0.0006091098, 8.110998e-05, 0.00032215624, 0.00046964173, 1.4088532e-06, 1.5953807e-06, 7.87321e-06, 2.5379609e-06, 0.00068771536, 3.999334e-05, 2.290561e-05, 5.8323712e-06, 5.215132e-06, 8.569762e-06, 2.5985943e-05, 1.8312074e-06, 0.0013795657, 9.208508e-06, 0.0010634112, 1.7186941e-06, 3.3320342e-05, 2.3887513e-05, 0.00032699862],
    [8.7670116e-05, 2.588019e-05, 0.00014446917, 2.2350772e-05, 1.7702901e-05, 1.8451114e-05, 2.8355982e-05, 3.003074e-05, 1.4555851e-06, 3.279159e-05]
]

# Plotting function
def plot_comparisons(all_predictions, all_targets):
    titles = ["Mg Based Alloys", "Fe-Ni Based Alloys", "Al Based Alloys"]
    materials = len(all_predictions)
    plt.figure(figsize=(15, 5 * materials))

    for i in range(materials):
        plt.subplot(materials, 1, i + 1)
        plt.plot(all_targets[i], 'o-', label='Actual', alpha=0.8)  # Circle markers for actual
        plt.plot(all_predictions[i], 'x-', label='Predicted', alpha=0.7)  # Cross markers for predicted
        plt.title(f'{titles[i]} - Predicted vs Actual')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the comparisons using the provided data
plot_comparisons(all_predictions, all_targets)


# In[1]:


pip freeze > requirements.txt


# In[2]:


import sys
print(sys.executable)


# In[3]:


import os
# Get the current working directory
cwd = os.getcwd()
print(f"Current Working Directory: {cwd}")


# In[74]:


import numpy as np

# Define a function to calculate percentage errors
def calculate_percentage_errors(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    targets = np.where(targets == 0, 1e-10, targets)
    percentage_errors = np.abs((predictions - targets) / targets) * 100
    return percentage_errors

# Material 1
material_1_predictions =[0.00044404887, 0.0004808659, 0.00044056427, 0.00025237259, 0.0005806881, 0.0006510963, 9.546525e-05, 6.418152e-05],
material_1_targets = [0.002127207, 0.0016795004, 0.001985226, 0.00021393207, 0.0020267735, 0.005239717, 0.001178148, 0.006337538]
material_1_errors = calculate_percentage_errors(material_1_predictions, material_1_targets)

# Material 2
material_2_predictions = [0.00011568826, 2.7999577e-05, 1.52267785e-05, 1.4833863e-05, 0.0031858054, 0.0012513507, 3.1111886e-06, 5.4731468e-06, 4.2258147e-05, 3.7535253e-06, 3.647185e-06, 4.3606506e-05, 4.131525e-06, 0.00129566, 3.8988623e-05, 3.218703e-06, 4.109174e-06, 6.0059924e-06, 3.2753449e-06, 4.0232603e-06, 2.0862737e-05, 6.199256e-06, 7.782112e-05, 0.0013063886, 3.7535294e-06, 4.392534e-05, 6.759076e-05, 5.5115168e-05],
material_2_targets =  [6.2840336e-05, 5.746271e-06, 2.816898e-07, 6.1720784e-06, 0.0016111361, 0.0006091098, 8.110998e-05, 0.00032215624, 0.00046964173, 1.4088532e-06, 1.5953807e-06, 7.87321e-06, 2.5379609e-06, 0.00068771536, 3.999334e-05, 2.290561e-05, 5.8323712e-06, 5.215132e-06, 8.569762e-06, 2.5985943e-05, 1.8312074e-06, 0.0013795657, 9.208508e-06, 0.0010634112, 1.7186941e-06, 3.3320342e-05, 2.3887513e-05, 0.00032699862]
material_2_errors = calculate_percentage_errors(material_2_predictions, material_2_targets)

# Material 3
material_3_predictions = [6.2251966e-05, 0.00031365952, 1.3705859e-05, 0.00054542435, 1.309075e-05, 4.9430128e-05, 0.00031369212, 0.00056111097, 1.8101176e-05, 0.0003282061]
material_3_targets = [8.7670116e-05, 2.588019e-05, 0.00014446917, 2.2350772e-05, 1.7702901e-05, 1.8451114e-05, 2.8355982e-05, 3.003074e-05, 1.4555851e-06, 3.279159e-05]
material_3_errors = calculate_percentage_errors(material_3_predictions, material_3_targets)

# Print percentage errors
print("Material 1 Percentage Errors:", material_1_errors)
print("Material 2 Percentage Errors:", material_2_errors)
print("Material 3 Percentage Errors:", material_3_errors)

# Calculate average percentage errors for each material
average_percentage_error_1 = np.mean(material_1_errors)
average_percentage_error_2 = np.mean(material_2_errors)
average_percentage_error_3 = np.mean(material_3_errors)

# Print average percentage errors
print("Average Percentage Error for Material 1:", average_percentage_error_1)
print("Average Percentage Error for Material 2:", average_percentage_error_2)
print("Average Percentage Error for Material 3:", average_percentage_error_3)


# In[5]:


import numpy as np

# Define a function to calculate percentage errors
def calculate_percentage_errors(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate the range of the targets
    target_range = np.max(targets) - np.min(targets)
    if target_range == 0:
        target_range = 1e-10  # Avoid division by zero if all targets are the same
    
    # Calculate the percentage errors based on the target range
    percentage_errors = np.abs((predictions - targets) / target_range) * 100
    return percentage_errors

# Material 1
material_1_predictions = [0.00044404887, 0.0004808659, 0.00044056427, 0.00025237259, 0.0005806881, 0.0006510963, 9.546525e-05, 6.418152e-05]
material_1_targets = [0.002127207, 0.0016795004, 0.001985226, 0.00021393207, 0.0020267735, 0.005239717, 0.001178148, 0.006337538]
material_1_errors = calculate_percentage_errors(material_1_predictions, material_1_targets)

# Material 2
material_2_predictions = [0.00011568826, 2.7999577e-05, 1.52267785e-05, 1.4833863e-05, 0.0031858054, 0.0012513507, 3.1111886e-06, 5.4731468e-06, 4.2258147e-05, 3.7535253e-06, 3.647185e-06, 4.3606506e-05, 4.131525e-06, 0.00129566, 3.8988623e-05, 3.218703e-06, 4.109174e-06, 6.0059924e-06, 3.2753449e-06, 4.0232603e-06, 2.0862737e-05, 6.199256e-06, 7.782112e-05, 0.0013063886, 3.7535294e-06, 4.392534e-05, 6.759076e-05, 5.5115168e-05]
material_2_targets = [6.2840336e-05, 5.746271e-06, 2.816898e-07, 6.1720784e-06, 0.0016111361, 0.0006091098, 8.110998e-05, 0.00032215624, 0.00046964173, 1.4088532e-06, 1.5953807e-06, 7.87321e-06, 2.5379609e-06, 0.00068771536, 3.999334e-05, 2.290561e-05, 5.8323712e-06, 5.215132e-06, 8.569762e-06, 2.5985943e-05, 1.8312074e-06, 0.0013795657, 9.208508e-06, 0.0010634112, 1.7186941e-06, 3.3320342e-05, 2.3887513e-05, 0.00032699862]
material_2_errors = calculate_percentage_errors(material_2_predictions, material_2_targets)

# Material 3
material_3_predictions = [6.2251966e-05, 0.00031365952, 1.3705859e-05, 0.00054542435, 1.309075e-05, 4.9430128e-05, 0.00031369212, 0.00056111097, 1.8101176e-05, 0.0003282061]
material_3_targets = [8.7670116e-05, 2.588019e-05, 0.00014446917, 2.2350772e-05, 1.7702901e-05, 1.8451114e-05, 2.8355982e-05, 3.003074e-05, 1.4555851e-06, 3.279159e-05]
material_3_errors = calculate_percentage_errors(material_3_predictions, material_3_targets)

# Print percentage errors
print("Material 1 Percentage Errors:", material_1_errors)
print("Material 2 Percentage Errors:", material_2_errors)
print("Material 3 Percentage Errors:", material_3_errors)

# Calculate average percentage errors for each material
average_percentage_error_1 = np.mean(material_1_errors)
average_percentage_error_2 = np.mean(material_2_errors)
average_percentage_error_3 = np.mean(material_3_errors)

# Print average percentage errors
print("Average Percentage Error for Material 1:", average_percentage_error_1)
print("Average Percentage Error for Material 2:", average_percentage_error_2)
print("Average Percentage Error for Material 3:", average_percentage_error_3)


# In[76]:


for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {training_losses[epoch]:.4f}, Validation Loss: {validation_losses[epoch]:.4f}')


# In[77]:


# Save the model's state dictionary
torch.save(model.state_dict(), 'material_specific_model_NN.pth')

# Optionally, you can save the optimizer state as well
torch.save(optimizer.state_dict(), 'optimizer_NN.pth')


# # Comparision plot among Ground truth, NN and the PCNN

# In[69]:


import matplotlib.pyplot as plt

# Defining the data
all_predictions_model1 = [
    [0.00057569897, 0.0015303588, 0.001539192, 0.0008056699, 0.0015494513, 0.0014318005, 0.0003708349, 0.00034132137],
    [7.0937465e-05, 8.293948e-06, 8.498094e-06, 8.234829e-06, 0.0006044002, 0.00031636172, 7.4889417e-06, 4.567256e-05, 4.1469066e-05, 2.163628e-06, 5.0449175e-06, 8.7604385e-06, 2.0028635e-06, 0.0003327197, 1.2450642e-05, 4.7087165e-06, 2.8133472e-06, 3.2469159e-06, 4.524176e-06, 1.5690588e-05, 8.2412e-06, 6.63382e-05, 1.4601825e-05, 0.00033940983, 2.163628e-06, 1.0464134e-05, 1.504931e-05, 4.507688e-05],
    [0.00015700866, 3.469539e-05, 4.1066338e-05, 5.4807755e-05, 4.0289215e-05, 4.2293665e-05, 7.44824e-05, 5.0209503e-05, 3.532572e-05, 7.062183e-05]
]

all_predictions_model2 = [
    [0.00057576015, 0.0010618076, 0.0010411938, 0.00046464682, 0.0012195064, 0.0014439968, 0.0001260252, 0.00013914402],
    [0.0001095022, 4.8870053e-05, 1.3812063e-05, 3.3375174e-05, 0.0034172966, 0.0009122986, 3.8201542e-06, 2.9788882e-05, 0.00013100311, 3.4337695e-06, 8.515682e-06, 2.1816657e-05, 5.590625e-06, 0.0009673433, 4.042762e-05, 1.0521402e-05, 9.987365e-06, 1.0825766e-05, 1.02362965e-05, 1.7387256e-05, 4.5416036e-05, 3.5754914e-05, 0.00021347657, 0.0009598809, 3.4337756e-06, 2.3972501e-05, 0.00015436704, 0.00013094152],
    [0.000103075, 0.00043426602, 4.209472e-05, 0.00061158504, 3.9310828e-05, 8.5994594e-05, 0.00061501865, 0.00060593337, 2.9414216e-05, 0.0006153305]
]

all_targets = [
    [0.002127207, 0.0016795004, 0.001985226, 0.00021393207, 0.0020267735, 0.005239717, 0.001178148, 0.006337538],
    [6.2840336e-05, 5.746271e-06, 2.816898e-07, 6.1720784e-06, 0.0016111361, 0.0006091098, 8.110998e-05, 0.00032215624, 0.00046964173, 1.4088532e-06, 1.5953807e-06, 7.87321e-06, 2.5379609e-06, 0.00068771536, 3.999334e-05, 2.290561e-05, 5.8323712e-06, 5.215132e-06, 8.569762e-06, 2.5985943e-05, 1.8312074e-06, 0.0013795657, 9.208508e-06, 0.0010634112, 1.7186941e-06, 3.3320342e-05, 2.3887513e-05, 0.00032699862],
    [8.7670116e-05, 2.588019e-05, 0.00014446917, 2.2350772e-05, 1.7702901e-05, 1.8451114e-05, 2.8355982e-05, 3.003074e-05, 1.4555851e-06, 3.279159e-05]
]

# Plotting function
def plot_comparisons(all_predictions_model1, all_predictions_model2, all_targets):
    titles = ["Mg Based Alloys", "Fe-Ni Based Alloys", "Al Based Alloys"]
    materials = len(all_predictions_model1)
    plt.figure(figsize=(15, 5 * materials))

    for i in range(materials):
        plt.subplot(materials, 1, i + 1)
        plt.plot(all_targets[i], 'o-', label='Ground TRuth', alpha=0.8)  # Circle markers for actual values
        plt.plot(all_predictions_model1[i], 'x-', label='Physics Constrained NN Model', alpha=0.7)  # Cross markers for predictions from model 1
        plt.plot(all_predictions_model2[i], 's-', label='Traditional NN Model', alpha=0.6)  # Square markers for predictions from model 2
        plt.title(f'{titles[i]} - Predicted vs Actual')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the comparisons using the provided data
plot_comparisons(all_predictions_model1, all_predictions_model2, all_targets)


# In[73]:


import matplotlib.pyplot as plt

# Plotting function
def plot_comparisons_compact(all_predictions_model1, all_predictions_model2, all_targets):
    titles = ["Mg Based Alloys", "Fe-Ni Based Alloys", "Al Based Alloys"]
    materials = len(all_predictions_model1)
    plt.figure(figsize=(8, 3 * materials))  # More compact figure size

    for i in range(materials):
        plt.subplot(materials, 1, i + 1)
        plt.plot(all_targets[i], 'o-', label='Ground Truth', markersize=4, alpha=0.8)  # Smaller markers
        plt.plot(all_predictions_model1[i], 'x-', label='Physics Constrained NN', markersize=4, alpha=0.7)
        plt.plot(all_predictions_model2[i], 's-', label='Traditional NN', markersize=4, alpha=0.6)
        plt.xlabel('Sample Index', fontsize=10)
        plt.ylabel('1/Rp', fontsize=10)
        plt.grid(False)  # No gridlines
        if i == 0:
            plt.legend(loc='upper left', fontsize=9)  # Legend in the upper left corner

    plt.tight_layout(pad=1.0)  # Tighter layout with minimal padding
    plt.show()

# Plot the comparisons using the provided data
plot_comparisons_compact(all_predictions_model1, all_predictions_model2, all_targets)



# In[76]:


import matplotlib.pyplot as plt

# Plotting function
def plot_comparisons_compact(all_predictions_model1, all_predictions_model2, all_targets):
    titles = ["Mg Based Alloys", "Fe-Ni Based Alloys", "Al Based Alloys"]
    materials = len(all_predictions_model1)
    plt.figure(figsize=(8, 3 * materials))  # More compact figure size

    for i in range(materials):
        plt.subplot(materials, 1, i + 1)
        plt.plot(all_targets[i], 'o-', color='blue', label='Ground Truth', markersize=5, alpha=0.8)  # Consistent and distinct colors
        plt.plot(all_predictions_model1[i], 'x-', color='red', label='Physics Constrained NN', markersize=5, alpha=0.7)
        plt.plot(all_predictions_model2[i], 's-', color='green', label='Traditional NN', markersize=5, alpha=0.6)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('1/Rp', fontsize=12)
        plt.title(titles[i], fontsize=14)
        plt.grid(False)  # No gridlines

        if i == 0:  # Display the legend only for the first subplot
            plt.legend(loc='upper left', fontsize=10)  # Legend inside the plot, top left

    plt.tight_layout(pad=1.0, h_pad=2.0)  # Tighter layout with minimal padding
    plt.show()

# Plot the comparisons using the provided data
plot_comparisons_compact(all_predictions_model1, all_predictions_model2, all_targets)


