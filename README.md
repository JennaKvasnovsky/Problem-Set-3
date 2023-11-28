# Problem Set 3
# Detailed Report on Fine-Tuning AlexNet for Flower Classification

## 1. Introduction

The provided code aims to fine-tune the AlexNet model for the specific task of classifying flower images from the Oxford-102 Flower dataset. This report provides a detailed overview of the code structure, including key steps such as dataset preparation, model setup, training, validation, and results visualization.

## 2. Installation of Libraries

The initial section focuses on ensuring the availability of essential libraries. The code installs torch, torchvision, and matplotlib using the pip package manager. These libraries are fundamental for deep learning operations and data visualization.

```bash
!pip install torch torchvision matplotlib
```

## 3. Dataset Download and Extraction

To facilitate experimentation with the Flower dataset, the code automates the download and extraction process. It retrieves the dataset and labels from online sources using wget and unzip commands. The dataset consists of images belonging to 102 different flower classes.

```bash
!wget https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
!wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip 'flower_data.zip'
```

## 4. Data Preprocessing and Loading

The subsequent part of the code is dedicated to data preprocessing and loading. It defines separate data transformations for training and validation datasets, including random resizing, cropping, and normalization. These transformations are encapsulated using the `transforms.Compose` class from torchvision. The torch DataLoader is then employed to efficiently load the datasets into batches for model training.

```python
# Data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load training dataset
train_dataset = datasets.ImageFolder(os.path.join('/content/flower_data/', 'train'), transform=train_transform)

# Load validation dataset
val_dataset = datasets.ImageFolder(os.path.join('/content/flower_data/', 'valid'), transform=val_transform)

# DataLoader setup
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

## 5. Model Setup

The AlexNet model, pretrained on ImageNet, is loaded and customized for the flower classification task. The final fully connected layer is replaced with a new one, adapting to the number of classes in the flower dataset.

```python
alexnet = models.alexnet(pretrained=True).to(device)
num_classes = len(train_dataset.classes)
alexnet.classifier[6] = torch.nn.Linear(4096, num_classes).to(device)
```

## 6. Training Loop

The core of the code is the training loop, which iterates over a specified number of epochs. For each epoch, the model is trained on the training dataset, and its performance is evaluated on the validation dataset. The code utilizes CrossEntropyLoss as the loss function and stochastic gradient descent (SGD) as the optimizer. Training progress, including loss and accuracy, is displayed for each epoch.

```python
for epoch in trange(num_epochs):
    # Training phase
    alexnet.train()
    # ... (training loop)

    # Validation phase
    alexnet.eval()
    # ... (validation loop)

    # Display a sample of images with true and predicted labels
sample_images, sample_labels = next(iter(val_loader))
sample_images, sample_labels = sample_images[:5].to(device), sample_labels[:5].to(device)

with torch.no_grad():
    outputs = alexnet(sample_images)
    _, predicted_classes = torch.max(outputs, 1)

# Convert to numpy arrays for visualization
sample_images = sample_images.cpu().numpy()
sample_labels = sample_labels.cpu().numpy()
predicted_classes = predicted_classes.cpu().numpy()

# Plot the images with true and predicted labels
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axes[i].imshow(np.transpose(sample_images[i], (1, 2, 0)))
    axes[i].set_title(f'True: {sample_labels[i]}, Predicted: {predicted_classes[i]}')
    axes[i].axis('off')

plt.show()
```

## 7. Results Visualization

At the end of each epoch, the code displays a sample of images from the validation set along with their true and predicted labels. This visual representation provides insights into the model's performance and aids in understanding its predictions.

## 8. Conclusion

In conclusion, the provided code offers a comprehensive implementation for fine-tuning the AlexNet model on the Flower dataset. It covers all essential aspects of a deep learning pipeline, including data preprocessing, model setup, training, validation, and results visualization. The code is well-structured and serves as a solid foundation for users interested in flower image classification tasks. Additionally, it provides flexibility for customization, allowing users to tweak parameters based on their specific requirements.
