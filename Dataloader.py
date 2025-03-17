import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

diseases = ["Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy", "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy", 
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy", "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", 
    "Grape Healthy", "Orange Haunglongbing", "Peach Bacterial Spot", "Peach Healthy", "Pepper Bacterial Spot", "Pepper Healthy", "Potato Early Blight", 
    "Potato Late Blight", "Potato Healthy", 'Raspberry Healthy', "Soybean Healthy", "Squash Powdery Mildew", "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites", "Tomato Target Spot", 
    "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato healthy"]
    
def extract_name_and_disease(category):
    name, disease = category.split('___')
    
    if '_' in name:
        name = name.split('_')[0]
    elif ',' in name:
        name = name.split(',')[0]
    
    name = name.split('/')[-1]
    return name, disease

def load_data(PATH):
    filenames = []
    plant = []
    disease = []
    data_folders = sorted(os.listdir(PATH))
    
    for file in tqdm(sorted(data_folders)):  # Iterate through sorted folders
        for img in os.listdir(os.path.join(PATH, file)):
            name, _ = extract_name_and_disease(file)
            plant.append(name)
            disease.append(diseases[data_folders.index(file)])  # Ensure correct indexing
            filenames.append(os.path.join(PATH, file, img))
    
    df = pd.DataFrame({
        'filename': filenames,
        'Name': plant,
        'Disease': disease
    })
    # plants_name = df['Disease'].unique()
    # print(plants_name)
    
    return df.sample(frac=1).reset_index(drop=True)

class PlantDiseaseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        name = self.dataframe.iloc[idx, 3]
        disease = self.dataframe.iloc[idx, 4]
        
        if self.transform:
            image = self.transform(image)
            
        return image, name, disease

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(3, p=0.5)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128), antialias=True),
        ])
    }

def create_data_loaders(batch_size=64, test_size=0.4):
    data_dir = r"C:\Users\USER\plantvillage dataset\color"
    df = load_data(data_dir)
    
    from sklearn.preprocessing import LabelEncoder
    le_name = LabelEncoder()
    le_disease = LabelEncoder()
    df['Name_label'] = le_name.fit_transform(df['Name'])
    df['Disease_label'] = le_disease.fit_transform(df['Disease'])
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=test_size, random_state=42)
    
    transformations = get_transforms()
    
    train_dataset = PlantDiseaseDataset(train_df, transform=transformations['train'])
    val_dataset = PlantDiseaseDataset(val_df, transform=transformations['val'])
    test_dataset = PlantDiseaseDataset(test_df, transform=transformations['val'])
    
    loaders = {
        'train': DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size*2, num_workers=4, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size*2)
    }
    
    return loaders, df
# loaders, df = create_data_loaders()
# print(df.head())