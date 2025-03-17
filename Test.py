import torch
import torchmetrics
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from Model import Model
import Train
from Dataloader import create_data_loaders, diseases

def evaluate(model, data_loader, device, plants_name, readable_diseases):
    model.eval()  # Set the model to evaluation mode
    
    preds1, preds2, true_leaf, true_disease = [], [], [], []
    
    with torch.no_grad():
        for x, y1, y2 in tqdm(data_loader, desc='Evaluating', leave=False):
            x, y1, y2 = [v.to(device) for v in (x, y1, y2)]
            pred = model(x)
            
            pred1 = torch.argmax(pred[0], axis=1).cpu().numpy()
            pred2 = torch.argmax(pred[1], axis=1).cpu().numpy()
            preds1.extend(pred1)
            preds2.extend(pred2)
            true_leaf.extend(y1.cpu().numpy())
            true_disease.extend(y2.cpu().numpy())
    
    # Calculate accuracy for plant name and disease classification
    leaf_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=14)(
        torch.tensor(preds1), torch.tensor(true_leaf)
    )
    disease_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=38)(
        torch.tensor(preds2), torch.tensor(true_disease)
    )
    
    print('Testing Accuracy')
    print(f"Plant Name Classification Accuracy: {leaf_accuracy.item() * 100:.2f}%")
    print(f"Disease Classification Accuracy: {disease_accuracy.item() * 100:.2f}%")

    # Generate classification reports
    report1 = classification_report(true_leaf, preds1, target_names=plants_name, output_dict=True)
    report2 = classification_report(true_disease, preds2, target_names=readable_diseases, output_dict=True)
    
    # Convert reports to DataFrames and save as CSV
    pd.DataFrame(report1).transpose().to_csv('best_classification_report_plants.csv')
    pd.DataFrame(report2).transpose().to_csv('best_classification_report_diseases.csv')
    # print(f"Unique classes in true_disease: {len(set(true_disease))}")
    # print(f"Unique classes in preds2: {len(set(preds2))}")

    return preds1, preds2, true_leaf, true_disease

if __name__ == "__main__":
    plants_name = ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 'Pepper,', 'Potato', 'Raspberry', 'Soybean', 'Squash','Strawberry', 'Tomato']
    
    loaders, _ = create_data_loaders()
    device = Train.get_default_device()
    test_dl = Train.DeviceDataLoader(loaders['test'], device)
    
    best_model = Model().to(device)  # Ensure model is on the correct device
    best_model.load_state_dict(torch.load('best_model.pth', map_location=device))  # Load model weights
    evaluate(best_model, test_dl, device, plants_name, diseases)

