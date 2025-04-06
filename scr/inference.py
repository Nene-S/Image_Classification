import torch
import load_data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import CNNModel
from resnetModel import ResNet, ResidualBlock
import config 

def inference(model, data, batch_size=30):
    """
    Performs inference on a given dataset using a trained model.

    Args:
        model (torch.nn.Module): The trained neural network model.
        data (torch.utils.data.Dataset): The dataset containing images and labels.
        batch_size (int, optional): The batch size for inference. Default is 20.

    Returns:
        tuple:
            - pred_labels (torch.Tensor): Tensor of predicted labels.
            - true_labels (torch.Tensor): Tensor of actual labels.
            - image_array (torch.Tensor): Tensor containing image data.
            - accuracy_test (float): Computed accuracy on the dataset.
    """
    test_dl = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    pred_labels = []
    true_labels = []
    image_array = []
    accuracy_test = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            pred = model(x_batch)
            pred_labels.append(torch.argmax(pred, dim=1))
            true_labels.append(y_batch)
            image_array.append(x_batch)
            
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_test += is_correct.sum()

    accuracy_test /= len(test_dl.dataset)
    print(f"Test accuracy: {accuracy_test:.4f}")

    pred_labels = torch.cat(pred_labels)
    true_labels = torch.cat(true_labels)
    image_array = torch.cat(image_array)

    return pred_labels, true_labels, image_array

def plot_predictions(pred_labels, true_labels, image_array, savefig_path):
        fig = plt.figure(figsize=(12, 5))
        label_map = {0: "Happy", 1: "Neutral", 2: "Sad"}
        for j in range(12):
            ax = fig.add_subplot(3, 4, j+1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(image_array[-j].cpu().permute(1, 2, 0))

            gt_label = label_map.get(true_labels[-j].item(), "Unknown")
            pred_label = label_map.get(pred_labels[-j].item(), "Unknown")
            ax.text(
                0.5, -0.15, 
                f'GT: {gt_label}\nPred: {pred_label}', 
                size=8,
                horizontalalignment='center',
                verticalalignment='center', 
                transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(savefig_path)
        plt.show()

def main():
    """
    Loads the trained models, performs inference, and visualizes the predictions.

    """
    _,_, test_data = load_data.load_dataset()
    # Load the model
    cnn_model = CNNModel()
    resnet_model = ResNet(ResidualBlock, 3, [2,2,2,2], [64,128,256,512], 3)
    
    cnn_model.load_state_dict(torch.load(config.cnn_model_path))
    resnet_model.load_state_dict(torch.load(config.resnet_model_path))
    
    # Make inference on both cnn and resnet model
    pred_labels_cnn, true_labels_cnn, image_array_cnn = inference(cnn_model, test_data)
    pred_labels_res, true_labels_res, image_array_res = inference(resnet_model, test_data)

    # Plot output of resnet predictions
    fig_path = config.inference_output_pth
    plot_predictions(pred_labels_res, true_labels_res, image_array_res, fig_path)

if __name__ == "__main__":
    main()
