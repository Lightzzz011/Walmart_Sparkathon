from ultralytics import YOLO
from roboflow import Roboflow
import torch
from wandb.integration.ultralytics import add_wandb_callback
import wandb

def main():

    #Initialize a Weights & Biases run
    wandb.init(project="shoppingcartsmall_yolov8n_nonpretrained", job_type="training")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = YOLO('yolov8n.pt').to(device)

    # Add W&B Callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)
    
    # supplementary code from official yolov8 docs: https://docs.ultralytics.com/usage/python/#train
    
    results = model.train(data="data.yaml", epochs=110, batch = 25, pretrained=False)
    validation = model.val()
    print(results)
    print(validation)

    # Step 7: Finalize the W&B Run
    wandb.finish()

if __name__ == '__main__':
    main()