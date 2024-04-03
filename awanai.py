
import os
import numpy as np
import gradio as gr
from glob import glob
from functools import partial
from dataclasses import dataclass

import torch
import torchvision
import torch.nn as nn
import lightning.pytorch as pl
import torchvision.transforms as TF

from torchmetrics import MeanMetric
from torchmetrics.classification import MultilabelF1Score


@dataclass
class DatasetConfig:
    IMAGE_SIZE: tuple = (384, 384)  # (W, H)
    CHANNELS: int = 3
    NUM_CLASSES: int = 10
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD: tuple = (0.229, 0.224, 0.225)


@dataclass
class TrainingConfig:
    METRIC_THRESH: float = 0.25
    MODEL_NAME: str = "efficientnet_v2_s"
    FREEZE_BACKBONE: bool = False


def get_model(model_name: str, num_classes: int, freeze_backbone: bool = True):
    """A helper function to load and prepare any classification model
    available in Torchvision for transfer learning or fine-tuning."""

    model = getattr(torchvision.models, model_name)(weights="DEFAULT")

    if freeze_backbone:
        # Set all layer to be non-trainable
        for param in model.parameters():
            param.requires_grad = False

    model_childrens = [name for name, _ in model.named_children()]

    try:
        final_layer_in_features = getattr(model, f"{model_childrens[-1]}")[-1].in_features
    except Exception as e:
        final_layer_in_features = getattr(model, f"{model_childrens[-1]}").in_features

    new_output_layer = nn.Linear(in_features=final_layer_in_features, out_features=num_classes)

    try:
        getattr(model, f"{model_childrens[-1]}")[-1] = new_output_layer
    except:
        setattr(model, model_childrens[-1], new_output_layer)

    return model


class ProteinModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        freeze_backbone: bool = False,
        init_lr: float = 0.001,
        optimizer_name: str = "Adam",
        weight_decay: float = 1e-4,
        use_scheduler: bool = False,
        f1_metric_threshold: float = 0.25,
    ):
        super().__init__()

        # Save the arguments as hyperparameters.
        self.save_hyperparameters()

        # Loading model using the function defined above.
        self.model = get_model(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            freeze_backbone=self.hparams.freeze_backbone,
        )

        # Intialize loss class.
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_train_f1 = MultilabelF1Score(num_labels=self.hparams.num_classes, average="macro", threshold=self.hparams.f1_metric_threshold)
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_f1 = MultilabelF1Score(num_labels=self.hparams.num_classes, average="macro", threshold=self.hparams.f1_metric_threshold)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, *args, **kwargs):
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits, target)

        self.mean_train_loss(loss, weight=data.shape[0])
        self.mean_train_f1(logits, target)

        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True)
        self.log("train/batch_f1", self.mean_train_f1, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Computing and logging the training mean loss & mean f1.
        self.log("train/loss", self.mean_train_loss, prog_bar=True)
        self.log("train/f1", self.mean_train_f1, prog_bar=True)
        self.log("step", self.current_epoch)

    def validation_step(self, batch, *args, **kwargs):
        data, target = batch  # Unpacking validation dataloader tuple
        logits = self(data)
        loss = self.loss_fn(logits, target)

        self.mean_valid_loss.update(loss, weight=data.shape[0])
        self.mean_valid_f1.update(logits, target)

    def on_validation_epoch_end(self):
        # Computing and logging the validation mean loss & mean f1.
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True)
        self.log("valid/f1", self.mean_valid_f1, prog_bar=True)
        self.log("step", self.current_epoch)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    self.trainer.max_epochs // 2,
                ],
                gamma=0.1,
            )

            # The lr_scheduler_config is a dictionary that contains the scheduler
            # and its associated configuration.
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "name": "multi_step_lr",
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        else:
            return optimizer


@torch.inference_mode()
def predict(input_image, threshold=0.25, model=None, preprocess_fn=None, device="cpu", idx2labels=None):
    input_tensor = preprocess_fn(input_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Generate predictions
    output = model(input_tensor).cpu()

    probabilities = torch.sigmoid(output)[0].numpy().tolist()

    output_probs = dict()
    predicted_classes = []

    for idx, prob in enumerate(probabilities):
        output_probs[idx2labels[idx]] = prob
        if prob >= threshold:
            predicted_classes.append(idx2labels[idx])

    predicted_classes = "\n".join(predicted_classes)
    return predicted_classes, output_probs, input_image #instead of input_image here return segmentation image


if __name__ == "__main__":
    # labels = { #labels from the protein model
    #     0: "Mitochondria",
    #     1: "Nuclear bodies",
    #     2: "Nucleoli",
    #     3: "Golgi apparatus",
    #     4: "Nucleoplasm",
    #     5: "Nucleoli fibrillar center",
    #     6: "Cytosol",
    #     7: "Plasma membrane",
    #     8: "Centrosome",
    #     9: "Nuclear speckles",
    # }
    labels = {
        0: "舌有瘀點", 
        1: "舌質白淡",
        2: "舌質淡紅",
        3: "舌質紅",
        4: "舌質紫",
        5: "舌有齒痕",
        6: "舌胖大",
        7: "舌有裂紋",
        8: "舌尖紅",
        9: "舌苔白",
        10: "舌苔薄",
        11: "舌苔膩",
        12: "舌濕_苔水滑",
    }

    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    CKPT_PATH = os.path.join(os.getcwd(), r"ckpt_022-vloss_0.1756_vf1_0.7919.ckpt")
    model = ProteinModel.load_from_checkpoint(CKPT_PATH)
    model.to(DEVICE)
    model.eval()
    _ = model(torch.randn(1, DatasetConfig.CHANNELS, *DatasetConfig.IMAGE_SIZE[::-1], device=DEVICE))

    preprocess = TF.Compose(
        [
            TF.Resize(size=DatasetConfig.IMAGE_SIZE[::-1]),
            TF.ToTensor(),
            TF.Normalize(DatasetConfig.MEAN, DatasetConfig.STD, inplace=True),
        ]
    )

    images_dir = glob(os.path.join(os.getcwd(), "samples") + os.sep + "*.png")
    examples = [[i, TrainingConfig.METRIC_THRESH] for i in np.random.choice(images_dir, size=10, replace=False)]
    # print(examples)
    

    
    
    with gr.Interface(
        fn=partial(predict, model=model, preprocess_fn=preprocess, device=DEVICE, idx2labels=labels),
        inputs=[
            gr.Image(type="pil", label="Image"),
            gr.Slider(0.0, 1.0, value=0.25, label="Threshold", info="Select the cut-off threshold for a node to be considered as a valid output."),
        ],
        outputs=[
            gr.Textbox(label="Labels Present"),
            gr.Label(label="Probabilities", show_label=False),
            gr.Image(label="Segmentation") #here is segmentation image
        ],
       
        examples=examples,
        cache_examples=False,
        allow_flagging="never",
        title="Awan AI Medical Image Classification",
        description="Upload picture of tongue or take picture with webcam.",
        theme=gr.themes.Soft(primary_hue="sky"),
    ) as iface:
        iface.launch(share=True)