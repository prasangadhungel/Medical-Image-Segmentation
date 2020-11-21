import logging
import sys
import os
from models import UNet_regressor, DiceCELoss
from dataloader import get_train_loader
import torch
import monai
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice

monai.utils.set_determinism(seed=0)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    selfnet = UNet_regressor(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
        ).to(device)
    ckpt = os.path.join("checkpoints", "net.pt")
    selfnet.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])

    train_loader, val_loader = get_train_loader()
    epoch_num = 100
    lr = 1e-4
    val_interval = 2
    loss_function = DiceCELoss()
    opt = torch.optim.Adam(selfnet.parameters(), lr=lr)
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)

    for epoch in range(epoch_num):
        batch_num = 0
        selfnet.train()
        step = 0
        for batch_data in train_loader:
            batch_num += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            opt.zero_grad()
            outputs, reg = selfnet(inputs)
            loss = loss_function(outputs, reg, labels)
            batch_num += 1
            print("loss = {:.8f}, epoch = {}, batch = {}".format(loss.item(), epoch, batch_num))
            loss.backward()
            opt.step()

        if (epoch + 1) % val_interval == 0:
            selfnet.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (192, 192, 16)
                    sw_batch_size, overlap = 2, 0.5
                    val_outputs = sliding_window_inference(inputs=val_inputs,
                                                           roi_size=roi_size,
                                                           sw_batch_size=sw_batch_size,
                                                           predictor=selfnet,
                                                           overlap=overlap,
                                                           mode="gaussian",
                                                           padding_mode="replicate",
                                                           )
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    value = compute_meandice(
                        y_pred=val_outputs,
                        y=val_labels,
                        include_background=False,
                    )
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': selfnet.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                        }, os.path.join("checkpoints", "net.pt"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
