import logging
import sys
import os
from models import UNet_regressor, DeepLabV3_3D
from losses import DiceCELoss
from dataloader import get_train_loader
import torch
import monai
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice

monai.utils.set_determinism(seed=0)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    net = DeepLabV3_3D(num_classes=2,input_channels=1).to(device)
    checkpoint = torch.load(os.path.join("checkpoints", "Latest.pt"))
    net.load_state_dict(checkpoint['model_state_dict'])
    max_epochs, lr, momentum = 500, 1e-4, 0.95
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    # curr_epoch = 0
    curr_epoch = checkpoint['epoch']
    print(curr_epoch)
    val_interval = 2
    epoch_num = 500
    loss_function = DiceCELoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)
    writer = SummaryWriter("DeeplabTB")

    train_loader, val_loader = get_train_loader()
    n_train = len(train_loader)
    lr = 1e-4
    val_interval = 2
    loss_function = DiceCELoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)

    for epoch in range(curr_epoch + 1, epoch_num):
        batch_num = 0
        net.train()
        step = 0
        for batch_data in train_loader:
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            opt.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            batch_num += 1
            print("loss = {:.8f}, epoch = {}, batch = {}".format(loss.item(), epoch, batch_num))
            writer.add_scalar("train_loss", loss.item(), n_train * epoch + batch_num)
            loss.backward()
            opt.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, os.path.join("DLcheckpoints", "Latest.pt"))

        if epoch + 1 == 120:
            for param_group in opt.param_groups:
                param_group['lr'] = 5e-5

        if epoch + 1 == 220:
            for param_group in opt.param_groups:
                param_group['lr'] = 1e-5

        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (192, 192, 24)
                    sw_batch_size, overlap = 2, 0.5
                    val_outputs = sliding_window_inference(inputs=val_inputs,
                                                           roi_size=roi_size,
                                                           sw_batch_size=sw_batch_size,
                                                           predictor=net,
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
                    torch.save(net.state_dict(), os.path.join("DLcheckpoints", "deeplab" + str(metric) + ".pt"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                plot_2d_or_3d_image(val_inputs, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")