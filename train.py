import logging
import sys
import os
from models import UNet_regressor, DiceCELoss
from dataloader import get_train_loader
import torch
import monai

monai.config.print_config()
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

    train_loader, val_loader = get_train_loader()
    epoch_num = 100
    lr = 1e-4
    val_interval = 2
    loss_function = DiceCELoss()
    opt = torch.optim.Adam(selfnet.parameters(), lr=lr)

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
            print(batch_num)
            outputs, reg = selfnet(inputs)
            loss = loss_function(outputs, reg, labels)
            batch_num += 1
            print("loss = {:.8f}, epoch = {}, batch = {}".format(loss.item(), epoch, batch_num))
            loss.backward()
            opt.step()

        val_loss = 0
        for val_data in val_loader:
            inputs, labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            outputs, reg = selfnet(inputs)
            loss = loss_function(outputs, reg, labels)
            val_loss += loss.item()

        print("Validation loss ={:.8f}".format(val_loss))
        torch.save({
            'epoch': epoch,
            'model_state_dict': selfnet.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, os.path.join("checkpoints","net.pt"))