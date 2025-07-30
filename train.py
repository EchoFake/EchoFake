import lightning as L
import torch

from models import RawNet2, AASIST, W2VLinear
from utils import *
from data import EchoFake


class BaseTrainer(L.LightningModule):
    def on_validation_epoch_start(self):
        self.labels = []
        self.preds = []
        self.logits = []
        self.target_scores = []
        self.nontarget_scores = []

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        emb, logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())

    def on_validation_epoch_end(self):

        precision, recall, f1 = compute_f1(self.labels, self.preds)
        eer = compute_eer(self.preds, self.labels)

        self.log("val/precision", precision, sync_dist=True)
        self.log("val/recall", recall, sync_dist=True)
        self.log("val/f1", f1, sync_dist=True)
        self.log("val/eer", eer, sync_dist=True)

    def on_test_epoch_start(self):
        self.labels = []
        self.preds = []
        self.logits = []
        self.target_scores = []
        self.nontarget_scores = []

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        emb, logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())
        self.logits.extend(logits[:, 1].tolist())
        # if self.config["num_classes"] == 2:
        #     self.logits.extend(logits[:, 1].tolist())
        for i in range(x.size(0)):
            if y[i] == 1:
                self.target_scores.append(logits[i, 1].item())
            else:
                self.nontarget_scores.append(logits[i, 1].item())

    def on_test_epoch_end(self):

        precision, recall, f1 = compute_f1(self.labels, self.preds)

        self.log("test/precision", precision, sync_dist=True)
        self.log("test/recall", recall, sync_dist=True)
        self.log("test/f1", f1, sync_dist=True)
        eer = compute_eer_by_scores(self.target_scores, self.nontarget_scores)
        self.log("test/eer", eer, sync_dist=True)

        output = self.logger.log_dir + "/output.txt"
        with open(output, "w") as f:
            for label, pred, score in zip(self.labels, self.preds, self.logits):
                f.write(f"{label} {pred} {score}\n")
        print(f"Test results saved to {output}")


class RawNet2Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = RawNet2(num_classes=config["train"]["num_classes"])
        # weight = torch.FloatTensor([0.1, 0.9])
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        lr = float(self.config["optimizer"]["lr"])
        weight_decay = float(self.config["optimizer"]["weight_decay"])
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        emb, logits = self.model(x)

        loss = self.ce_loss(logits, y)
        self.log("train/loss", loss)

        return loss


class AASISTTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = AASIST(
            d_args=config["d_args"], num_classes=config["train"]["num_classes"]
        )
        # states = torch.load("models/weights/AASIST.pth")
        # self.model.load_state_dict(states)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        lr = float(self.config["optimizer"]["lr"])
        weight_decay = float(self.config["optimizer"]["weight_decay"])
        betas = tuple(self.config["optimizer"]["betas"])
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        min_lr = float(self.config["scheduler"]["min_lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                min_lr / lr,
            ),
        )
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        emb, logits = self.model(x)

        loss = self.ce_loss(logits, y)
        self.log("train/loss", loss)

        return loss


class W2VTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = W2VLinear()
        # states = torch.load("models/weights/conf-3-linear.pth")
        # self.model.load_state_dict(states)

        self.loss_ce = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        lr = float(self.config["optimizer"]["lr"])
        weight_decay = float(self.config["optimizer"]["weight_decay"])
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        min_lr = float(self.config["scheduler"]["min_lr"])
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=min_lr,
            max_lr=lr,
            step_size_up=3,
            mode="exp_range",
            gamma=0.85,
            cycle_momentum=False,
        )
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        self.train_num_total = 0
        self.train_num_correct = 0

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        out, feats, emb = self.model(x)

        bs = out.size(0)
        self.train_num_total += bs
        _, batch_pred = out.max(dim=1)
        self.train_num_correct += (batch_pred == y).sum(dim=0).item()
        loss = self.loss_ce(out, y)
        # feats = feats.unsqueeze(1)
        # loss_cf1 = 1 / bs * supcon_loss(feats, labels=y)
        # emb = emb.unsqueeze(1)
        # emb = emb.unsqueeze(-1)
        # loss_cf2 = 1 / bs * supcon_loss(emb, labels=y)
        # loss = loss_ce + loss_cf1 + loss_cf2

        # self.log("train/loss_ce", loss_ce, sync_dist=True)
        # self.log("train/loss_cf1", loss_cf1, sync_dist=True)
        # self.log("train/loss_cf2", loss_cf2, sync_dist=True)
        self.log("train/total_loss", loss, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log(
            "train_acc",
            self.train_num_correct / self.train_num_total,
            sync_dist=True,
        )
        self.train_num_total = 0
        self.train_num_correct = 0

    def validation_step(self, val_batch, batch_idx):
        x, y, *_ = val_batch
        out, *_ = self.model(x)

        preds = torch.argmax(out, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())

    def test_step(self, test_batch, batch_idx):
        x, y, *_ = test_batch
        out, *_ = self.model(x)

        for i in range(x.size(0)):
            if y[i] == 1:
                self.target_scores.append(out[i, 1].item())
            else:
                self.nontarget_scores.append(out[i, 1].item())

        preds = torch.argmax(out, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())
        self.logits.extend(out[:, 1].tolist())
