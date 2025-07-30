import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# from torchaudio.pipelines import WAV2VEC2_XLSR_300M
import fairseq

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SSLModel(nn.Module):
    def __init__(self):
        super(SSLModel, self).__init__()

        cp_path = os.path.join(BASE_DIR, "./weights/xlsr2_300m.pt")
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_path]
        )
        self.model = model[0]

        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)["x"]
        # print(emb.shape)
        return emb


class BackEnd(nn.Module):
    """Back End Wrapper"""

    def __init__(
        self, input_dim, out_dim, num_classes, dropout_rate, dropout_flag=True
    ):
        super(BackEnd, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes

        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag

        # a simple full-connected network for frame-level feature processing
        self.m_frame_level = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            nn.Linear(self.in_dim, self.out_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate),
        )
        # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag))

        # linear layer to produce output logits
        self.m_utt_level = nn.Linear(self.out_dim, self.num_class)

        return

    def forward(self, feat):
        """logits, emb_vec = back_end_emb(feat)

        input:
        ------
          feat: tensor, (batch, frame_num, feat_feat_dim)

        output:
        -------
          logits: tensor, (batch, num_output_class)
          emb_vec: tensor, (batch, emb_dim)

        """
        # through the frame-level network
        # (batch, frame_num, self.out_dim)
        feat_ = self.m_frame_level(feat)

        # average pooling -> (batch, self.out_dim)
        feat_utt = feat_.mean(1)

        # output linear
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt


class W2VLinear(nn.Module):
    def __init__(self):
        super().__init__()

        self.ssl_model = SSLModel()
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.loss_CE = nn.CrossEntropyLoss()
        self.backend = BackEnd(128, 128, 2, 0.5, False)

        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)
        ).mean(0)

        ################ Load pretrained model ################
        # pretrained_dict = torch.load("./pretrained/epoch_30.pth")
        # self.load_state_dict(pretrained_dict)

    def forward(self, x):
        if next(self.ssl_model.parameters()).device != x.device:
            self.ssl_model.model.to(x.device)

        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)
        feats = x
        x = nn.ReLU()(x)

        # output [batch, 2]
        # emb [batch, 128]
        output, emb = self.backend(x)
        output = F.log_softmax(output, dim=1)

        return output, feats, emb


if __name__ == "__main__":
    x = torch.randn(32, 64000).to("cuda:1")
    model = W2VLinear().to("cuda:1")
    states = torch.load("models/weights/epoch_30.pth")
    model.load_state_dict(states)
    out, feats, emb = model(x)

    print(out.shape, feats.shape, emb.shape)
