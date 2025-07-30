import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    if x_len == max_len:
        return x

    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt : stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def preprocess(batch):
    wav = batch["path"]["array"]  # np.ndarray [T]
    wav = pad_random(wav, max_len=64000).astype(np.float32)
    return {"input_values": wav, "labels": batch["label"]}


def preprocess_for_test(batch):
    wav = batch["path"]["array"]  # np.ndarray [T]
    wav = pad(wav, max_len=64000).astype(np.float32)
    return {"input_values": wav, "labels": batch["label"]}


def data_collator(features):
    input_values = [f["input_values"] for f in features]
    labels = [f["label"] for f in features]
    return {
        "input_values": torch.tensor(input_values),
        "labels": torch.tensor(labels),
    }


def compute_f1(labels, preds, average="weighted"):
    precision = precision_score(labels, preds, average=average)
    recall = recall_score(labels, preds, average=average)
    f1 = f1_score(labels, preds, average=average)
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1


# def compute_metrics(labels, preds, scores, average="weighted"):
#     precision = precision_score(labels, preds, average=average)
#     recall = recall_score(labels, preds, average=average)
#     f1 = f1_score(labels, preds, average=average)
#     auc = roc_auc_score(labels, scores)
#     return precision, recall, f1, auc


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )
    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size)
    )  # false rejection rates
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )  # Thresholds are the sorted scores
    return frr, far, thresholds


def compute_eer(pred, label):
    """Returns equal error rate (EER) and the corresponding threshold."""
    if isinstance(pred, torch.Tensor):
        pred, label = pred.cpu().numpy(), label.cpu().numpy()

    pred, label = np.array(pred).flatten(), np.array(label).flatten()
    target_scores = pred[label == 1.0]
    nontarget_scores = pred[label == 0.0]
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer  # , thresholds[min_index]


def compute_eer_by_scores(target_scores, nontarget_scores):
    """Returns equal error rate (EER) and the corresponding threshold."""
    target_scores = np.array(target_scores).flatten()
    nontarget_scores = np.array(nontarget_scores).flatten()
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer  # , thresholds[min_index]


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def supcon_loss(
    input_feat,
    labels=None,
    mask=None,
    sim_metric=lambda mat1, mat2: torch.bmm(
        mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)
    ).mean(0),
    t=0.07,
    contra_mode="all",
    length_norm=False,
):
    """
    loss = SupConLoss(feat,
                      labels = None, mask = None, sim_metric = None,
                      t=0.07, contra_mode='all')
    input
    -----
      feat: tensor, feature vectors z [bsz, n_views, ...].
      labels: ground truth of shape [bsz].
      mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
      sim_metric: func, function to measure the similarity between two
            feature vectors
      t: float, temperature
      contra_mode: str, default 'all'
         'all': use all data in class i as anchors
         'one': use 1st data in class i as anchors
      length_norm: bool, default False
          if True, l2 normalize feat along the last dimension

    output
    ------
      A loss scalar.

    Based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    Example:
      feature = torch.rand([16, 2, 1000], dtype=torch.float32)
      feature = torch_nn_func.normalize(feature, dim=-1)
      label = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 1, 1, 1, 1, 1],
               dtype=torch.long)
      loss = supcon_loss(feature, labels=label)
    """
    if length_norm:
        feat = F.normalize(input_feat, dim=-1)
    else:
        feat = input_feat

    # batch size
    bs = feat.shape[0]
    # device
    dc = feat.device
    # dtype
    dt = feat.dtype
    # number of view
    nv = feat.shape[1]

    # get the mask
    # mask[i][:] indicates the data that has the same class label as data i
    if labels is not None and mask is not None:
        raise ValueError("Cannot define both `labels` and `mask`")
    elif labels is None and mask is None:
        mask = torch.eye(bs, dtype=dt, device=dc)
    elif labels is not None:
        labels = labels.view(-1, 1)
        if labels.shape[0] != bs:
            raise ValueError("Num of labels does not match num of features")
        mask = torch.eq(labels, labels.T).type(dt).to(dc)
    else:
        mask = mask.type(dt).to(dc)

    # prepare feature matrix
    # -> (num_view * batch, feature_dim, ...)
    contrast_feature = torch.cat(torch.unbind(feat, dim=1), dim=0)
    #
    if contra_mode == "one":
        # (batch, feat_dim, ...)
        anchor_feature = feat[:, 0]
        anchor_count = 1
    elif contra_mode == "all":
        anchor_feature = contrast_feature
        anchor_count = nv
    else:
        raise ValueError("Unknown mode: {}".format(contra_mode))

    # compute logits
    # logits_mat is a matrix of size [num_view * batch, num_view * batch]
    # or [batch, num_view * batch]

    if sim_metric is not None:
        logits_mat = torch.div(sim_metric(anchor_feature, contrast_feature), t)
    else:
        logits_mat = torch.div(torch.matmul(anchor_feature, contrast_feature.T), t)

    # print(anchor_feature.shape)
    # mask based on the label
    # -> same shape as logits_mat
    mask_ = mask.repeat(anchor_count, nv)
    # mask on each data itself (
    self_mask = torch.scatter(
        torch.ones_like(mask_), 1, torch.arange(bs * anchor_count).view(-1, 1).to(dc), 0
    )
    # print(self_mask)
    #
    mask_ = mask_ * self_mask
    # print(mask_)

    # for numerical stability, remove the max from logits
    # see https://en.wikipedia.org/wiki/LogSumExp trick
    # for numerical stability
    logits_max, _ = torch.max(logits_mat * self_mask, dim=1, keepdim=True)
    logits_mat_ = logits_mat - logits_max.detach()
    # compute log_prob
    exp_logits = torch.exp(logits_mat_ * self_mask) * self_mask
    log_prob = logits_mat_ - torch.log(exp_logits.sum(1, keepdim=True))

    # print("log_prob.shape", log_prob.shape)
    # compute mean of log-likelihood over positive
    # print(mask_ * log_prob)
    mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)
    # print("mean_log_prob_pos.shape", mean_log_prob_pos.shape)
    # print(mean_log_prob_pos)
    # loss
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, bs).mean()

    return loss
