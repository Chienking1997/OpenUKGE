import torch


def mean_ndcg(hr_map, model, device):
    ndcg_sum = 0  # nDCG with linear gain
    exp_ndcg_sum = 0  # nDCG with exponential gain
    count = 0

    for h in hr_map:
        for r in hr_map[h]:
            tw_dict = hr_map[h][r]  # {t:w}
            tw_truth = sorted(tw_dict.items(), key=lambda x: x[1], reverse=True)  # sort by weight descending
            ndcg_score, exp_ndcg_score = ndcg(int(h), int(r), tw_truth, model, device)  # compute nDCG
            ndcg_sum += ndcg_score
            exp_ndcg_sum += exp_ndcg_score
            count += 1

    return ndcg_sum / count, exp_ndcg_sum / count


def ndcg(h, r, tw_truth, model, device):
    h, r = torch.tensor(h).to(device), torch.tensor(r).to(device)
    ts = torch.tensor([int(t) for t, _ in tw_truth], dtype=torch.int64).to(device)
    ranks = get_t_ranks(h, r, ts, model, device)

    # linear gain
    gains = torch.tensor([w for _, w in tw_truth], dtype=torch.float32).to(device)
    discounts = torch.log2(ranks.float() + 1).to(device)
    discounted_gains = gains / discounts
    dcg = discounted_gains.sum()  # discounted cumulative gain

    # normalize
    max_possible_dcg = (gains / torch.log2(torch.arange(len(gains), dtype=torch.float32).to(device) + 2)).sum()  # max DCG
    ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain

    # exponential gain
    exp_gains = 2 ** gains - 1
    exp_discounted_gains = exp_gains / discounts
    exp_dcg = exp_discounted_gains.sum()

    # normalize
    exp_max_possible_dcg = (exp_gains / torch.log2(torch.arange(len(exp_gains), dtype=torch.float32).to(device) + 2)).sum()
    exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain

    return ndcg.item(), exp_ndcg.item()


def get_t_ranks(h, r, ts, model, device):
    scores = model.get_hrt_score(h, r, ts)

    all_scores = model.get_tail_score(h, r)

    # 计算排名增加
    rank_plus = (scores.unsqueeze(1) < all_scores).int().sum(dim=1) + 1

    return rank_plus
