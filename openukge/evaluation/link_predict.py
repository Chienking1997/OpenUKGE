import torch
from tqdm.auto import trange


def link_predict(test_triples, test_pro, model, test_data):
    num_partitions = test_data["num_partitions"]
    h_rank_raw, h_rank_filter, t_rank_raw, t_rank_filter = compute_ranks(test_triples, test_pro, model, test_data,
                                                                         num_partitions, prediction_mode='average')

    # Calculate metrics
    metrics_raw = calculate_all_metrics(h_rank_raw, t_rank_raw)
    metrics_filtered = calculate_all_metrics(h_rank_filter, t_rank_filter)

    # Print metrics as a table
    print_metrics(metrics_raw, metrics_filtered)


def high_link_predict(test_triples, test_pro, model, test_data):
    h_rank_raw, h_rank_filter, t_rank_raw, t_rank_filter \
        = compute_high_ranks(test_triples, model, test_data, prediction_mode='average')

    metrics_raw = calculate_all_metrics(h_rank_raw, t_rank_raw)
    metrics_filtered = calculate_all_metrics(h_rank_filter, t_rank_filter)

    print_high_metrics(metrics_raw, metrics_filtered)


def weight_link_predict(test_triples, test_pro, model, test_data):
    (h_rank, h_rank_filter,
     t_rank, t_rank_filter) = compute_weight_ranks(test_triples, test_pro,
                                                   model, test_data, prediction_mode='average')
    sum_pro = torch.sum(test_pro)
    wmr_h_raw = torch.sum(h_rank * test_pro) / sum_pro
    wmrr_h_raw = torch.sum((1.0 / h_rank) * test_pro) / sum_pro
    wh20_h_raw = torch.sum((h_rank <= 20).float() * test_pro) / sum_pro
    wh40_h_raw = torch.sum((h_rank <= 40).float() * test_pro) / sum_pro

    wmr_h_filter = torch.sum(h_rank_filter * test_pro) / sum_pro
    wmrr_h_filter = torch.sum((1.0 / h_rank_filter) * test_pro) / sum_pro
    wh20_h_filter = torch.sum((h_rank_filter <= 20).float() * test_pro) / sum_pro
    wh40_h_filter = torch.sum((h_rank_filter <= 40).float() * test_pro) / sum_pro

    wmr_t_raw = torch.sum(t_rank * test_pro) / sum_pro
    wmrr_t_raw = torch.sum((1.0 / t_rank) * test_pro) / sum_pro
    wh20_t_raw = torch.sum((t_rank <= 20).float() * test_pro) / sum_pro
    wh40_t_raw = torch.sum((t_rank <= 40).float() * test_pro) / sum_pro

    wmr_t_filter = torch.sum(t_rank_filter * test_pro) / sum_pro
    wmrr_t_filter = torch.sum((1.0 / t_rank_filter) * test_pro) / sum_pro
    wh20_t_filter = torch.sum((t_rank_filter <= 20).float() * test_pro) / sum_pro
    wh40_t_filter = torch.sum((t_rank_filter <= 40).float() * test_pro) / sum_pro

    metrics = {
        'MR': (wmr_h_raw, wmr_t_raw, wmr_h_filter, wmr_t_filter),
        'MRR': (wmrr_h_raw, wmrr_t_raw, wmrr_h_filter, wmrr_t_filter),
        'Hit@20': (wh20_h_raw, wh20_t_raw, wh20_h_filter, wh20_t_filter),
        'Hit@40': (wh40_h_raw, wh40_t_raw, wh40_h_filter, wh40_t_filter)
    }
    print_weight_metrics(metrics)


def val_link_predict(val_triples, val_pro, model, val_data, prediction_mode='tail'):
    num_partitions = val_data["num_partitions"]
    h_rank_raw, h_rank_filter, t_rank_raw, t_rank_filter = compute_ranks(val_triples, val_pro, model, val_data,
                                                                         num_partitions, prediction_mode)

    # Calculate metrics for raw and filtered ranks
    if prediction_mode == 'head':
        mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw = calculate_metrics(h_rank_raw)
        mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter = calculate_metrics(h_rank_filter)
    elif prediction_mode == 'tail':
        mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw = calculate_metrics(t_rank_raw)
        mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter = calculate_metrics(t_rank_filter)
    else:  # 'average'
        h_raw = calculate_metrics(h_rank_raw)
        t_raw = calculate_metrics(t_rank_raw)
        h_filter = calculate_metrics(h_rank_filter)
        t_filter = calculate_metrics(t_rank_filter)
        mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw = avg_metrics(h_raw, t_raw)
        mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter = avg_metrics(h_filter, t_filter)

    return (mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw,
            mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter)


def val_high_link_predict(val_triples, val_pro, model, val_data, prediction_mode='tail'):
    h_rank_raw, h_rank_filter, t_rank_raw, t_rank_filter \
        = compute_high_ranks(val_triples, model, val_data, prediction_mode)

    # Calculate metrics for raw and filtered ranks
    if prediction_mode == 'head':
        mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw = calculate_metrics(h_rank_raw)
        mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter = calculate_metrics(h_rank_filter)
    elif prediction_mode == 'tail':
        mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw = calculate_metrics(t_rank_raw)
        mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter = calculate_metrics(t_rank_filter)
    else:  # 'average'
        h_raw = calculate_metrics(h_rank_raw)
        t_raw = calculate_metrics(t_rank_raw)
        h_filter = calculate_metrics(h_rank_filter)
        t_filter = calculate_metrics(t_rank_filter)
        mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw = avg_metrics(h_raw, t_raw)
        mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter = avg_metrics(h_filter, t_filter)

    return (mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw,
            mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter)


def val_weight_link_predict(val_triples, val_pro, model, val_data, prediction_mode='tail'):
    weight_h_rank, h_rank_filter, weight_t_rank, t_rank_filter \
        = compute_weight_ranks(val_triples, val_pro, model, val_data, prediction_mode)
    sum_pro = torch.sum(val_pro)

    # Calculate metrics for raw and filtered ranks
    if prediction_mode == 'head':
        w_mr_raw = torch.sum(weight_h_rank * val_pro) / sum_pro
        w_mrr_raw = torch.sum((1.0 / weight_h_rank) * val_pro) / sum_pro
        w_h20_raw = torch.sum((weight_h_rank <= 20).float() * val_pro) / sum_pro
        w_h40_raw = torch.sum((weight_h_rank <= 40).float() * val_pro) / sum_pro
    elif prediction_mode == 'tail':
        w_mr_raw = torch.sum(weight_t_rank * val_pro) / sum_pro
        w_mrr_raw = torch.sum((1.0 / weight_t_rank) * val_pro) / sum_pro
        w_h20_raw = torch.sum((weight_t_rank <= 20).float() * val_pro) / sum_pro
        w_h40_raw = torch.sum((weight_t_rank <= 40).float() * val_pro) / sum_pro

    else:  # 'average'
        mr_h_raw = torch.sum(weight_h_rank * val_pro) / sum_pro
        mrr_h_raw = torch.sum((1.0 / weight_h_rank) * val_pro) / sum_pro
        h20_h_raw = torch.sum((weight_h_rank <= 20).float() * val_pro) / sum_pro
        h40_h_raw = torch.sum((weight_h_rank <= 40).float() * val_pro) / sum_pro
        mr_t_raw = torch.sum(weight_t_rank * val_pro) / sum_pro
        mrr_t_raw = torch.sum((1.0 / weight_t_rank) * val_pro) / sum_pro
        h20_t_raw = torch.sum((weight_t_rank <= 20).float() * val_pro) / sum_pro
        h40_t_raw = torch.sum((weight_t_rank <= 40).float() * val_pro) / sum_pro
        w_mr_raw, w_mrr_raw, w_h20_raw, w_h40_raw = avg_metrics((mr_h_raw, mrr_h_raw, h20_h_raw, h40_h_raw),
                                                                (mr_t_raw, mrr_t_raw, h20_t_raw, h40_t_raw))

    return w_mr_raw, w_mrr_raw, w_h20_raw, w_h40_raw


def compute_ranks(triples, pro, model, data, num_partitions, prediction_mode='average'):
    len_data = data["len_triples"]
    device = triples.device
    h_rank_raw = torch.zeros(len_data).to(device)
    h_rank_filter = torch.zeros(len_data).to(device)
    t_rank_raw = torch.zeros(len_data).to(device)
    t_rank_filter = torch.zeros(len_data).to(device)

    for i in trange(len_data, leave=False):
        h, r, t = triples[i]
        p = pro[i]
        part = min(int(p * num_partitions), num_partitions - 1)

        if prediction_mode in ['head', 'average']:
            temp_rt = list(data["tr2h_partition_mappings"][part][(t.item(), r.item())])
            predict_h = model.get_head_score(t, r)
            dealt_h = torch.abs(predict_h - p)
            h_rank_raw[i] = 1 + torch.sum(dealt_h < dealt_h[h])
            h_rank_filter[i] = h_rank_raw[i] - torch.sum(dealt_h[temp_rt] < dealt_h[h])
        if prediction_mode in ['tail', 'average']:
            temp_hr = list(data["hr2t_partition_mappings"][part][(h.item(), r.item())])
            predict_t = model.get_tail_score(h, r)
            dealt_t = torch.abs(predict_t - p)
            t_rank_raw[i] = 1 + torch.sum(dealt_t < dealt_t[t])
            t_rank_filter[i] = t_rank_raw[i] - torch.sum(dealt_t[temp_hr] < dealt_t[t])

    return h_rank_raw, h_rank_filter, t_rank_raw, t_rank_filter


def compute_high_ranks(triples, model, data, prediction_mode='average'):
    len_data = data["len_high_triples"]
    device = triples.device
    h_rank_raw = torch.zeros(len_data).to(device)
    h_rank_filter = torch.zeros(len_data).to(device)
    t_rank_raw = torch.zeros(len_data).to(device)
    t_rank_filter = torch.zeros(len_data).to(device)

    for i in trange(len_data, leave=False):
        h, r, t = triples[i]
        if prediction_mode in ['head', 'average']:
            temp_rt = list(data["tr2h_high_score"][(t.item(), r.item())])
            predict_h = model.get_head_score(t, r)
            h_rank_raw[i] = 1 + torch.sum(predict_h > predict_h[h])
            h_rank_filter[i] = h_rank_raw[i] - torch.sum(predict_h[temp_rt] > predict_h[h])
        if prediction_mode in ['tail', 'average']:
            temp_hr = list(data["hr2t_high_score"][(h.item(), r.item())])
            predict_t = model.get_tail_score(h, r)
            t_rank_raw[i] = 1 + torch.sum(predict_t > predict_t[t])
            t_rank_filter[i] = t_rank_raw[i] - torch.sum(predict_t[temp_hr] > predict_t[t])

    return h_rank_raw, h_rank_filter, t_rank_raw, t_rank_filter


def compute_weight_ranks(triples, pro, model, data, prediction_mode):
    len_data = data["len_triples"]
    device = triples.device
    h_rank_raw = torch.zeros(len_data).to(device)
    t_rank_raw = torch.zeros(len_data).to(device)
    h_rank_filter = torch.zeros(len_data).to(device)
    t_rank_filter = torch.zeros(len_data).to(device)

    for i in trange(len_data, leave=False):
        h, r, t = triples[i]
        if prediction_mode in ['head', 'average']:
            temp_rt = list(data["tr2h_full"][(t.item(), r.item())])
            predict_h = model.get_head_score(t, r)
            h_rank_raw[i] = 1 + torch.sum(predict_h > predict_h[h])
            h_rank_filter[i] = h_rank_raw[i] - torch.sum(predict_h[temp_rt] > predict_h[h])
        if prediction_mode in ['tail', 'average']:
            temp_hr = list(data["hr2t_full"][(h.item(), r.item())])
            predict_t = model.get_tail_score(h, r)
            t_rank_raw[i] = 1 + torch.sum(predict_t > predict_t[t])
            t_rank_filter[i] = t_rank_raw[i] - torch.sum(predict_t[temp_hr] > predict_t[t])

    return h_rank_raw, h_rank_filter, t_rank_raw, t_rank_filter


def print_metrics(metrics_raw, metrics_filtered):
    print("\n Uncertain Metrics Summary:")
    print(f"{'Metric':<15} {'Head Raw':<10} {'Tail Raw':<10} {'Head Filtered':<15} "
          f"{'Tail Filtered':<15} {'Avg Raw':<10} {'Avg Filtered':<15}")
    print("-" * 90)
    for metric in metrics_raw:
        avg_raw = (metrics_raw[metric][0] + metrics_raw[metric][1]) / 2.0
        avg_filtered = (metrics_filtered[metric][0] + metrics_filtered[metric][1]) / 2.0
        print(f"U{metric:<15} {metrics_raw[metric][0]:<10.4f} {metrics_raw[metric][1]:<10.4f} "
              f"{metrics_filtered[metric][0]:<15.4f} {metrics_filtered[metric][1]:<15.4f} "
              f"{avg_raw:<10.4f} {avg_filtered:<15.4f}")
    print("-" * 90)


def print_high_metrics(metrics_raw, metrics_filtered):
    print("\n High Confidence Metrics Summary:")
    print(f"{'Metric':<15} {'Head Raw':<10} {'Tail Raw':<10} {'Head Filtered':<15} "
          f"{'Tail Filtered':<15} {'Avg Raw':<10} {'Avg Filtered':<15}")
    print("-" * 90)
    for metric in metrics_raw:
        avg_raw = (metrics_raw[metric][0] + metrics_raw[metric][1]) / 2.0
        avg_filtered = (metrics_filtered[metric][0] + metrics_filtered[metric][1]) / 2.0
        print(f"{metric:<15} {metrics_raw[metric][0]:<10.4f} {metrics_raw[metric][1]:<10.4f} "
              f"{metrics_filtered[metric][0]:<15.4f} {metrics_filtered[metric][1]:<15.4f} "
              f"{avg_raw:<10.4f} {avg_filtered:<15.4f}")
    print("-" * 90)


def print_weight_metrics(metrics):
    print("\n Weight Metrics Summary:")
    print(f"{'Metric':<15} {'Head Raw':<10} {'Tail Raw':<10} {'Head Filtered':<15} "
          f"{'Tail Filtered':<15} {'Avg Raw':<10} {'Avg Filtered':<15}")
    print("-" * 90)
    for metric in metrics:
        avg = (metrics[metric][0] + metrics[metric][1]) / 2.0
        avg_filtered = (metrics[metric][2] + metrics[metric][3]) / 2.0
        print(f"W{metric:<15} {metrics[metric][0]:<10.4f} {metrics[metric][1]:<10.4f}"
              f"{metrics[metric][2]:<15.4f} {metrics[metric][3]:<15.4f} "
              f"{avg:<10.4f} {avg_filtered:<15.4f}")
    print("-" * 90)


def calculate_all_metrics(h_rank, t_rank):
    h_metrics = calculate_metrics(h_rank)
    t_metrics = calculate_metrics(t_rank)

    return {
        'MR': (h_metrics[0], t_metrics[0]),
        'MRR': (h_metrics[1], t_metrics[1]),
        'Hit@1': (h_metrics[2], t_metrics[2]),
        'Hit@3': (h_metrics[3], t_metrics[3]),
        'Hit@10': (h_metrics[4], t_metrics[4]),
    }


def avg_metrics(h_rank, t_rank):
    h_metrics = torch.tensor(h_rank)
    t_metrics = torch.tensor(t_rank)
    avg_metric = (h_metrics + t_metrics) / 2.0
    return avg_metric


def calculate_metrics(rank_tensor):
    mr = torch.mean(rank_tensor)
    mrr = torch.mean(1.0 / rank_tensor)
    hit1 = torch.mean((rank_tensor <= 1).float())
    hit3 = torch.mean((rank_tensor <= 3).float())
    hit10 = torch.mean((rank_tensor <= 10).float())
    return mr, mrr, hit1, hit3, hit10
