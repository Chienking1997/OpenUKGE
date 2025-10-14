import torch
import torch.nn as nn

def ece_t(triples, probabilities, model):
    pred_score = torch.zeros(5).cuda()
    test_split, confi, split_avg, ocu = split_tri(triples, probabilities, 5)
    for i in range(len(test_split)):
        score = model(test_split[i])
        score = torch.clamp((score * 5).to(torch.long), 0, 4)
        pred_score[i] = (score == i).float().mean()
    ece = torch.abs(pred_score - split_avg) * ocu
    ece = ece.sum()
    return ece.item()




def split_tri(triples, probabilities, num_partitions=5):
    triples = triples.tolist()
    probabilities = probabilities.tolist()
    partitions = [[] for _ in range(num_partitions)]
    confi = [[] for _ in range(num_partitions)]
    split_avg = [[0] for _ in range(num_partitions)]
    ocu_tri = [[0] for _ in range(num_partitions)]
    # Place each triple in the appropriate partition based on its confidence level
    for index, triple in enumerate(triples):
        confidence = probabilities[index]
        partition_index = min(int(confidence * num_partitions), num_partitions - 1)
        partitions[partition_index].append(triple)
        confi[partition_index].append(confidence)
    for i in range(num_partitions):
        partitions[i] = torch.tensor(partitions[i], dtype=torch.int64).cuda()
        split_avg[i] = sum(confi[i])/len(confi[i])
        ocu_tri[i] = len(confi[i])/len(triples)

    return partitions, confi, torch.tensor(split_avg).cuda(), torch.tensor(ocu_tri).cuda()