import torch
import torch.nn as nn
import torch.nn.functional as F


class BEUrRE(nn.Module):
    def __init__(self, num_ent=None, num_rel=None, emb_dim=None, gumbel_beta=None,
                 min_init_value=None, delta_init_value=None, config=None):
        super(BEUrRE, self).__init__()
        if delta_init_value is None:
            delta_init_value = [-0.1, -0.001]
        if min_init_value is None:
            min_init_value = [1e-4, 0.01]
        self.euler_gamma = 0.57721566490153286060
        self.min_init_value = min_init_value
        self.delta_init_value = delta_init_value
        self.num_ent = num_ent
        self.num_rel = num_rel

        self.min_embedding = nn.Embedding(num_ent, emb_dim)
        self.delta_embedding = nn.Embedding(num_ent, emb_dim)
        self.rel_trans_for_head = nn.Embedding(num_rel, emb_dim)
        self.rel_scale_for_head = nn.Embedding(num_rel, emb_dim)
        self.rel_trans_for_tail = nn.Embedding(num_rel, emb_dim)
        self.rel_scale_for_tail = nn.Embedding(num_rel, emb_dim)
        self.init_emb()
        self.gumbel_beta = gumbel_beta
        self.alpha = 1e-16
        self.clamp_min = 0.0
        self.clamp_max = 1e10
        self.config = config

    def init_emb(self):
        nn.init.uniform_(self.min_embedding.weight, a=self.min_init_value[0], b=self.min_init_value[1])
        nn.init.uniform_(self.delta_embedding.weight, a=self.delta_init_value[0], b=self.delta_init_value[1])
        nn.init.normal_(self.rel_trans_for_head.weight, mean=0, std=1e-4)
        nn.init.normal_(self.rel_scale_for_head.weight, mean=1, std=0.2)
        nn.init.normal_(self.rel_trans_for_tail.weight, mean=0, std=1e-4)
        nn.init.normal_(self.rel_scale_for_tail.weight, mean=1, std=0.2)

    def tri2emb(self, triples):
        head_emb = self.ent_emb(triples[:, 0])
        relation_emb = self.rel_emb(triples[:, 1])
        tail_emb = self.ent_emb(triples[:, 2])
        return head_emb, relation_emb, tail_emb

    def forward(self, triples):
        """The functions used in the training phase

        """
        head_ids, rel_ids, tail_ids = triples[:, 0], triples[:, 1], triples[:, 2]
        head_boxes = self.transform_head_boxes(head_ids, rel_ids)
        tail_boxes = self.transform_tail_boxes(tail_ids, rel_ids)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on subject or object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        predictions = torch.exp(log_prob)
        return predictions

    def transform_head_boxes(self, head_ids, rel_ids):
        """
        Transforms head entity boxes based on the relation-specific parameters.
.
        """
        head_boxes = self.get_entity_boxes(head_ids)

        translations = self.rel_trans_for_head(rel_ids)
        scales = F.relu(self.rel_scale_for_head(rel_ids))

        # affine transformation
        head_boxes.min_embed += translations
        head_boxes.delta_embed *= scales
        head_boxes.max_embed = head_boxes.min_embed + head_boxes.delta_embed

        return head_boxes

    def transform_tail_boxes(self, tail_ids, rel_ids):
        """
        Transforms tail entity boxes based on the relation-specific parameters.

        Returns:
            tail_boxes: Transformed tail entity boxes after applying relation-specific affine transformations.
        """
        tail_boxes = self.get_entity_boxes(tail_ids)
        translations = self.rel_trans_for_tail(rel_ids)
        scales = F.relu(self.rel_scale_for_tail(rel_ids))

        # affine transformation
        tail_boxes.min_embed += translations
        tail_boxes.delta_embed *= scales
        tail_boxes.max_embed = tail_boxes.min_embed + tail_boxes.delta_embed

        return tail_boxes

    def get_entity_boxes(self, entities):
        """
        Retrieves entity boxes based on entity indices.
        """
        min_rep = self.min_embedding(entities)
        delta_rep = self.delta_embedding(entities)
        max_rep = min_rep + torch.exp(delta_rep)
        boxes = Box(min_rep, max_rep)
        return boxes

    def intersection(self, boxes1, boxes2):
        """
        Computes the intersection of two sets of boxes using the Gumbel softmax trick.

        Args:
            boxes1: The first set of entity boxes.
            boxes2: The second set of entity boxes.

        Returns:
            intersection_box: The intersection of the two sets of boxes.
        """
        intersections_min = self.gumbel_beta * torch.logsumexp(
            torch.stack((boxes1.min_embed / self.gumbel_beta, boxes2.min_embed / self.gumbel_beta)),
            0
        )
        intersections_min = torch.max(
            intersections_min,
            torch.max(boxes1.min_embed, boxes2.min_embed)
        )
        intersections_max = - self.gumbel_beta * torch.logsumexp(
            torch.stack((-boxes1.max_embed / self.gumbel_beta, -boxes2.max_embed / self.gumbel_beta)),
            0
        )
        intersections_max = torch.min(
            intersections_max,
            torch.min(boxes1.max_embed, boxes2.max_embed)
        )

        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box

    def log_volumes(self, boxes, temp=1., scale=1.):
        """
        Calculates the logarithm of the volumes of boxes.

        Args:
            boxes: A set of entity boxes.
            temp: The temperature parameter for the softplus function.
            scale: Scaling factor for the volumes.

        Returns:
            log_vol: The logarithm of the volumes of the given boxes.
        """
        eps = torch.finfo(boxes.min_embed.dtype).tiny  # type: ignore

        # if isinstance(scale, float):
        #     s = torch.tensor(scale)
        # else:
        #     s = scale

        log_vol = torch.sum(
            torch.log(
                F.softplus(boxes.delta_embed - 2 * self.euler_gamma * self.gumbel_beta, beta=temp).clamp_min(eps)
            ),
            dim=-1
        )

        return log_vol

    def transitive_rule_loss(self, triples, rule_config):
        subsets = [triples[triples[:, 1] == r] for r in rule_config['transitive']['relations']]
        sub_triples = torch.cat(subsets, dim=0)

        # only optimize relation parameters
        head_boxes = self.get_entity_boxes_detached(sub_triples[:, 0])
        tail_boxes = self.get_entity_boxes_detached(sub_triples[:, 2])
        head_boxes = self.head_transformation(head_boxes, sub_triples[:, 1])
        tail_boxes = self.tail_transformation(tail_boxes, sub_triples[:, 1])

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # P(f_r(epsilon_box)|g_r(epsilon_box)) should be 1
        vol_loss = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(tail_boxes)))
        return vol_loss

    def composition_rule_loss(self, triples, rule_config, device):
        def rels(size, rid):
            # fill a tensor with relation id
            return torch.full((size,), rid, dtype=torch.long).to(device)

        def biconditioning(boxes1, boxes2):
            intersection_boxes = self.intersection(boxes1, boxes2)
            log_intersection = self.log_volumes(intersection_boxes)
            # || 1-P(Box2|Box1) ||
            condition_on_box1 = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(boxes1)))
            # || 1-P(Box1|Box2) ||
            condition_on_box2 = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(boxes2)))
            loss = condition_on_box1 + condition_on_box2
            return loss

        vol_loss = 0
        for rule_combn in rule_config['composite']['relations']:
            r1, r2, r3 = rule_combn
            r1_triples = triples[triples[:, 1] == r1]
            r2_triples = triples[triples[:, 1] == r2]

            # use heads and tails from r1, r2 as reasonable entity samples to help optimize relation parameters
            if len(r1_triples) > 0 and len(r2_triples) > 0:
                # The Cartesian product returns a new tensor
                # containing all possible combinations of elements from the input tensors.
                entities = torch.cartesian_prod(r1_triples[:, 0], r2_triples[:, 2])
                head_ids, tail_ids = entities[:, 0], entities[:, 1]
                size = len(entities)

                # only optimize relation parameters
                head_boxes_r1r2 = self.get_entity_boxes_detached(head_ids)
                tail_boxes_r1r2 = self.get_entity_boxes_detached(tail_ids)
                r1r2_head = self.head_transformation(head_boxes_r1r2, rels(size, r1))
                r1r2_head = self.head_transformation(r1r2_head, rels(size, r2))
                r1r2_tail = self.tail_transformation(tail_boxes_r1r2, rels(size, r1))
                r1r2_tail = self.tail_transformation(r1r2_tail, rels(size, r2))

                # head_boxes_r1r2 have been modified in transformation
                # so make separate box objects with the same parameters
                head_boxes_r3 = self.get_entity_boxes_detached(head_ids)
                tail_boxes_r3 = self.get_entity_boxes_detached(tail_ids)
                r3_head = self.head_transformation(head_boxes_r3, rels(size, r3))
                r3_tail = self.tail_transformation(tail_boxes_r3, rels(size, r3))

                head_transform_loss = biconditioning(r1r2_head, r3_head)
                tail_transform_loss = biconditioning(r1r2_tail, r3_tail)
                vol_loss += head_transform_loss
                vol_loss += tail_transform_loss
        return vol_loss

    def get_entity_boxes_detached(self, entities):
        """
        For logic constraint. We only want to optimize relation parameters, so detach entity parameters
        """
        min_rep = self.min_embedding(entities).detach()
        delta_rep = self.delta_embedding(entities).detach()
        max_rep = min_rep + torch.exp(delta_rep)
        boxes = Box(min_rep, max_rep)
        return boxes

    def head_transformation(self, head_boxes, rel_ids):
        translations = self.rel_trans_for_head(rel_ids)
        scales = F.relu(self.rel_scale_for_head(rel_ids))
        # affine transformation
        head_boxes.min_embed += translations
        head_boxes.delta_embed *= scales
        head_boxes.max_embed = head_boxes.min_embed + head_boxes.delta_embed

        return head_boxes

    def tail_transformation(self, tail_boxes, rel_ids):
        translations = self.rel_trans_for_tail(rel_ids)
        scales = F.relu(self.rel_scale_for_tail(rel_ids))
        # affine transformation
        tail_boxes.min_embed += translations
        tail_boxes.delta_embed *= scales
        tail_boxes.max_embed = tail_boxes.min_embed + tail_boxes.delta_embed

        return tail_boxes

    def get_all_entity_boxes(self):
        """
        Retrieves entity boxes based on entity indices.
        """
        min_rep = self.min_embedding.weight.data
        delta_rep = self.delta_embedding.weight.data
        max_rep = min_rep + torch.exp(delta_rep)
        boxes = Box(min_rep, max_rep)
        return boxes

    def transform_all_tail_boxes(self, rel_ids):
        """
        Transforms tail entity boxes based on the relation-specific parameters.

        Returns:
            tail_boxes: Transformed tail entity boxes after applying relation-specific affine transformations.
        """
        tail_boxes = self.get_all_entity_boxes()
        translations = self.rel_trans_for_tail(rel_ids)
        scales = F.relu(self.rel_scale_for_tail(rel_ids))

        # affine transformation
        tail_boxes.min_embed += translations
        tail_boxes.delta_embed *= scales
        tail_boxes.max_embed = tail_boxes.min_embed + tail_boxes.delta_embed

        return tail_boxes

    def transform_all_head_boxes(self, rel_ids):
        """
        Transforms head entity boxes based on the relation-specific parameters.
.
        """
        head_boxes = self.get_all_entity_boxes()
        translations = self.rel_trans_for_head(rel_ids)
        scales = F.relu(self.rel_scale_for_head(rel_ids))

        # affine transformation
        head_boxes.min_embed += translations
        head_boxes.delta_embed *= scales
        head_boxes.max_embed = head_boxes.min_embed + head_boxes.delta_embed

        return head_boxes

    def get_tail_score(self, head_id, relation_id):
        head_boxes = self.transform_head_boxes(head_id.repeat(self.num_ent), relation_id)
        tail_boxes = self.transform_all_tail_boxes(relation_id)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on subject or object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        pos_predictions = torch.exp(log_prob)
        return pos_predictions

    def get_head_score(self, tail_id, relation_id):
        head_boxes = self.transform_all_head_boxes(relation_id)
        tail_boxes = self.transform_tail_boxes(tail_id.repeat(self.num_ent), relation_id)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on subject or object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        predictions = torch.exp(log_prob)
        return predictions

    def get_hrt_score(self, head_id, relation_id, tail_id):
        head_boxes = self.transform_head_boxes(head_id.repeat(len(tail_id)), relation_id)
        tail_boxes = self.transform_tail_boxes(tail_id, relation_id)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on subject or object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        predictions = torch.exp(log_prob)
        return predictions

    def get_trh_score(self, head_id, relation_id, tail_id):
        head_boxes = self.transform_head_boxes(head_id, relation_id)
        tail_boxes = self.transform_tail_boxes(tail_id.repeat(len(head_id)), relation_id)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on subject or object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        predictions = torch.exp(log_prob)
        return predictions


class Box:
    """
    A class representing an n-dimensional axis-aligned hyper rectangle (box) in the embedding space.

    Attributes:
        min_embed: The minimum boundary vector of the box in the embedding space.
        max_embed: The maximum boundary vector of the box in the embedding space.
        delta_embed: The difference between the maximum and minimum boundaries,
        representing the size of the box in each dimension.
    """

    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed
