import random
import numpy as np
import torch
from itertools import chain
from torch.utils.data import Sampler
from torch.distributions.categorical import Categorical

from data_generate.dataset import get_df_inds_per_col_value

class SuppQueryBatchSampler(Sampler):
    def __init__(self, dataset, subset_indices_per_cls=None, seqtask=True, num_batch=10, num_task=100,
                 task_by_supercls=True, num_way=5, num_shot=1, num_query_per_cls=15):
                 
        self.dataset = dataset
        self.subset_indices_per_cls = subset_indices_per_cls
        self.seqtask = seqtask
        self.num_batch = num_batch
        self.num_task = num_task
        self.task_by_supercls = task_by_supercls
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query_per_cls = num_query_per_cls
        self.num_sample_per_cls = self.num_shot + self.num_query_per_cls

    def __len__(self):
        return self.num_batch if self.seqtask else self.num_task

    def __iter__(self):
        if self.seqtask:
            if self.subset_indices_per_cls is None:
                inds_per_cls = get_df_inds_per_col_value(self.dataset.df, 'cls_lbl', shuffle=True)
            else:
                inds_per_cls = self.subset_indices_per_cls

            supp_inds_per_cls_batched = []
            query_inds_per_cls_batched = []
            for cls, inds in inds_per_cls:

                random.shuffle(inds)
                supp_inds_per_cls_batched.append(
                    (cls, np.array_split(inds[:(self.num_shot * self.num_batch)], self.num_batch))
                )

                query_inds_per_cls_batched.append(
                    (cls, np.array_split(inds[-(self.num_query_per_cls * self.num_batch):], self.num_batch))
                )

            supp_batched_unchain = list(zip(*list(zip(*supp_inds_per_cls_batched))[1]))
            query_batched_unchain = list(zip(*list(zip(*query_inds_per_cls_batched))[1]))
            for supp_batch, query_batch in zip(supp_batched_unchain, query_batched_unchain):
                supp_chained = list(chain.from_iterable(supp_batch))
                query_chained = list(chain.from_iterable(query_batch))

                random.shuffle(supp_chained)
                random.shuffle(query_chained)

                yield supp_chained + query_chained
        else:
            for _ in range(self.num_task):
                if self.task_by_supercls:
                    sampled_supercls = random.choice(self.dataset.df['supercls'].unique())
                    sampled_task = random.sample(
                        list(self.dataset.df.loc[self.dataset.df['supercls'] == sampled_supercls, 'cls_name'].unique()),
                        k=self.num_way
                    )
                else:
                    sampled_task = random.sample(list(self.dataset.df['cls_name'].unique()), k=self.num_way)
                # relabel according to sampled task
                self.dataset.relabel = ('cls_name', sampled_task)
                self.dataset.relbl_df()
                # get task df
                inds_per_cls = get_df_inds_per_col_value(
                    self.dataset.df.loc[self.dataset.df['cls_name'].isin(sampled_task)],
                    col='cls_name'
                )
                supp_inds = []
                query_inds = []
                for clsname, inds in inds_per_cls:
                    sampled_inds = random.sample(inds, self.num_sample_per_cls)

                    supp_inds.extend(sampled_inds[:self.num_shot])
                    query_inds.extend(sampled_inds[self.num_shot:])
                random.shuffle(supp_inds)
                random.shuffle(query_inds)
                yield supp_inds + query_inds