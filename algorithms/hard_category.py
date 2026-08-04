"""DEMix-style hard category routing — each branch dedicated to one category.

mask[b, k] = 1 iff k == category_ids[b], else 0. Equivalent to a hard top-1
select-by-category. Fixed-denominator merge with `expected_mask_sum=1`
makes output = h_{category_ids[b]}.

This is the metadata-aware baseline: "what does the
strongest possible category-routed baseline look like, given the same
ground-truth metadata SpecDrop receives?". Requires K = M (one branch per
category).
"""
import torch
from .base import ModularDropout


class HardCategory(ModularDropout):
    """One-hot mask: branch k_i selected for sample with category_ids[i]=k_i.

    Trains end-to-end with the same MultiBranch + fixed-denom merge pipeline
    as ours; only the mask is one-hot per sample instead of soft. Equivalent
    to dedicating each branch to one category.
    """

    def __init__(self, num_modules: int, num_categories: int):
        super().__init__(num_modules=num_modules, num_categories=num_categories)
        if num_modules != num_categories:
            raise ValueError(
                f'HardCategory requires K==M (each branch dedicated to one '
                f'category), got num_modules={num_modules}, '
                f'num_categories={num_categories}')

    @property
    def expected_mask_sum(self):
        # Exactly one branch active per sample → S = 1.
        return 1

    def get_mask(self, category_ids, training=True, **kwargs):
        B = category_ids.size(0)
        K = self.num_modules
        # one-hot: mask[b, k] = 1 iff k == category_ids[b]
        return torch.nn.functional.one_hot(
            category_ids.long(), num_classes=K
        ).to(dtype=torch.float32, device=category_ids.device)
