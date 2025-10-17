import math
from multiprocessing import Value
import torch
import json

file = open("././parameters.json")
parameters: dict[str, int] = json.load(file)["ijepa"]

class Mask(object):
    
    def __init__(
        self,
        input_size=(parameters["IMAGE_SIZE"], parameters["IMAGE_SIZE"]),
        patch_size=parameters["PATCH_SIZE"],
        nctx=1,
        ntarg=parameters["NUM_TARGET_BLOCKS"],
        targ_mask_scale=(0.15, 0.15),
        ctx_mask_scale=(0.85, 0.85),
        aspect_ratio=(0.75, 1.5),
        min_keep=4,
        max_tries=50
    ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.patch_size = patch_size
        self.nctx = nctx
        self.ntarg = ntarg
        self.targ_mask_scale = targ_mask_scale
        self.ctx_mask_scale = ctx_mask_scale
        self.aspect_ratio = aspect_ratio
        self.min_keep = min_keep
        self.max_tries = max_tries
        self.iteration_counter = Value('i', -1)
        
    def step(self):
        """
        Increments ``i`` for every image, resulting in different context and target blocks.
        """
        i = self.iteration_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
        
    def _sample_block_size(self, generator, scale, ascpect_ratio_scale):
        u = torch.rand(1, generator=generator).item()
        
        # block size scale
        min_s, max_s = scale 
        
        # ratio of block size / total patch number
        frac = min_s + u * (max_s - min_s)
        
        # number of patches in the block
        num_patches = int(self.height*self.width * frac)
        
        # provide the minimum amount of patches per block
        num_patches = max(num_patches, self.min_keep)
        
        v = torch.rand(1, generator=generator).item()
        min_asp_ratio, max_asp_ratio = ascpect_ratio_scale
        aspect_ratio = min_asp_ratio + v * (max_asp_ratio - min_asp_ratio)
        
        h = int(round(math.sqrt((num_patches * aspect_ratio))))
        w = int(round(math.sqrt((num_patches / aspect_ratio))))
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        h = max(h, 1)
        w = max(w, 1)
        return h, w
    
    def _place_block_without_overlap(self, h, w, occ):
        H, W = occ.shape
        tries = 0
        while tries < self.max_tries:
            top = torch.randint(0, H - h + 1, (1,)).item()
            left = torch.randint(0, W- w + 1, (1,)).item()
            cut_region = occ[top:top+h, left:left+w]
            
            if torch.count_nonzero(cut_region) == 0:
                mask = torch.zeros((H, W), dtype=torch.int32)
                mask[top:top+h, left:left+w] = 1
                occ[top:top+h, left:left+w] = 1
                idx = torch.nonzero(mask.flatten(), as_tuple=False).squeeze()
                
                if idx.numel() >= self.min_keep:
                    return idx, occ
            
            tries += 1
        
        idx = torch.randperm(H*W)[:max(self.min_keep, h*w // 2)]
        
        return idx, occ
    
    def __call__(self, batch):
        B = len(batch)
        if isinstance(batch, torch.Tensor):
            collated_batch = batch
        else: collated_batch = torch.utils.data.default_collate(batch)
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        ctx_h, ctx_w = self._sample_block_size(g, self.ctx_mask_scale, (1., 1.))
        target_h, target_w = self._sample_block_size(g, self.targ_mask_scale, self.aspect_ratio)
        
        all_mask_ctx, all_mask_target = [], []
        
        for _ in range(B):
            occ = torch.zeros((self.height, self.width), dtype=torch.int32)
            
            target_mask = []
            for _ in range(self.ntarg):
                idx, occ = self._place_block_without_overlap(target_h, target_w, occ)
                target_mask.append(idx)
            
            free = (occ == 0).to(torch.int32) # create context from target complement
            
            tries = 0
            cmask = None
            
            while tries < self.max_tries:
                top = torch.randint(0, self.height - ctx_h + 1, (1,)).item()
                left = torch.randint(0, self.width - ctx_w + 1, (1,)).item()
                region = free[top:top+ctx_h, left:left+ctx_w]
                if torch.all(region == 1):
                    cmask2d = torch.zeros((self.height, self.width), dtype=torch.int32)
                    cmask2d[top:top+ctx_h, left:left+ctx_w] = 1
                    cmask = torch.nonzero(cmask2d.flatten(), as_tuple=False).squeeze()
                    break
                tries += 1
            
            if cmask is None:
                ## using fallback
                free_idx = torch.nonzero(free.flatten(), as_tuple=False).view(-1)
                n_free = free_idx.numel()
                if n_free == 0:
                    cmask = torch.randperm(self.height * self.width)[:max(self.min_keep, ctx_h * ctx_w // 2)]
                else:
                    perm = torch.randperm(n_free)
                    take = max(self.min_keep, min(ctx_h *ctx_w, n_free))
                    cmask = free_idx[perm[:take]]
                    
            all_mask_target.append(target_mask)
            all_mask_ctx.append([cmask])
            
        masked_ctx_batch = batch.clone()
        masked_target_batch = batch.clone()
        
        for i in range(B):
            ctx_idx = all_mask_ctx[i][0] + 1
            targ_id_list = all_mask_target[i]
            
            target_idx = torch.cat([idx for idx in targ_id_list]) + 1
            masked_ctx_batch[i, target_idx, :] = 0  # Mask target tokens
            
            masked_target_batch[i, ctx_idx, :] = 0  # Mask context tokens
            
        return collated_batch, masked_ctx_batch, masked_target_batch