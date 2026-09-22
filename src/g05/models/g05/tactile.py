"""FTP-1-style image tactile encoding and an independent Qwen3.5 KV expert.

Architecture reference: michaelyuancb/ftp1-policy, ftp1_blocks.py and
ftp1_attention_masks.py. This is a G05 adaptation, not a compatible FTP-1
checkpoint loader. Image encoders and the tactile expert start from scratch.
"""

from copy import deepcopy

import torch
from torch import nn
from timm.models.vision_transformer import Block, PatchEmbed

from g05.models.kv_cache import SparseKVCache
from .qwen35.mixture_qwen35 import MixtureQwen35


class TactileImageEncoder(nn.Module):
    """Sensor-specific ViT stem → shared ViT trunk → CLS/area token.

    Two cameras of the same sensor type share a stem. Function-area IDs identify
    the two physical contact sites; their order is explicit, never dict order.
    Input: camera dict of normalized [-1, 1] RGB [B, T, 3, H, W].
    """

    def __init__(self, cfg):
        super().__init__()
        self.camera_keys = list(cfg.camera_keys)
        self.sensor_ids = list(cfg.sensor_ids)
        self.area_ids = list(cfg.area_ids)
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("tactile.camera_keys must be nonempty and unique")
        if not len(self.camera_keys) == len(self.sensor_ids) == len(self.area_ids):
            raise ValueError("camera_keys, sensor_ids and area_ids must have equal lengths")
        if len(set(self.area_ids)) != len(self.area_ids):
            raise ValueError("Each tactile camera must have a distinct function-area ID")
        if any(i < 0 or i >= cfg.num_areas for i in self.area_ids):
            raise ValueError("Tactile area ID outside num_areas")
        self.image_size = tuple(cfg.image_size)
        dim = int(cfg.encoder_dim)
        patch = int(cfg.patch_size)
        if any(s % patch for s in self.image_size):
            raise ValueError("Tactile image dimensions must be divisible by patch_size")
        self.stems = nn.ModuleDict()
        for sensor in sorted(set(self.sensor_ids)):
            if not sensor or "." in sensor:
                raise ValueError("Sensor IDs must be nonempty module-safe strings")
            self.stems[sensor] = nn.ModuleDict({
                "patch_embed": PatchEmbed(self.image_size, patch, 3, dim),
                "blocks": nn.Sequential(*[
                    Block(dim, int(cfg.encoder_heads), mlp_ratio=4, qkv_bias=True)
                    for _ in range(int(cfg.stem_depth))
                ]),
            })
        patches = (self.image_size[0] // patch) * (self.image_size[1] // patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, patches + 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.trunk = nn.Sequential(*[
            Block(dim, int(cfg.encoder_heads), mlp_ratio=4, qkv_bias=True)
            for _ in range(int(cfg.trunk_depth))
        ], nn.LayerNorm(dim))
        self.image_proj = nn.Linear(dim, int(cfg.hidden_size))
        self.area_embedding = nn.Embedding(int(cfg.num_areas), int(cfg.hidden_size))
        self.unified_proj = nn.Linear(int(cfg.hidden_size), int(cfg.hidden_size))

    def forward(self, images):
        if images is None or set(images) != set(self.camera_keys):
            raise ValueError(f"Expected tactile cameras {self.camera_keys}; got {None if images is None else list(images)}")
        first = images[self.camera_keys[0]]
        if first.ndim != 5:
            raise ValueError("Tactile input must be [B,T,3,H,W]")
        b, t = first.shape[:2]
        tokens = []
        for camera, sensor, area in zip(self.camera_keys, self.sensor_ids, self.area_ids):
            x = images[camera]
            if tuple(x.shape) != (b, t, 3, *self.image_size):
                raise ValueError(f"Invalid tactile shape for {camera}: {tuple(x.shape)}")
            stem = self.stems[sensor]
            x = x.reshape(b * t, 3, *self.image_size)
            x = stem["patch_embed"](x.to(stem["patch_embed"].proj.weight.dtype))
            x = torch.cat([self.cls_token.expand(b * t, -1, -1).to(x.dtype), x], dim=1)
            x = stem["blocks"](x + self.pos_embed.to(x.dtype))
            x = self.image_proj(self.trunk(x)[:, 0])
            tokens.append(x.reshape(b, t, -1) + self.area_embedding.weight[area])
        # Time-major, then function-area order, as in FTP-1.
        return self.unified_proj(torch.stack(tokens, dim=2).flatten(1, 2))


class TactileBranch(nn.Module):
    """Independent tactile self-attention; action attends [vision | touch | action]."""

    def __init__(self, cfg, action_cfg):
        super().__init__()
        self.encoder = TactileImageEncoder(cfg)
        expert_cfg = deepcopy(action_cfg)
        expert_cfg.hidden_size = int(cfg.hidden_size)
        expert_cfg.intermediate_size = int(cfg.intermediate_size)
        expert_cfg.adaptive_mode = None
        expert_cfg.time_hidden_size = 0
        expert_cfg.input_dim = int(cfg.hidden_size)
        expert_cfg.output_dim = int(cfg.hidden_size)
        expert_cfg.use_final_norm = False
        expert_cfg.layer_types = ["full_attention"] * int(action_cfg.num_hidden_layers)
        self.expert = MixtureQwen35(expert_cfg)
        self.expert.input_proj = nn.Identity()
        self.expert.output_proj = nn.Identity()
        # Only per-layer K/V are consumed, not final hidden states. The final
        # layer's Q/O/MLP cannot affect any consumed tensor; exclude them from DDP.
        last = self.expert.layers[-1]
        last.requires_grad_(False)
        for module in (last.input_layernorm, last.self_attn.k_proj,
                       last.self_attn.v_proj, last.self_attn.k_norm):
            module.requires_grad_(True)
        if cfg.get("gradient_checkpointing", False):
            self.expert.gradient_checkpointing_enable()

    def forward(self, images, prefix_mask, prefix_positions):
        tokens = self.encoder(images)
        b, n = tokens.shape[:2]
        if b != prefix_mask.shape[0]:
            raise ValueError("Tactile and visual batch sizes differ")
        # Qwen MRoPE: use the next free position on every axis.
        pos = prefix_positions
        if pos.ndim == 2:
            pos = pos.unsqueeze(0).expand(3, -1, -1)
        valid = prefix_mask > 0
        offset = pos.masked_fill(~valid.unsqueeze(0), -1).amax(dim=(0, 2)) + 1
        tactile_pos = offset[:, None] + torch.arange(n, device=tokens.device)[None, :]
        tactile_pos = tactile_pos.unsqueeze(0).expand(3, -1, -1)
        mask = torch.zeros(b, 1, n, n, device=tokens.device, dtype=tokens.dtype)
        cache = SparseKVCache()
        _, cache = self.expert(
            inputs_embeds=tokens, attention_mask=mask, position_ids=tactile_pos,
            kv_cache=cache, return_kv_cache=True, attn_implementation="eager",
            mixture_name="tactile",
        )
        return cache, tactile_pos


def append_tactile_context(visual_cache, tactile_cache, prefix_mask, prefix_positions, tactile_positions):
    """No in-place cache edits: inference requests cannot contaminate one another.

    Qwen VLM linear-attention layers have no visual KV. Those action layers get
    touch-only KV; MixtureQwen35 slices the corresponding trailing mask region.
    """
    combined = SparseKVCache(last_linear_layer=visual_cache.last_linear_layer)
    combined.recurrent_states.update(visual_cache.recurrent_states)
    combined.split_recurrent_states.update(visual_cache.split_recurrent_states)
    for layer, tk in tactile_cache.key_cache.items():
        tv = tactile_cache.value_cache[layer]
        if visual_cache.has_item(layer):
            vk, vv = visual_cache.get(layer)
            tk, tv = torch.cat([vk, tk.to(vk.dtype)], dim=-2), torch.cat([vv, tv.to(vv.dtype)], dim=-2)
        combined.key_cache[layer], combined.value_cache[layer] = tk, tv
    n = tactile_positions.shape[-1]
    # IMAGE (=1) tokens are readable by FM masks, never part of VLM tokenization.
    mask = torch.cat([prefix_mask, prefix_mask.new_ones(prefix_mask.shape[0], n)], dim=-1)
    if prefix_positions.ndim == 2:
        prefix_positions = prefix_positions.unsqueeze(0).expand(3, -1, -1)
    positions = torch.cat([prefix_positions, tactile_positions], dim=-1)
    return combined, mask, positions
