"""CPU regression tests for the FTP-1-style tactile/action interface."""
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from g05.models.g05.g05_model_qwen35 import G05ModelQwen35
from g05.models.g05.helpers.fm_helper import FMHelper
from g05.models.g05.helpers.mask_helper import MaskHelperQwen35
from g05.models.g05.qwen35.mixture_qwen35 import MixtureQwen35
from g05.models.g05.tactile import TactileBranch
from g05.models.kv_cache import SparseKVCache
from g05.utils.data.processor_utils import build_processors


def task_config():
    with initialize_config_dir(str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"):
        return compose("train", ["task=combined_rj45_tactile_shift3"])


def tiny_model(checkpoint=True, joint=False, samples=1):
    cfg = OmegaConf.create(OmegaConf.to_container(task_config().model.model_arch, resolve=True))
    ae = cfg.action_expert
    ae.hidden_size = ae.time_hidden_size = 32
    ae.intermediate_size = 64
    ae.num_hidden_layers = 2
    ae.layer_types = ["full_attention"] * 2
    ae.num_attention_heads = 2
    ae.num_key_value_heads = 1
    ae.head_dim = 16
    ae.rope_parameters.mrope_section = [1, 1, 0]
    ae.input_dim = ae.output_dim = 3
    tc = cfg.tactile
    tc.image_size = [16, 16]
    tc.patch_size = 8
    tc.encoder_dim = tc.hidden_size = 32
    tc.encoder_heads = 2
    tc.stem_depth = tc.trunk_depth = 1
    tc.intermediate_size = 64
    tc.gradient_checkpointing = checkpoint
    cfg.fm.joint_training = joint
    cfg.fm.num_flow_samples = samples
    cfg.fm.horizon_steps = 4
    cfg.fm.action_dim = 3
    cfg.fm.num_inference_steps = 2
    model = G05ModelQwen35.__new__(G05ModelQwen35)
    torch.nn.Module.__init__(model)
    model.cfg = cfg
    model.action_expert = MixtureQwen35(ae)
    # Production loads a trained action head; a zero-init head hides touch grads.
    torch.nn.init.normal_(model.action_expert.output_proj.weight, std=0.1)
    for layer in model.action_expert.layers:
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            with torch.no_grad():
                norm.dense.bias[2 * ae.hidden_size:].fill_(0.5)
    model.tactile_branch = TactileBranch(tc, ae)
    model.mask_helper = MaskHelperQwen35(cfg)
    model.fm_helper = FMHelper(cfg.fm)
    model.attn_implementation = "eager"
    return model


def inputs():
    cache = SparseKVCache()
    # Mimic Qwen's sparse visual KV: layer 0 is absent, layer 1 has visual KV.
    for dest in (cache.key_cache, cache.value_cache):
        dest[1] = torch.randn(2, 1, 5, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 4, 4, 0], [1, 1, 4, 4, 4]])
    positions = torch.arange(5).expand(3, 2, -1)
    touch = {k: torch.randn(2, 1, 3, 16, 16) for k in ("tactile_left_1", "tactile_left_2")}
    return cache, mask, positions, touch


@pytest.mark.parametrize("checkpoint,joint,samples", [(False, False, 1), (True, False, 2), (True, True, 1)])
def test_fm_gradient_reaches_all_trainable_tactile_parameters(checkpoint, joint, samples):
    torch.manual_seed(7)
    model = tiny_model(checkpoint, joint, samples)
    cache, mask, positions, touch = inputs()
    loss = model.fm_helper.train_step(
        model, cache, mask, positions, torch.randn(2, 4, 3),
        torch.zeros(2, 4, dtype=torch.bool), None, torch.float32,
        tactile_pixel_values=touch,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.tactile_branch.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
    assert model.tactile_branch.encoder.stems["deeptouch"]["patch_embed"].proj.weight.grad.abs().sum() > 0
    assert (cache.value_cache[1].grad is not None) == joint
    assert set(cache.key_cache) == {1}
    assert cache.key_cache[1].shape[-2] == 5


def test_sparse_cache_positions_order_and_no_inplace_mutation():
    model = tiny_model().eval()
    cache, mask, positions, touch = inputs()
    merged, am, pos = model.prepare_action_context(cache, mask, positions, touch)
    assert merged.key_cache[0].shape == (2, 1, 2, 16)
    assert merged.key_cache[1].shape == (2, 1, 7, 16)
    assert am.shape == (2, 7) and pos.shape == (3, 2, 7)
    torch.testing.assert_close(pos[0, :, -2:], torch.tensor([[4, 5], [5, 6]]))
    torch.testing.assert_close(merged.key_cache[1][..., :5, :], cache.key_cache[1])
    assert set(cache.key_cache) == {1}
    assert mask.shape[-1] == positions.shape[-1] == 5
    encoder = model.tactile_branch.encoder
    torch.testing.assert_close(encoder(touch), encoder(dict(reversed(list(touch.items())))))
    with pytest.raises(ValueError, match="Expected tactile cameras"):
        encoder({"tactile_left_1": touch["tactile_left_1"]})


def test_inference_encodes_touch_once_and_responds_to_touch():
    model = tiny_model().eval()
    cache, mask, positions, touch = inputs()
    calls = []
    handle = model.tactile_branch.register_forward_hook(lambda *args: calls.append(1))
    def run(images):
        torch.manual_seed(17)
        with torch.no_grad():
            return model.fm_helper.infer(
                model, mask, {"exterior": torch.zeros(2, 1, 3, 16, 16)}, cache,
                position_ids_override=positions, tactile_pixel_values=images,
            )
    first = run(touch)
    assert len(calls) == 1
    second = run({k: -v for k, v in touch.items()})
    handle.remove()
    assert first.shape == (2, 4, 3) and torch.isfinite(first).all()
    assert not torch.allclose(first, second, atol=1e-6)
    assert cache.key_cache[1].shape[-2] == 5


def test_disabled_branch_is_identity_and_rejects_unexpected_touch():
    model = tiny_model()
    model.tactile_branch = None
    cache, mask, positions, touch = inputs()
    result = model.prepare_action_context(cache, mask, positions)
    assert all(a is b for a, b in zip(result, (cache, mask, positions)))
    with pytest.raises(ValueError, match="without a tactile branch"):
        model.prepare_action_context(cache, mask, positions, touch)


def test_config_keeps_touch_out_of_visual_slots_and_uses_no_extra_shift():
    cfg = task_config()
    p = build_processors(cfg)["galaxea_r1pro"]
    p.eval()
    assert p.samples_builder._image_keys == ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
    metas = cfg.data.embodiment_datasets.galaxea_r1pro.shape_meta
    assert all(m.time_offset == 0 for group in metas.values() for m in group)
    data = {"images": {m.key: torch.zeros(1, *m.raw_shape, dtype=torch.uint8) for m in metas.images}}
    visual = p.process_images(data)
    assert set(visual) == {"exterior", "wrist_left", "wrist_right"}
    # Parent modality filter uses the same normalization/resize pipeline at eval.
    tactile = super(type(p), p).process_images(data, modality="tactile")
    assert set(tactile) == set(cfg.model.model_arch.tactile.camera_keys)
    assert all(v.shape == (1, 3, 224, 224) for v in tactile.values())
    assert all(torch.all(v == -1) for v in tactile.values())


def test_tactile_checkpoint_roundtrip():
    model = tiny_model()
    restored = tiny_model()
    restored.load_state_dict(model.state_dict(), strict=True)
    _, mask, pos, touch = inputs()
    model.eval()
    restored.eval()
    for a, b in zip(model.tactile_branch(touch, mask, pos)[0].value_cache.values(),
                    restored.tactile_branch(touch, mask, pos)[0].value_cache.values()):
        torch.testing.assert_close(a, b)


def test_dataset_config_resolves_processor_references_before_stripping(monkeypatch):
    from g05.utils.data import processor_utils
    monkeypatch.setattr(processor_utils, "instantiate", lambda cfg, **kw: cfg)
    data_cfg = processor_utils.instantiate_dataset(task_config())
    assert "processors" not in data_cfg
    assert len(data_cfg.embodiment_datasets.galaxea_r1pro.shape_meta.images) == 5
