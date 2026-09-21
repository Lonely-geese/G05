#!/usr/bin/env python3
"""Export a completed G0.5 checkpoint with portable inference sidecars."""
import argparse
import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
from g05.utils.checkpoint.ckpt_utils import (
    copy_hf_processor_files, find_run_dir, load_config_from_run_dir,
)
from g05.utils.config.config_resolvers import register_default_resolvers


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--resume', action='store_true', help='Reuse a partial export after re-verifying all weights')
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    dest = args.output.resolve()
    staging = dest.with_name(dest.name + '.partial')
    archive = dest.with_name(dest.name + '.tar.gz')
    archive_partial = archive.with_name(archive.name + '.partial')
    for p in (dest, archive, archive_partial):
        if p.exists():
            raise FileExistsError(f'Refusing to overwrite {p}')
    if staging.exists() and not args.resume:
        raise FileExistsError(f'Partial export exists: {staging}; use --resume to verify and continue')
    if not shutil.which('pigz'):
        raise RuntimeError('pigz is required for compression')
    torch.set_num_threads(args.threads)
    register_default_resolvers()
    run = find_run_dir(str(source))
    before = source.stat()
    print(f'Loading final checkpoint: {source}', flush=True)
    checkpoint = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint.get('step') != args.expected_step:
        raise ValueError('Checkpoint step does not match expected final step')
    if checkpoint.get('optimizer_state_dict') is not None or checkpoint.get('scheduler_state_dict') is not None:
        raise ValueError('Expected completed inference-only final checkpoint, not the intermediate periodic save')
    weights = checkpoint['model_state_dict']
    (staging / 'checkpoints').mkdir(parents=True, exist_ok=args.resume)
    target = staging / 'checkpoints/model_state_dict.pt'
    if not target.exists():
        torch.save({'model_state_dict': weights}, target)
    exported = torch.load(target, map_location='cpu', weights_only=True, mmap=True)['model_state_dict']
    if exported.keys() != weights.keys():
        raise ValueError('Exported weight keys differ')
    for name, tensor in weights.items():
        other = exported[name]
        if tensor.dtype != other.dtype or not torch.equal(tensor, other):
            raise ValueError(f'Weight changed during export: {name}')
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite weights: {name}')
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source checkpoint changed while being exported')
    metadata = {
        'source_checkpoint': str(source), 'source_step': checkpoint['step'],
        'source_epoch': checkpoint['epoch'], 'original_bytes': before.st_size,
        'weights_bytes': target.stat().st_size, 'model_entries': len(weights),
        'exact_weights_verified': True, 'finite_weights_verified': True,
        'dtype_conversion': False, 'removed_keys': [k for k in checkpoint if k != 'model_state_dict'],
        'exported_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    del weights, exported, checkpoint, tensor, other
    gc.collect()
    print(f"All {metadata['model_entries']} model entries verified without dtype conversion", flush=True)

    # Materialize the training-time ${tokenizer} alias before the shared loader
    # patches nested tokenizer checkpoint paths. OmegaConf.update otherwise
    # replaces the alias node with a partial mapping and loses its _target_.
    raw_cfg = OmegaConf.load(run / '.hydra/config.yaml')
    if OmegaConf.is_interpolation(raw_cfg.model, 'tokenizer'):
        raw_cfg.model.tokenizer = OmegaConf.create(OmegaConf.to_container(raw_cfg.tokenizer, resolve=False))
    (staging / '.hydra').mkdir(exist_ok=True)
    OmegaConf.save(raw_cfg, staging / '.hydra/config.yaml', resolve=False)
    cfg = load_config_from_run_dir(staging, str(target), [])
    copy_hf_processor_files(cfg.model.model_arch.hf_processor_path, staging / 'hf_processor')
    for name in ['action_tokenizer.pt', 'dataset_stats.json']:
        shutil.copy2(run / name, staging / name)
    cfg.model.pretrained_ckpt = None
    cfg.model.model_arch.pretrained_model_path = None
    cfg.model.model_arch.hf_processor_path = '${run_dir}/hf_processor'
    cfg.model.processor.tokenizer_params.pretrained_model_name_or_path = '${run_dir}/hf_processor'
    for key in ['tokenizer.vq_config.ckpt_dir', 'model.tokenizer.vq_config.ckpt_dir', 'model.model_arch.AT_CONFIG.ckpt_dir']:
        if OmegaConf.select(cfg, key) is not None:
            OmegaConf.update(cfg, key, '${run_dir}/action_tokenizer.pt', merge=False)
    cfg.ckpt_path = '${run_dir}/checkpoints/model_state_dict.pt'
    cfg.datastatics_path = '${run_dir}/dataset_stats.json'
    cfg.run_dir = '.'
    cfg.output_dir = '${run_dir}/inference_output'
    cfg.resume_ckpt = None
    OmegaConf.save(cfg, staging / '.hydra/config.yaml', resolve=False)

    print('Checking portable CPU model and processor setup', flush=True)
    from serve_policy import setup
    check_cfg = load_config_from_run_dir(staging, str(target), ['model.use_torch_compile=false'])
    policy, processor = setup(check_cfg, device='cpu')
    metadata['cpu_inference_setup_verified'] = True
    metadata['gpu_action_inference_tested'] = False
    del policy, processor
    gc.collect()
    (staging / 'export_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (staging / 'README.txt').write_text(
        'G0.5 人工纠错 SFT 最终推理模型包（2,000 步）。\n'
        '初始化模型：0912 morning；旧示范/纠错采样 70/30。\n'
        f'来源：{source}\n'
        '保留原始权重精度；包含动作 tokenizer、归一化统计、HF processor 和展开后的配置。\n'
        '需要与训练兼容的 GalaxeaVLA 代码及推理环境。\n'
        '在 GalaxeaVLA 仓库根目录运行：\n'
        f'python scripts/serve_policy.py --ckpt_path /path/to/{dest.name}/checkpoints/model_state_dict.pt eval_embodiment=galaxea_r1pro\n'
        '全部模型权重与最终 checkpoint 逐元素一致，未发现非有限值；CPU 模型/处理器加载通过。\n'
        '本次打包未运行 GPU 动作推理或上机成功率测试。\n'
        '在解压后的包目录内执行 sha256sum -c SHA256SUMS 可核验文件。\n'
    )
    hashes = {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
    (staging / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in hashes.items()))
    hashes['SHA256SUMS'] = sha256(staging / 'SHA256SUMS')
    staging.rename(dest)
    print(f'Compressing {dest} with pigz', flush=True)
    with archive_partial.open('wb') as out:
        tar = subprocess.Popen(['tar', '-C', str(dest.parent), '-cf', '-', dest.name], stdout=subprocess.PIPE)
        gzip = subprocess.Popen(['pigz', '-6', '-p', str(args.threads)], stdin=tar.stdout, stdout=out)
        tar.stdout.close()
        gzip_status = gzip.wait()
        tar_status = tar.wait()
        if gzip_status or tar_status:
            raise RuntimeError(f'Compression failed: tar={tar_status}, pigz={gzip_status}')
    print('Verifying every archived file against SHA256SUMS', flush=True)
    decoder = subprocess.Popen(['pigz', '-dc', '-p', str(args.threads), str(archive_partial)], stdout=subprocess.PIPE)
    seen = set()
    with tarfile.open(fileobj=decoder.stdout, mode='r|') as tar:
        for member in tar:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f'Unexpected archive entry: {member.name}')
            relative = str(Path(member.name).relative_to(dest.name))
            if relative in seen or relative not in hashes:
                raise ValueError(f'Unexpected or duplicate file: {relative}')
            seen.add(relative)
            digest = hashlib.sha256()
            f = tar.extractfile(member)
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
            if digest.hexdigest() != hashes[relative]:
                raise ValueError(f'Archive payload checksum mismatch: {relative}')
    # Drain gzip trailer to let the decompressor finish its CRC checks.
    for _ in iter(lambda: decoder.stdout.read(8 * 1024 * 1024), b''):
        pass
    decoder.stdout.close()
    if decoder.wait() or seen != hashes.keys():
        raise RuntimeError('Archive integrity verification failed')
    archive_partial.rename(archive)
    archive_hash = sha256(archive)
    archive.with_name(archive.name + '.sha256').write_text(f'{archive_hash}  {archive.name}\n')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': archive_hash, 'verified_files': len(seen)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
