"""One-time bridge from a frozen teacher to framework-neutral Safetensors shards.

Adapter-driven (see ADAPTER.md), so it serves both the LongLive teacher (Plan A,
CUDA) and the Wan2.1 teacher (Plan B, CPU). It implements the three cache-format
changes from TEACHER-OPTIONS.md that turn a handful of clips into a trainable
dataset without a GPU:

  * `--draws-per-clip` — multiple (noise, timestep) draws per clip. Effective
    supervision is shard count, not step count (video-train re-shows a shard's
    frozen (noise, t) every time its cursor wraps), so this is what scales a few
    hundred prompts into thousands of samples, at the cost of teacher forward
    passes rather than video data.
  * `--relation-layers` — cache a small linspaced subset of layers, not all ~30.
    The grams dominate shard size; the loss only ever consumes a few.
  * `--device cpu` — the teacher pass is inference-only, so a CPU run is viable
    (Plan B's whole premise).

Each shard carries the `video-contract` tensors — `noisy_latents`, `timestep`
(int64), `prompt_embeds`, `teacher_noise_pred`, and `teacher_relation.{0..k-1}` —
plus `student_prompt_embeds` (umt5-small) when the dataset item provides it, so the
Burn student conditions on a browser-runnable encoder. Relation grams are stored at
the teacher's post-patchify token count, which matches the patchified student.
"""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import torch
from safetensors.torch import save_file


def resolve(spec):
    module, name = spec.split(':', 1)
    return getattr(importlib.import_module(module), name)


def select_layers(n: int, cap: int) -> list[int]:
    """`cap` linspaced layer indices (matching torch.linspace(...).round()), or
    every layer when cap<=0 or exceeds what the teacher exposes."""
    if cap <= 0 or cap >= n:
        return list(range(n))
    return torch.linspace(0, n - 1, cap).round().long().tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', required=True, help='module:function returning the frozen teacher')
    p.add_argument('--dataset', required=True, help='dir of .pt items (latents, prompt_embeds[, student_prompt_embeds])')
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--scheduler', default='wan21')
    p.add_argument('--teacher-name', default='Wan2.1-T2V-1.3B')
    p.add_argument('--device', default='cpu')
    p.add_argument('--draws-per-clip', type=int, default=1)
    p.add_argument('--relation-layers', type=int, default=0, help='cap on cached relation layers; 0 = every layer')
    p.add_argument('--adapter-arg', action='append', default=[], help='key=value forwarded to the adapter (value JSON-parsed)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    cfg: dict = {}
    for kv in a.adapter_arg:
        key, value = kv.split('=', 1)
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    teacher = resolve(a.adapter)(cfg).to(a.device).eval()

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.dataset).glob('*.pt'))[:a.limit or None]
    if not files:
        raise SystemExit(f'no .pt items in {a.dataset}')

    shards: list[str] = []
    shapes: dict[str, dict] = {}
    num_layers = 0
    k = 0
    with torch.no_grad():
        for i, file in enumerate(files):
            item = torch.load(file, map_location='cpu', weights_only=True)
            lat = item['latents'].float()
            prompt = item['prompt_embeds'].float()
            student_prompt = item.get('student_prompt_embeds')
            for draw in range(max(1, a.draws_per_clip)):
                g = torch.Generator().manual_seed(a.seed + i * 9973 + draw)
                noise = torch.randn(lat.shape, generator=g)
                t = torch.randint(0, 1000, (lat.shape[0],), generator=g)
                sigma = (t.float() / 1000).view(-1, *([1] * (lat.dim() - 1)))
                noisy = lat + noise * sigma
                result = teacher(noisy.to(a.device), t.to(a.device), prompt.to(a.device))
                tensors = {
                    'noisy_latents': noisy.cpu().contiguous(),
                    'timestep': t.to(torch.int64).cpu().contiguous(),
                    'prompt_embeds': prompt.cpu().contiguous(),
                    'teacher_noise_pred': result['noise_pred'].float().cpu().contiguous(),
                }
                if student_prompt is not None:
                    tensors['student_prompt_embeds'] = student_prompt.float().cpu().contiguous()
                hidden = result.get('hidden_states', [])
                selected = select_layers(len(hidden), a.relation_layers)
                for out_idx, layer in enumerate(selected):
                    h = torch.nn.functional.normalize(hidden[layer].float(), dim=-1)
                    tensors[f'teacher_relation.{out_idx}'] = (h @ h.transpose(-1, -2)).half().cpu().contiguous()
                num_layers = len(selected)
                name = f'shard-{k:06d}.safetensors'
                save_file(tensors, out / name)
                shards.append(name)
                k += 1
                if not shapes:
                    for key, value in tensors.items():
                        shapes[key] = {'name': key, 'shape': list(value.shape), 'dtype': str(value.dtype).replace('torch.', '').upper()}

    manifest = {
        'format_version': 1,
        'teacher': a.teacher_name,
        'scheduler': a.scheduler,
        'shards': shards,
        'tensors': list(shapes.values()),
        'hidden_relation_layers': list(range(num_layers)),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'wrote {len(shards)} shards from {len(files)} clips ({num_layers} relation layers) to {out}')


if __name__ == '__main__':
    main()
