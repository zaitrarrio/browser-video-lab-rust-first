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
    every layer when cap==0 or exceeds what the teacher exposes.

    `cap < 0` caches no grams at all. That is not a degenerate case: a gram is
    `tokens²` fp16 and at any real geometry it dominates the shard (10.6 MB per
    layer at 2304 tokens against ~3 MB for everything else combined), so trading
    the relation term away buys roughly an order of magnitude more distinct
    (noise, sigma) draws for the same disk — and draw count, not step count, is
    what actually bounds supervision here (TEACHER-OPTIONS.md)."""
    if cap < 0:
        return []
    if cap == 0 or cap >= n:
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
    p.add_argument('--relation-layers', type=int, default=0,
                   help='cap on cached relation layers; 0 = every layer, negative = none (see select_layers)')
    p.add_argument('--adapter-arg', action='append', default=[], help='key=value forwarded to the adapter (value JSON-parsed)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--noising', choices=['flow', 'legacy'], default='flow',
                   help="flow: x_t=(1-s)x0+s*eps, Wan's actual flow-matching parameterization. "
                        "legacy: the original x_t=x0+s*eps, kept only to reproduce old caches.")
    p.add_argument('--shift', type=float, default=3.0,
                   help="Wan flow_shift; warps the sigma draw towards high noise the way inference does")
    p.add_argument('--guidance', type=float, default=0.0,
                   help='>0 caches the CFG-combined velocity uncond + w*(cond-uncond) instead of the bare '
                        'conditional one. Costs a second teacher pass per shard and is what lets the student '
                        'be sampled without running CFG itself.')
    p.add_argument('--teacher-dtype', choices=['f32', 'bf16', 'f16'], default='f32',
                   help='element type the teacher runs in. bf16 is 2.6x faster than f32 (the single '
                        'largest lever measured — docs/PERF-ROUND-1.md) and is the precision Wan was '
                        'trained in and that make_wan_dataset.py already generates the latents with. It '
                        'is NOT free: cached targets differ from the f32 ones by ~7% mean relative error, '
                        'so validate against parity_baseline.py before building a large cache with it.')
    p.add_argument('--draw-batch', type=int, default=8,
                   help='draws evaluated in one teacher forward. Worth +18%%; the throughput knee is at '
                        '4-8 and batch 16 doubles VRAM for under 1%%, so 8 is the measured setting. '
                        '1 restores the historical one-forward-per-draw behaviour.')
    a = p.parse_args()

    cfg: dict = {}
    for kv in a.adapter_arg:
        key, value = kv.split('=', 1)
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    # `--seed` seeds the global RNG too, not just the per-draw generators. The real
    # teacher loads its weights, so this changes nothing there — but the toy teacher
    # is randomly initialized `nn.Linear` layers, and without this two runs of this
    # script build two *different* teachers. Any test that compares one invocation
    # against another is then comparing weights, not the thing it meant to compare.
    torch.manual_seed(a.seed)

    # The adapter builds in the requested element type rather than being cast after
    # the fact: `from_pretrained(torch_dtype=...)` is what was measured, and casting
    # a built module converts parameters the loader deliberately left in f32.
    TDT = {'f32': torch.float32, 'bf16': torch.bfloat16, 'f16': torch.float16}[a.teacher_dtype]
    if TDT is not torch.float32:
        cfg.setdefault('dtype', TDT)
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
            negative = item.get('negative_prompt_embeds')
            student_prompt = item.get('student_prompt_embeds')
            if a.guidance > 0 and negative is None:
                raise SystemExit(
                    f'--guidance {a.guidance} needs negative_prompt_embeds in {file.name}; '
                    'build the dataset with make_wan_dataset.py, which stores them'
                )
            # Every draw's input is built up front so a batch of them can go through
            # the teacher in one forward. The generator is seeded per (clip, draw),
            # so batching reproduces exactly the values the one-at-a-time loop
            # produced — only the kernel's reduction order differs.
            draws, noisy_all, t_all = max(1, a.draws_per_clip), [], []
            for draw in range(draws):
                g = torch.Generator().manual_seed(a.seed + i * 9973 + draw)
                noise = torch.randn(lat.shape, generator=g)
                if a.noising == 'flow':
                    # Wan is flow-matching: x_s = (1-s)x0 + s*eps, and the model
                    # predicts the velocity eps-x0. Draw sigma directly (warped by
                    # `shift` the way the inference schedule is) rather than an
                    # integer timestep, so the cache covers exactly the sigmas a
                    # sampler will visit. The old `x0 + s*eps` never reached pure
                    # noise at s=1 and never reached clean data at s=0, so both
                    # ends of the trajectory were supervised at the wrong point.
                    u = torch.rand((lat.shape[0],), generator=g)
                    s = a.shift * u / (1 + (a.shift - 1) * u) if a.shift > 0 else u
                    t = (s * 1000).round().clamp(0, 999).to(torch.int64)
                    sigma = s.view(-1, *([1] * (lat.dim() - 1)))
                    noisy_all.append((1 - sigma) * lat + sigma * noise)
                else:
                    t = torch.randint(0, 1000, (lat.shape[0],), generator=g)
                    sigma = (t.float() / 1000).view(-1, *([1] * (lat.dim() - 1)))
                    noisy_all.append(lat + noise * sigma)
                t_all.append(t)
            noisy_all, t_all = torch.cat(noisy_all, 0), torch.cat(t_all, 0)

            for off in range(0, draws, max(1, a.draw_batch)):
                noisy = noisy_all[off:off + max(1, a.draw_batch)]
                t = t_all[off:off + max(1, a.draw_batch)]
                n = noisy.shape[0]
                # Inputs are cast to the teacher's element type explicitly. Sniffing
                # it from the module does not work: under `torch_dtype=bfloat16`
                # diffusers leaves some Wan parameters in f32, so the first one says
                # f32 while the patch embedding — which rejects an f32 input — is
                # bf16.
                nd, td = noisy.to(a.device, dtype=TDT), t.to(a.device)
                pc = prompt.to(a.device, dtype=TDT).expand(n, *prompt.shape[1:])
                result = teacher(nd, td, pc)
                target = result['noise_pred'].float()
                if a.guidance > 0:
                    # Deliberately a second forward rather than one batch of 2n:
                    # concatenating cond and uncond measured 8% *slower*, because
                    # the teacher pass is already compute-bound and the extra input
                    # materialisation costs more than the saved launch.
                    pu = negative.float().to(a.device, dtype=TDT).expand(n, *negative.shape[1:])
                    uncond = teacher(nd, td, pu)['noise_pred'].float()
                    target = uncond + a.guidance * (target - uncond)
                hidden = result.get('hidden_states', [])
                selected = select_layers(len(hidden), a.relation_layers)
                num_layers = len(selected)
                for b in range(n):
                    tensors = {
                        'noisy_latents': noisy[b:b + 1].contiguous(),
                        'timestep': t[b:b + 1].contiguous(),
                        'prompt_embeds': prompt.half().cpu().contiguous(),
                        'teacher_noise_pred': target[b:b + 1].cpu().contiguous(),
                    }
                    if student_prompt is not None:
                        tensors['student_prompt_embeds'] = student_prompt.float().cpu().contiguous()
                    for out_idx, layer in enumerate(selected):
                        h = torch.nn.functional.normalize(hidden[layer][b:b + 1].float(), dim=-1)
                        tensors[f'teacher_relation.{out_idx}'] = (h @ h.transpose(-1, -2)).half().cpu().contiguous()
                    name = f'shard-{k:06d}.safetensors'
                    save_file(tensors, out / name)
                    shards.append(name)
                    k += 1
                    if not shapes:
                        for key, value in tensors.items():
                            shapes[key] = {'name': key, 'shape': list(value.shape), 'dtype': str(value.dtype).replace('torch.', '').upper()}

    # The scheduler string is the cache's own record of the convention its targets
    # were built under. A student trained on `flow` targets and sampled as if they
    # were `legacy` ones produces garbage silently, so the two must travel together.
    scheduler = a.scheduler if a.noising == 'legacy' else f'{a.scheduler}-flow-shift{a.shift:g}'
    if a.guidance > 0:
        scheduler += f'-cfg{a.guidance:g}'
    # Teacher precision belongs in that record for the same reason: bf16 targets
    # differ from f32 ones by ~7% mean relative error, which is far too small to
    # notice by eye in a shard and far too large to be a rounding detail when two
    # caches get compared or concatenated.
    if a.teacher_dtype != 'f32':
        scheduler += f'-{a.teacher_dtype}'

    manifest = {
        'format_version': 1,
        'teacher': a.teacher_name,
        'scheduler': scheduler,
        'shards': shards,
        'tensors': list(shapes.values()),
        'hidden_relation_layers': list(range(num_layers)),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'wrote {len(shards)} shards from {len(files)} clips ({num_layers} relation layers) to {out}')


if __name__ == '__main__':
    main()
