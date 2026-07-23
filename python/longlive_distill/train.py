from __future__ import annotations
import argparse,importlib,json,itertools
from pathlib import Path
import torch,yaml
from .losses import distillation_loss
from .model import build_student,build_toy_teacher

def factory(spec,cfg):
    if spec=="builtin:student":return build_student(cfg)
    if spec=="builtin:toy-teacher":return build_toy_teacher(cfg)
    module,name=spec.split(":",1);return getattr(importlib.import_module(module),name)(cfg)
def samples(cfg,device,dtype):
    # Yields (latents, teacher_prompt_embeds, student_prompt_embeds). The student
    # may condition on a smaller/different text encoder than the teacher; when a
    # shard omits student_prompt_embeds (or no student_text_shape is configured)
    # the student reuses the teacher embeddings, preserving prior behaviour.
    if path:=cfg.get("path"):
        files=sorted(Path(path).glob("*.pt"))
        if not files:raise RuntimeError(f"No .pt latent shards in {path}")
        for file in itertools.cycle(files):
            item=torch.load(file,map_location="cpu",weights_only=True)
            teacher_prompt=item["prompt_embeds"].to(device=device,dtype=dtype)
            student_prompt=item.get("student_prompt_embeds",item["prompt_embeds"]).to(device=device,dtype=dtype)
            yield item["latents"].to(device=device,dtype=dtype),teacher_prompt,student_prompt
    else:
        student_shape=cfg.get("student_text_shape",cfg["text_shape"])
        while True:yield torch.randn(*cfg["latent_shape"],device=device,dtype=dtype),torch.randn(*cfg["text_shape"],device=device,dtype=dtype),torch.randn(*student_shape,device=device,dtype=dtype)
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());device=torch.device(cfg["training"].get("device","cuda" if torch.cuda.is_available() else "cpu"));dtype=torch.bfloat16 if cfg["training"].get("bf16",False) and device.type=="cuda" else torch.float32
    teacher=factory(cfg["teacher"]["factory"],cfg["teacher"]["model"]).to(device,dtype).eval();student=factory(cfg["student"]["factory"],cfg["student"]["model"]).to(device,dtype).train()
    if ckpt:=cfg["teacher"].get("checkpoint"):
        state=torch.load(ckpt,map_location="cpu",weights_only=True);teacher.load_state_dict(state.get("model",state),strict=cfg["teacher"].get("strict",True))
    for p_ in teacher.parameters():p_.requires_grad_(False)
    opt=torch.optim.AdamW(student.parameters(),lr=float(cfg["training"]["lr"]),weight_decay=float(cfg["training"].get("weight_decay",.01)));stream=samples(cfg["data"],device,dtype);out=Path(cfg["training"]["output"]);out.mkdir(parents=True,exist_ok=True)
    for step in range(1,int(cfg["training"]["steps"])+1):
        lat,teacher_prompt,student_prompt=next(stream);time=torch.randint(0,1000,(lat.shape[0],),device=device);noisy=lat+torch.randn_like(lat)*(time.float()/1000).view(-1,1,1,1,1)
        with torch.no_grad():target=teacher(noisy,time,teacher_prompt)
        pred=student(noisy,time,student_prompt);loss,metrics=distillation_loss(pred,target,cfg["loss"]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),cfg["training"].get("grad_clip",1.0));opt.step()
        if step%int(cfg["training"].get("log_every",10))==0 or step==1:print(json.dumps({"step":step,**metrics}))
    torch.save({"model":student.state_dict(),"config":cfg},out/"student.pt")
if __name__=="__main__":main()
