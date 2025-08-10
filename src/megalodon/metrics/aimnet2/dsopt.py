import torch
import numpy as np
from tqdm.auto import tqdm
from time import perf_counter


@torch.jit.script
class FIRE():
    def __init__(self, M:int, N:int, device:'str'):
        ## default parameters
        self.dt_max = 0.1
        self.Nmin = 5
        self.maxstep = 0.1
        self.finc = 1.2
        self.fdec = 0.8
        self.astart = 0.1
        self.fa = 0.99
        self.dt_start = 0.1

        self.v = torch.zeros(M, N, 3, device=device)
        self.Nsteps = torch.zeros(M, dtype=torch.long, device=device)
        self.dt = torch.full((M,), self.dt_start, device=device)
        self.a = torch.full((M,), self.astart, device=device)

    def __call__(self, forces):
        vf = (forces * self.v).flatten(-2, -1).sum(-1)
        w_vf = vf > 0.0
        if w_vf.all():
            a = self.a.unsqueeze(-1).unsqueeze(-1)
            v = self.v
            f = forces
            self.v = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
            self.Nsteps += 1
        elif w_vf.any():
            a = self.a[w_vf].unsqueeze(-1).unsqueeze(-1)
            v = self.v[w_vf]
            f = forces[w_vf]
            self.v[w_vf] = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)

            w_N = self.Nsteps > self.Nmin
            w_vfN = w_vf & w_N
            self.dt[w_vfN] = (self.dt[w_vfN] * self.finc).clamp(max=self.dt_max)
            self.a[w_vfN] *= self.fa
            self.Nsteps[w_vfN] += 1

        w_vf = ~w_vf
        if w_vf.all():
            self.v[:] = 0.0
            self.a[:] = torch.tensor(self.astart, device=self.a.device)
            self.dt[:] *= self.fdec
            self.Nsteps[:] = 0
        elif w_vf.any():
            self.v[w_vf] = torch.tensor(0.0, device=self.v.device)
            self.a[w_vf] = torch.tensor(self.astart, device=self.a.device)
            self.dt[w_vf] *= self.fdec
            self.Nsteps[w_vf] = torch.tensor(0, device=self.v.device)

        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v += dt * forces
        dr = dt * self.v
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
        dr *= (self.maxstep / normdr).clamp(max=1.0)
        return dr

    def clean(self, mask) -> bool:
        self.v = self.v[mask]
        self.Nsteps = self.Nsteps[mask]
        self.dt = self.dt[mask]
        self.a = self.a[mask]
        return True

    def extend(self, n: int) -> bool:
        self.v = torch.cat([self.v, torch.zeros(n, self.v.shape[1], self.v.shape[2], dtype=self.v.dtype, device=self.v.device)], dim=0)
        self.Nsteps = torch.cat([self.Nsteps, torch.zeros(n, dtype=self.Nsteps.dtype, device=self.Nsteps.device)], dim=0)
        self.dt = torch.cat([self.dt, torch.full((n, ), self.dt_start, device=self.dt.device)], dim=0)
        self.a = torch.cat([self.a, torch.full((n, ), self.astart, device=self.a.device)], dim=0)
        return True


def group_opt(model, coord, numbers, charge, batchsize=None, fmax=2e-3, max_nstep=5000, device="cuda"):
    num_converged = 0
    converged_coord = []
    converged_idx = []
    converged_energy = []
    unconverged_coord = []
    unconverged_idx = []
    unconverged_energy = []
    idx = torch.arange(coord.shape[0]).cuda()
    runtime = torch.zeros_like(idx, dtype=torch.float32)
    nstep = torch.zeros_like(idx)

    if batchsize is None:
       batchsize = len(numbers)
    act_idx = idx[:batchsize]
    act_i = batchsize
    act_coord = coord[act_idx]
    act_numbers = numbers[act_idx]
    act_charge = charge[act_idx]
    act_runtime = torch.zeros(act_idx.shape[0]).cuda() 
    act_nstep = torch.zeros(act_idx.shape[0]).cuda()

    istep = 0
    opt = FIRE(act_coord.shape[0], act_coord.shape[1], device)
    pbar = tqdm(total=len(coord), leave=True)
    pbar1 = tqdm(total=max_nstep, leave=False)
    _t = perf_counter()
    with torch.no_grad():
      while istep < max_nstep:
        _need_ext = False

        with torch.enable_grad():
          act_coord.requires_grad_(True)
          d = dict(coord=act_coord, numbers=act_numbers, charge=act_charge)
          dout = model(d)
          e = dout['energy']
          if 'forces' in dout:
              f = dout['forces']
          else:
              f = - torch.autograd.grad([e.sum()], [act_coord], retain_graph=False)[0]

        w = act_nstep >= max_nstep
        if w.any():
           nw = ~w
           unconverged_coord.append(act_coord[w].detach())
           unconverged_idx.append(act_idx[w])
           unconverged_energy.append(e[w].detach())
           nstep[act_idx[w]] = act_nstep[w].long()
           runtime[act_idx[w]] = act_runtime[w]
           act_idx = act_idx[nw]
           act_coord = act_coord[nw]
           act_numbers = act_numbers[nw]
           act_charge = act_charge[nw]
           act_runtime = act_runtime[nw]
           act_nstep = act_nstep[nw]
           f = f[nw]
           e = e[nw]
           opt.clean(nw)
           _need_ext = True

        if istep and not istep % 10:
          _t1 = perf_counter()
          act_runtime += (_t1 - _t) / act_runtime.shape[0]
          _t = _t1
          act_nstep += 10
          _fmax = f.norm(dim=-1).max(dim=-1)[0]
          w = _fmax < fmax
          if w.any():
            pbar.update(w.sum().item())
            num_converged += w.sum().item()
            converged_coord.append(act_coord[w].detach())
            converged_idx.append(act_idx[w])
            converged_energy.append(e[w].detach())
            nstep[act_idx[w]] = act_nstep[w].long()
            runtime[act_idx[w]] = act_runtime[w]

            nw = ~w
            act_idx = act_idx[nw]
            act_coord = act_coord[nw]
            act_numbers = act_numbers[nw]
            act_charge = act_charge[nw]
            act_runtime = act_runtime[nw]
            act_nstep = act_nstep[nw]

            f = f[nw]
            opt.clean(nw)
            _need_ext = act_i <= coord.shape[0]

        _prev_istep = istep

        istep += 1

        assert act_coord.shape[0] == opt.v.shape[0]
        if act_coord.numel() > 0:
            act_coord += opt(f)

        if _need_ext: 
            _n_add = batchsize - act_idx.shape[0] 
            act_idx = torch.cat([act_idx, idx[act_i:act_i+_n_add]], dim=0)
            _n1 = act_coord.shape[0]
            act_coord = torch.cat([act_coord, coord[act_i:act_i+_n_add]], dim=0)
            act_numbers = torch.cat([act_numbers, numbers[act_i:act_i+_n_add]], dim=0)
            act_charge = torch.cat([act_charge, charge[act_i:act_i+_n_add]], dim=0)
            act_runtime = torch.cat([act_runtime, torch.zeros(act_charge.shape[0]-act_runtime.shape[0], device='cuda')], dim=0)
            act_nstep = torch.cat([act_nstep, torch.zeros(act_charge.shape[0]-act_nstep.shape[0], device='cuda')])

            act_i += _n_add
            opt.extend(act_coord.shape[0] - _n1)
            istep = 0

        pbar1.update(istep - _prev_istep)

        if act_coord.numel() == 0:
            break

    pbar.close()
    pbar1.close()
    converged = torch.zeros_like(charge)
    res_coord = torch.zeros_like(coord)
    res_forces = torch.zeros_like(coord)
    res_energy = torch.zeros_like(charge, dtype=torch.double)
    if len(converged_idx) > 0:
        opt_coord = torch.cat(converged_coord, dim=0)
        opt_idx = torch.cat(converged_idx, dim=0)
        opt_energy = torch.cat(converged_energy)

        converged[opt_idx] = 1
        res_coord[opt_idx] = opt_coord
        res_energy[opt_idx] = opt_energy
        res_forces[opt_idx] = fmax/1.73205
    if not w.all():
        res_coord[act_idx] = act_coord
        res_energy[act_idx] = e
        res_forces[act_idx] = f
        nstep[act_idx] = max_nstep

    if len(unconverged_idx) > 0:
        opt_coord = torch.cat(unconverged_coord, dim=0)
        opt_idx = torch.cat(unconverged_idx, dim=0)
        opt_energy = torch.cat(unconverged_energy)

        res_coord[opt_idx] = opt_coord
        res_energy[opt_idx] = opt_energy
        res_forces[opt_idx] = -fmax/1.73205

    return converged, res_coord, res_energy, res_forces, nstep