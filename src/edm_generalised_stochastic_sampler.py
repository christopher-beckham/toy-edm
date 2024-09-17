"""
NOTE: This particular file is licensed under CC-NC-SA (see `LICENSE-NC_SA`) since it
implements the most "complete" version of the algorithm presented by EDM. EDM's original
repository is specifically licensed under CC-NC-SA and I have done the same here.

Having said that, this method is a heavily modified version of `sample_euler` in crowsonkb's
`k-diffusion` repository, which is found here https://github.com/crowsonkb/k-diffusion. That
particular repository however is MIT-licensed. (See `LICENSE-MIT`.)

The deterministic version of this algorithm -- found in the parent directory -- retains the
MIT license.
"""

import pdb
import math
import torch
from torch.autograd import grad
from tqdm import trange

def edm_generalised_stochastic_sampler(
    model, 
    x, 
    timesteps,
    extra_args       = None, 
    callback         = None, 
    disable          = None, 
    sigma_fn         = lambda t: t*1.0,
    s_fn             = lambda t: t*0 + 1,
    stochastic: bool = True,
    # NOTE: the S_{} variables are only for when stochasticity is enabled.
    s_churn: float   = 0., 
    s_tmin: float    = 0., 
    s_tmax: float    = 1.,
    s_noise: float   = 1.,
    debug: bool      = False,
    use_heun: bool   = True,
):
    
    t = timesteps
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    trajectory = []
    for i in trange(len(timesteps) - 1, disable=disable):
        
        ti = (t[i]*s_in).view(-1, 1)
        ti.requires_grad = True
            
        if stochastic:
            gamma_i = min(s_churn / len(timesteps), math.sqrt(2)-1) \
                if s_tmin <= sigma_fn(ti[0:1]).item() <= s_tmax else 0.
        else:
            gamma_i = 0.
        
        if debug:
            print(i, gamma_i, "-->", s_tmin, sigma_fn(ti[0:1]).item(), s_tmax, s_tmin <= t[i] <= s_tmax)
        
        ti_hat = (t[i]*s_in) + gamma_i*(t[i]*s_in)
        ti_hat = ti_hat.view(-1, 1)
        ti_hat.requires_grad = True
        
        dot_sigma_fn_ti_hat = grad(sigma_fn(ti_hat).sum(), ti_hat)[0]
        dot_s_fn_ti_hat = grad(s_fn(ti_hat).sum(), ti_hat)[0]
        
        if use_heun:
            tip1 = (t[i+1]*s_in).view(-1, 1)
            tip1.requires_grad = True
            dot_sigma_fn_tip1 = grad(sigma_fn(tip1).sum(), tip1)[0]
            dot_s_fn_tip1 = grad(s_fn(tip1).sum(), tip1)[0]
        
        with torch.no_grad():
            
            if stochastic:
                eps_i = torch.ones_like(x).normal_(0, s_noise**2)
                x_hat = x + torch.sqrt( sigma_fn(ti_hat)**2 - sigma_fn(ti)**2 ) * eps_i
            else:
                x_hat = x
        
            denoised = model(x_hat / s_fn(ti_hat), sigma_fn(ti_hat), **extra_args)

            d = ( (dot_sigma_fn_ti_hat / sigma_fn(ti_hat)) + (dot_s_fn_ti_hat / s_fn(ti_hat)) ) * x_hat - \
                ( dot_sigma_fn_ti_hat * s_fn(ti_hat) / sigma_fn(ti_hat) ) * denoised

            if debug:
                print(f"{i} {x.shape} {sigma_fn(ti).shape} {sigma_fn(ti)[0]} : {denoised.shape}", "{:.3f}".format((d**2).mean()))
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

            dt = t[i+1] - (t[i] + gamma_i*t[i])
            if i==0 and debug:
                print(dt)
                
            # Take the Euler step
            x = x_hat + dt*d
            
            if use_heun and sigma_fn(tip1)[0].item() != 0:
                denoised2 = model(x_next / s_fn(tip1), sigma_fn(tip1), **extra_args)
                d2 = ( (dot_sigma_fn_tip1 / sigma_fn(tip1)) + (dot_s_fn_tip1 / s_fn(tip1)) ) * x - \
                     ( dot_sigma_fn_tip1 * s_fn(tip1) / sigma_fn(tip1) ) * denoised2
                x = x_hat + dt*(0.5*d + 0.5*d2)
            else:
                pass
                
            #else:
            #    print("we hit 0", sigma_fn(tip1))
            
            trajectory.append(x)
            
    return x, trajectory