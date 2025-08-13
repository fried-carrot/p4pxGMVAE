import torch
from torch import nn
from torch.nn import functional as F

def gmvae_losses(Xdata, Xtarget, pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz,\
    x_recons_zerop, x_recons_mean, x_recons_disper): # recon_X, X, mu_w, logvar_w, qz,
	#pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz, 
	# x_recons_zerop, x_recons_mean, x_recons_disper, x_recon_zerop, x_recon_mean, x_recon_disper
	# print(Xtarget.device)
	eps = 1e-10
	batchsize = Xdata.size(0)
	K = pi_x.size(1)
	z_dim = mu_zs.size(1)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Xdata = Xdata.to(device).view(Xdata.size(0),-1)
	Xdata = Xdata.to(device)
	pi_target = torch.nn.functional.one_hot(Xtarget, num_classes=K).to(device)
	
	KLD_pi = torch.sum(pi_x*(torch.log(pi_x + eps)-torch.log(pi_target + eps)))
	
	KLD_gaussian = 0.5 * (((logvar_genz - logvar_zs) + \
		((logvar_zs.exp() + (mu_zs - mu_genz).pow(2))/logvar_genz.exp())) - 1)
	
	KLD_gaussian = torch.sum(KLD_gaussian * pi_x.unsqueeze(1).expand(batchsize,z_dim,K))
	
	pi_x_expanded = pi_x.unsqueeze(1).expand(batchsize, Xdata.shape[-1], K)

	###################
	# zinb_loss_i
	# zero case
	zero_cases_i = torch.stack([((-1.0)*torch.log( x_recons_zerop[:,:,ii] \
		+ (1 - x_recons_zerop[:,:,ii])\
			*torch.pow(((x_recons_disper[:,:,ii])/(x_recons_mean[:,:,ii] + x_recons_disper[:,:,ii] + eps)),\
				x_recons_disper[:,:,ii]) + eps)) for ii in range(K)], dim=x_recons_disper.dim()-1)
	# print(f'zero_cases_i.shape_{zero_cases_i.shape}')

	# non zero case
	nzero_cases1_i = torch.stack([(torch.lgamma(Xdata+1) + torch.lgamma(x_recons_disper[:,:,ii] + eps) \
        + (-1.0)*torch.lgamma(Xdata + x_recons_disper[:,:,ii]+eps)  ) for ii in range(K)], \
            dim=x_recons_disper.dim()-1)
	# print(f'nzero_cases1_i.shape_{nzero_cases1_i.shape}')

	nzero_cases2_i = torch.stack([( (-1.0)*torch.log(1.0 - x_recons_zerop[:,:,ii] + eps) \
     + (-1.0)*x_recons_disper[:,:,ii]*torch.log(x_recons_disper[:,:,ii] + eps) \
         + (-1.0)*Xdata*torch.log(x_recons_mean[:,:,ii] + eps) \
             + (x_recons_disper[:,:,ii] + Xdata)*torch.log(x_recons_disper[:,:,ii] + x_recons_mean[:,:,ii] + eps) ) \
                 for ii in range(K)], dim=x_recons_disper.dim()-1)
	# print(f'nzero_cases2_i.shape_{nzero_cases2_i.shape}')

	nzero_cases_i = nzero_cases1_i + nzero_cases2_i
	# print(f'nzero_cases_i.shape_{nzero_cases_i.shape}')
	## non zero case done.

	#Choose one case
	Xdata_expanded = Xdata.unsqueeze(Xdata.dim()).expand(batchsize,Xdata.size(1),K)
	# print(f'Xdata_expanded.shape_{Xdata_expanded.shape}')

	zinb_losses = torch.where(torch.le(Xdata_expanded, 0.01), zero_cases_i, nzero_cases_i)
	# print(f'zinb_losses_{zinb_losses.shape}')

	#Sum them up
	zinb_loss = torch.sum(zinb_losses*pi_x_expanded)
	# print(f'zinb_loss_{zinb_loss}')

	total_loss = KLD_gaussian + KLD_pi + zinb_loss

	return total_loss, KLD_gaussian, KLD_pi, zinb_loss


	# recon_loss = torch.stack([F.binary_cross_entropy(
	# x_recons[:,:,ii],  Xdata[:,:], reduction='none') for ii in range(K)], dim=x_recons.dim()-1)

	# recon_loss = torch.stack([F.mse_loss(
	# x_recons[:,:,ii],  Xdata[:,:], reduction='none') for ii in range(K)], dim=x_recons.dim()-1)

	# reconst_loss = torch.sum(recon_loss*pi_x_expanded)