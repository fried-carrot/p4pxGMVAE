import torch
from torch import nn
from torch.nn import functional as F

class GMVAE_ZINB(nn.Module):

	def __init__(self, args):
		super(GMVAE_ZINB, self).__init__()
		self.device = args.device
		self.args = args
		self.onehot = torch.nn.functional.one_hot(torch.arange(0, self.args.K), 
                                            num_classes=self.args.K).to(self.device)*(1.0)

		self.pi1 = nn.Linear(self.args.input_dim, self.args.h_dim)
		self.pi2 = nn.Linear(self.args.h_dim, self.args.K)

		self.mu_x1 = nn.Linear(self.args.input_dim + self.args.K, self.args.h_dim1)
		self.mu_x2 = nn.Linear(self.args.h_dim1, self.args.h_dim2)
		self.mu_x3 = nn.Linear(self.args.h_dim2, self.args.z_dim)
  
		self.logvar_x1 = nn.Linear(self.args.input_dim + self.args.K, self.args.h_dim1)
		self.logvar_x2 = nn.Linear(self.args.h_dim1, self.args.h_dim2)
		self.logvar_x3 = nn.Linear(self.args.h_dim2, self.args.z_dim)
		
		self.mu_w1 = nn.Linear(self.args.K, self.args.h_dim)
		self.mu_w2 = nn.Linear(self.args.h_dim, self.args.z_dim)
  
		self.logvar_w1 = nn.Linear(self.args.K, self.args.h_dim)
		self.logvar_w2 = nn.Linear(self.args.h_dim, self.args.z_dim)

		self.recon1 = nn.Linear(self.args.z_dim, self.args.h_dim2)

		self.recon2z = nn.Linear(self.args.h_dim2, self.args.h_dim1)
		self.recon3z = nn.Linear(self.args.h_dim1, self.args.input_dim)

		self.recon2m = nn.Linear(self.args.h_dim2, self.args.h_dim1)
		self.recon3m = nn.Linear(self.args.h_dim1, self.args.input_dim)

		self.recon2d = nn.Linear(self.args.h_dim2, self.args.h_dim1)
		self.recon3d = nn.Linear(self.args.h_dim1, self.args.input_dim)

	def pi_of_x(self,x):
		x = F.relu(self.pi1(x))
		pi_x = F.softmax(self.pi2(x), dim=1)
		return pi_x

	def musig_of_z(self,x):
		batchSize = x.size(0)
		mu_zs = torch.empty(batchSize, self.args.z_dim, self.args.K, 
                      device=self.device, requires_grad=False)
		logvar_zs = torch.empty(batchSize, self.args.z_dim, self.args.K, 
                          device=self.device, requires_grad=False)
		for i in range(self.args.K):
			xy = torch.cat((x, self.onehot[i,:].expand(x.size(0),self.args.K)), 1)
			mu_zs[:, :, i] = self.mu_x3(F.relu(self.mu_x2(F.relu(self.mu_x1(xy)))))
			logvar_zs[:, :, i] = self.logvar_x3(F.relu(self.logvar_x2(F.relu(self.logvar_x1(xy)))))

		# mu_z = self.mu_x2(F.relu(self.mu_x1(xy)))
		# logvar_z = self.logvar_x2(F.relu(self.logvar_x1(xy)))
		return mu_zs, logvar_zs  

	def musig_of_genz(self,y,batchsize):
		mu_genz = self.mu_w2(F.relu(self.mu_w1(y)))
		logvar_genz = self.logvar_w2(F.relu(self.logvar_w1(y)))
		return torch.t(mu_genz).expand(batchsize, self.args.z_dim, self.args.K),\
      torch.t(logvar_genz).expand(batchsize, self.args.z_dim, self.args.K)


	def decoder(self, z_sample):
		
		h = F.relu(self.recon1(z_sample))
		
		hz = F.relu(self.recon2z(h))  #zero_probability
		hz = self.recon3z(hz)         #zero_probability

		hm = F.relu(self.recon2m(h))  #mean
		hm = self.recon3m(hm)         #mean

		hd = F.relu(self.recon2d(h))  #dispersion
		hd = self.recon3d(hd)         #dispersion

		Yzerop = F.sigmoid(hz)   #scRNA zinb zero probability
		# Ymean = torch.exp(hm)   #scRNA zinb mean    # This cause nan
		# Ydisper = torch.exp(hd)   #scRNA zinb dispersion  # This cause nan
		# Ymean = F.relu(hm)+1e-10   #scRNA zinb mean   # This didn't cause nan 
		# Ydisper = F.relu(hd)+1e-10   #scRNA zinb dispersion  # This didn't cause nan
		Ymean = nn.ELU(1)(hm*(1.0))+1.0
		Ydisper = nn.ELU(1)(hd*(1.0))+1.0
		return Yzerop, Ymean, Ydisper

	def reparameterize(self, mu, logvar):
		'''
		compute z = mu + std * epsilon
		'''
		if self.training:
			# do this only while training
			# compute the standard deviation from logvar
			std = torch.exp(0.5 * logvar)
			# sample epsilon from a normal distribution with mean 0 and
			# variance 1
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, Xdata, Xtarget):

		batchsize = Xdata.size(0)
		Xdata=Xdata.to(self.device)
		Xtarget=Xtarget.to(self.device)

		X_conv = Xdata
		pi_x = self.pi_of_x(X_conv)
		mu_zs, logvar_zs = self.musig_of_z(X_conv)
		
		mu_genz, logvar_genz = self.musig_of_genz(self.onehot, batchsize)
		z_samples = self.reparameterize(mu_zs, logvar_zs)
		x_recons_zerop = torch.stack([self.decoder(z_samples[:,:,ii].squeeze())[0] \
                                                for ii in (range(z_samples.size(2)))], dim=len(z_samples[:,:,0].squeeze().shape)\
                                                    )
		x_recons_mean = torch.stack([self.decoder(z_samples[:,:,ii].squeeze())[1] \
                                                for ii in (range(z_samples.size(2)))], dim=len(z_samples[:,:,0].squeeze().shape)\
                                                    )
		x_recons_disper = torch.stack([self.decoder(z_samples[:,:,ii].squeeze())[2] \
                                                for ii in (range(z_samples.size(2)))], dim=len(z_samples[:,:,0].squeeze().shape)\
                                                    )
		pi_x_expanded = pi_x.unsqueeze(1).expand(batchsize,X_conv.size(-1),self.args.K)
		x_recon_zerop = torch.sum(x_recons_zerop*pi_x_expanded, dim=pi_x_expanded.dim()-1)
		x_recon_mean = torch.sum(x_recons_mean*pi_x_expanded, dim=pi_x_expanded.dim()-1)
		x_recon_disper = torch.sum(x_recons_disper*pi_x_expanded, dim=pi_x_expanded.dim()-1)
		return pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz, x_recons_zerop, x_recons_mean, \
      x_recons_disper, x_recon_zerop, x_recon_mean, x_recon_disper #, x_recon 
