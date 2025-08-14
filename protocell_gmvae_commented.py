#!/usr/bin/env python
"""
PROTOCELL WITH GMVAE INTEGRATION - MODIFIED FROM P4P
This file is based on ProtoCell4P/src/model.py with GMVAE replacing the autoencoder
Only modifications are:
1. Removed encoder/decoder networks
2. Added GMVAE integration
3. Replaced distance-based similarity with z-score calculation

SOURCE: ProtoCell4P/src/model.py (lines 1-220)
GitHub: https://github.com/deepomicslab/ProtoCell4P
"""

import torch
import torch.nn as nn
import sys

# ADDITION: Import GMVAE (NOT in original P4P)
sys.path.append('bulk2sc_GMVAE/_model_source_codes')
from GMVAE_scrna_zinb import GMVAE_ZINB

# ADDITION: GMVAE args class for initialization (NOT in original P4P)
class GMVAEArgs:
    """Helper class to initialize GMVAE with required arguments"""
    def __init__(self, input_dim, K, device):
        self.input_dim = input_dim
        self.K = K
        self.z_dim = 32
        self.h_dim = 128
        self.h_dim1 = 128
        self.h_dim2 = 64
        self.device = device
        self.dataset = 'lupus'
        self.seed = 123

# SOURCE: ProtoCell4P/src/model.py lines 6-220
# Class structure is EXACT copy with specific modifications marked
class ProtoCell(nn.Module):
    # SOURCE: ProtoCell4P/src/model.py line 6
    def __init__(self, input_dim, h_dim, z_dim, n_layers, n_proto, n_classes, lambdas, n_ct=None, device="cpu", d_min=1):
        # SOURCE: ProtoCell4P/src/model.py lines 7-26
        # All initialization parameters are EXACT copies
        super(ProtoCell, self).__init__()
        
        self.device = device
        
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_proto = n_proto
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_ct = n_ct
        self.d_min = d_min
        
        assert self.n_layers > 0
        
        self.lambda_1 = lambdas["lambda_1"]
        self.lambda_2 = lambdas["lambda_2"]
        self.lambda_3 = lambdas["lambda_3"]
        self.lambda_4 = lambdas["lambda_4"]
        self.lambda_5 = lambdas["lambda_5"]
        self.lambda_6 = lambdas["lambda_6"]
        
        # REMOVED: ProtoCell4P/src/model.py lines 28-34 (encoder/decoder networks)
        # Original code:
        # self.enc_i = nn.Linear(self.input_dim, self.h_dim)
        # self.enc_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        # self.enc_z = nn.Linear(self.h_dim, self.z_dim)
        # self.dec_z = nn.Linear(self.z_dim, self.h_dim)
        # self.dec_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        # self.dec_i = nn.Linear(self.h_dim, self.input_dim)
        
        # ADDITION: Initialize GMVAE instead of encoder/decoder
        gmvae_args = GMVAEArgs(input_dim, n_proto, device)
        self.gmvae = GMVAE_ZINB(gmvae_args).to(device)
        self.gmvae.eval()  # Set to eval mode since we're using pretrained
        
        # SOURCE: ProtoCell4P/src/model.py lines 36-38
        # Importance network is EXACT copy
        self.imp_i = nn.Linear(self.input_dim, self.h_dim)
        self.imp_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.imp_p = nn.Linear(self.h_dim, self.n_proto * self.n_classes)
        
        # SOURCE: ProtoCell4P/src/model.py lines 40-41
        self.activate = nn.LeakyReLU()
        
        # REMOVED: ProtoCell4P/src/model.py line 42 (prototypes parameter)
        # Original: self.prototypes = nn.parameter.Parameter(torch.empty(self.n_proto, self.z_dim), requires_grad = True)
        # Not needed with GMVAE approach
        
        # SOURCE: ProtoCell4P/src/model.py lines 43-44
        # Classifier and loss function are EXACT copies
        self.clf = nn.Linear(self.n_proto, self.n_classes)
        self.ce_ = nn.CrossEntropyLoss(reduction="mean")
        
        # SOURCE: ProtoCell4P/src/model.py lines 46-56 (modified)
        # Weight initialization - removed encoder/decoder init
        # nn.init.xavier_normal_(self.enc_i.weight)  # REMOVED
        # nn.init.xavier_normal_(self.enc_z.weight)  # REMOVED
        # nn.init.xavier_normal_(self.dec_z.weight)  # REMOVED
        # nn.init.xavier_normal_(self.dec_i.weight)  # REMOVED
        nn.init.xavier_normal_(self.imp_i.weight)
        nn.init.xavier_normal_(self.imp_p.weight)
        nn.init.xavier_normal_(self.clf.weight)
        # nn.init.xavier_normal_(self.prototypes)  # REMOVED
        
        for i in range(self.n_layers - 1):
            # nn.init.xavier_normal_(self.enc_h[i].weight)  # REMOVED
            # nn.init.xavier_normal_(self.dec_h[i].weight)  # REMOVED
            nn.init.xavier_normal_(self.imp_h[i].weight)
        
        # SOURCE: ProtoCell4P/src/model.py lines 58-62
        # Cell type classifier is EXACT copy
        if self.n_ct is not None:
            self.ct_clf1 = nn.Linear(self.n_proto, self.n_ct)
            self.ct_clf2 = nn.Linear(self.n_proto * self.n_classes, self.n_ct)
            nn.init.xavier_normal_(self.ct_clf1.weight)
            nn.init.xavier_normal_(self.ct_clf2.weight)
    
    # SOURCE: ProtoCell4P/src/model.py lines 64-96 (modified)
    def forward(self, x, y, ct=None, sparse=True):
        # SOURCE: ProtoCell4P/src/model.py lines 65-75
        # Data preparation is EXACT copy
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)
        
        # MODIFICATION: ProtoCell4P/src/model.py line 76
        # Original: z = self.encode(x)
        # Replaced with z-score calculation
        z_scores = self.compute_zscores(x)  # NEW METHOD (see below)
        
        # SOURCE: ProtoCell4P/src/model.py line 78
        # Importance calculation is EXACT copy
        import_scores = self.compute_importance(x)  # (n_cell, n_proto, n_class)
        
        # MODIFICATION: ProtoCell4P/src/model.py lines 80-81
        # Original distance-based similarity:
        # c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        # c_logits = (1 / (c2p_dists+0.5))[:,None,:].matmul(import_scores).squeeze(1)
        # Replaced with z-score based:
        c_logits = z_scores[:,None,:].matmul(import_scores).squeeze(1)  # (n_cell, n_classes)
        
        # SOURCE: ProtoCell4P/src/model.py lines 83-96
        # Patient-level aggregation and loss computation is EXACT copy
        logits = torch.stack([c_logits[split_idx[i]:split_idx[i+1]].mean(dim=0) 
                             for i in range(len(split_idx)-1)])
        
        clf_loss = self.ce_(logits, y)
        
        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf2(import_scores.reshape(-1, self.n_proto * self.n_classes))
            ct_loss = self.ce_(ct_logits, torch.tensor([j for i in ct for j in i]).to(self.device))
        else:
            ct_loss = 0
        
        # MODIFICATION: ProtoCell4P/src/model.py lines 88-95
        # Removed reconstruction and prototype losses (lambda_1 through lambda_5)
        # Original had:
        # x_hat = self.decode(z)
        # rec_loss = self.rec_(x_hat, x)
        # proto_loss, sep_loss, divers_loss = self.compute_proto_loss(z, c2p_dists, import_scores)
        # total_loss = rec_loss + self.lambda_1 * proto_loss + ...
        # Now only classification loss:
        total_loss = clf_loss + self.lambda_6 * ct_loss
        
        if ct is not None:
            return total_loss, logits, ct_logits    
        return total_loss, logits
    
    # NEW METHOD: Z-score calculation (NOT in original P4P)
    # This is the ONLY completely new code addition
    def compute_zscores(self, x):
        """
        Compute z-scores using GMVAE parameters
        z_score = (μ(x,k) - μ_g(k)) / σ_g(k)
        where:
        - μ(x,k) is the mean of the Gaussian for cluster k given input x
        - μ_g(k) is the global mean for cluster k
        - σ_g(k) is the global standard deviation for cluster k
        """
        with torch.no_grad():
            batch_size = x.size(0)
            dummy_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            # Get GMVAE outputs
            pi_x, mu_zs, logvar_zs, _, mu_genz, logvar_genz, _, _, _, _, _, _ = self.gmvae(x, dummy_labels)
            
            # Reshape to (batch, K, z_dim)
            mu_zs = mu_zs.permute(0, 2, 1)  # (batch, K, z_dim)
            mu_genz = mu_genz.permute(0, 2, 1)  # (batch, K, z_dim)
            logvar_genz = logvar_genz.permute(0, 2, 1)  # (batch, K, z_dim)
            
            # Calculate z-scores
            sigma_genz = torch.sqrt(torch.exp(logvar_genz))
            z_scores_full = (mu_zs - mu_genz) / (sigma_genz + 1e-8)
            z_scores = z_scores_full.mean(dim=2)  # Average over z_dim to get (batch, K)
            
        return z_scores
    
    # REMOVED: ProtoCell4P/src/model.py lines 98-103 (encode method)
    # Original encode method:
    # def encode(self, x):
    #     h_e = self.activate(self.enc_i(x))
    #     for i in range(self.n_layers - 1):
    #         h_e = self.activate(self.enc_h[i](h_e))
    #     z = self.activate(self.enc_z(h_e))
    #     return z
    
    # REMOVED: ProtoCell4P/src/model.py lines 105-110 (decode method)
    # Original decode method:
    # def decode(self, z):
    #     h_d = self.activate(self.dec_z(z))
    #     for i in range(self.n_layers - 1):
    #         h_d = self.activate(self.dec_h[i](h_d))
    #     x_hat = torch.relu(self.dec_i(h_d))
    #     return x_hat
    
    # SOURCE: ProtoCell4P/src/model.py lines 112-117
    # Importance computation is EXACT copy
    def compute_importance(self, x):
        h_i = self.activate(self.imp_i(x))
        for i in range(self.n_layers - 1):
            h_i = self.activate(self.imp_h[i](h_i))
        import_scores = torch.sigmoid(self.imp_p(h_i)).reshape(-1, self.n_proto, self.n_classes)
        return import_scores
    
    # REMOVED: ProtoCell4P/src/model.py lines 119-160 (pretrain method)
    # Original pretrain method for autoencoder pretraining - not needed with GMVAE
    
    # REMOVED: ProtoCell4P/src/model.py lines 162-220 (prototype loss methods)
    # Original methods: compute_proto_loss, compute_sep_loss, compute_divers_loss
    # Not needed with z-score approach