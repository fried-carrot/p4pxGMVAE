#!/usr/bin/env python
# copy of ProtoCell4P/src/model.py with changes marked

import torch
import torch.nn as nn
import sys

# ADDITION: import gmvae
sys.path.append('bulk2sc_GMVAE/_model_source_codes')
from GMVAE_scrna_zinb import GMVAE_ZINB

# ADDITION: gmvae args class
class GMVAEArgs:
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

class ProtoCell(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, n_layers, n_proto, n_classes, lambdas, n_ct=None, device="cpu", d_min=1):
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

        # REMOVED: lines 28-34 (encoder and decoder networks)
        # self.enc_i = nn.Linear(self.input_dim, self.h_dim)
        # self.enc_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        # self.enc_z = nn.Linear(self.h_dim, self.z_dim)
        # self.dec_z = nn.Linear(self.z_dim, self.h_dim)
        # self.dec_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        # self.dec_i = nn.Linear(self.h_dim, self.input_dim)
        
        # ADDITION: gmvae instead of encoder/decoder
        gmvae_args = GMVAEArgs(input_dim, n_proto, device)
        self.gmvae = GMVAE_ZINB(gmvae_args).to(device)
        import os
        if os.path.exists('models/gmvae_model.pth'):
            self.gmvae.load_state_dict(torch.load('models/gmvae_model.pth', map_location=device))
        for param in self.gmvae.parameters():
            param.requires_grad = False

        self.imp_i = nn.Linear(self.input_dim, self.h_dim)
        self.imp_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.imp_p = nn.Linear(self.h_dim, self.n_proto * self.n_classes)
        
        # ADDITION: classifier layer from P4P
        self.clf = nn.Linear(self.n_proto, self.n_classes, bias=False)

        self.activate = nn.LeakyReLU()

        # REMOVED: line 42 (prototypes)
        # self.prototypes = nn.parameter.Parameter(torch.empty(self.n_proto, self.z_dim), requires_grad = True)

        self.ce_ = nn.CrossEntropyLoss(reduction="mean")

        # REMOVED: lines 46-49, 52 (encoder/decoder/prototype init)
        # nn.init.xavier_normal_(self.enc_i.weight)
        # nn.init.xavier_normal_(self.enc_z.weight)
        # nn.init.xavier_normal_(self.dec_z.weight)
        # nn.init.xavier_normal_(self.dec_i.weight)
        nn.init.xavier_normal_(self.imp_i.weight)
        nn.init.xavier_normal_(self.imp_p.weight)
        nn.init.xavier_normal_(self.clf.weight)
        # nn.init.xavier_normal_(self.prototypes)
        for i in range(self.n_layers - 1):
            # REMOVED: lines 54-55 (encoder/decoder init)
            # nn.init.xavier_normal_(self.enc_h[i].weight)
            # nn.init.xavier_normal_(self.dec_h[i].weight)
            nn.init.xavier_normal_(self.imp_h[i].weight)

        if self.n_ct is not None:
            self.ct_clf1 = nn.Linear(self.n_proto, self.n_ct)
            self.ct_clf2 = nn.Linear(self.n_proto * self.n_classes, self.n_ct)
            nn.init.xavier_normal_(self.ct_clf1.weight)
            nn.init.xavier_normal_(self.ct_clf2.weight)

    def forward(self, x, y, ct=None, sparse=True):
        
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)

        # CHANGED: line 76 - replace encode with compute_zscores
        # z = self.encode(x)
        z_scores = self.compute_zscores(x)
        
        import_scores = self.compute_importance(x) # (n_cell, n_proto, n_class)
        
        # CHANGED: lines 80-81 - use z_scores as similarity instead of computing distances
        # c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        # c_logits = (1 / (c2p_dists+0.5))[:,None,:].matmul(import_scores).squeeze(1) # (n_cell, n_classes)
        c_logits = z_scores[:,None,:].matmul(import_scores).squeeze(1) # (n_cell, n_classes)
        
        # aggregate c_logits to patient level (as in original P4P)
        logits = torch.stack([c_logits[split_idx[i]:split_idx[i+1]].mean(dim=0) for i in range(len(split_idx)-1)])

        clf_loss = self.ce_(logits, y)

        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf2(import_scores.reshape(-1, self.n_proto * self.n_classes))
            ct_loss = self.ce_(ct_logits, torch.tensor([j for i in ct for j in i]).to(self.device))
        else:
            ct_loss = 0

        total_loss = clf_loss + self.lambda_6 * ct_loss

        if ct is not None:
            return total_loss, logits, ct_logits    
        return total_loss, logits
    
    # ADDITION: compute_zscores method
    def compute_zscores(self, x):
        with torch.no_grad():
            batch_size = x.size(0)
            dummy_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            pi_x, mu_zs, logvar_zs, _, mu_genz, logvar_genz, _, _, _, _, _, _ = self.gmvae(x, dummy_labels)
            
            mu_zs = mu_zs.permute(0, 2, 1)  # (batch, K, z_dim)
            mu_genz = mu_genz.permute(0, 2, 1)  # (batch, K, z_dim)
            logvar_genz = logvar_genz.permute(0, 2, 1)  # (batch, K, z_dim)
            
            sigma_genz = torch.sqrt(torch.exp(logvar_genz))
            z_scores_full = (mu_zs - mu_genz) / (sigma_genz + 1e-8)
            z_scores = z_scores_full.mean(dim=2)  # (batch, K)
            
        return z_scores
    
    # REMOVED: encode method (lines 98-103)
    # def encode(self, x):
    #     h_e = self.activate(self.enc_i(x))
    #     for i in range(self.n_layers - 1):
    #         h_e = self.activate(self.enc_h[i](h_e))
    #     z = self.activate(self.enc_z(h_e))
    #     return z

    # REMOVED: decode method (lines 105-110)
    # def decode(self, z):
    #     h_d = self.activate(self.dec_z(z))
    #     for i in range(self.n_layers - 1):
    #         h_d = self.activate(self.dec_h[i](h_d))
    #     x_hat = torch.relu(self.dec_i(h_d))
    #     return x_hat

    def compute_importance(self, x):
        h_i = self.activate(self.imp_i(x))
        for i in range(self.n_layers - 1):
            h_i = self.activate(self.imp_h[i](h_i))
        import_scores = torch.sigmoid(self.imp_p(h_i)).reshape(-1, self.n_proto, self.n_classes)
        return import_scores
    
    # REMOVED: entire pretrain method (lines 119-160) - no longer needed
