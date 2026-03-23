import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WeatherLSTM(nn.Module):
    """LSTM-based weather forecaster with regression + classification heads.

    Concepts: Ch.3 (predictor as composition of primitives), Ch.8 (classification via softmax),
    Ch.9 (regression via predicted distribution parameters).
    """

    def __init__(self, n_features=7, hidden_dim=128, n_layers=2, horizon=6, n_classes=6, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features

        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.reg_head = nn.Linear(hidden_dim, horizon * n_features)
        self.cls_head = nn.Linear(hidden_dim, horizon * n_classes)

        self.n_classes = n_classes

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden_dim)

        reg = self.reg_head(last)  # (batch, horizon * n_features)
        reg = reg.view(-1, self.horizon, self.n_features)

        cls = self.cls_head(last)  # (batch, horizon * n_classes)
        cls = cls.view(-1, self.horizon, self.n_classes)

        return reg, cls


class WeatherTransformer(nn.Module):
    """Transformer-based weather forecaster. CPU-friendly: 2 layers, 4 heads.

    Concepts: Ch.3 (DAG of differentiable primitives, chain rule / backprop).
    """

    def __init__(self, n_features=7, d_model=64, n_heads=4, n_layers=2,
                 horizon=6, n_classes=6, dropout=0.2, max_len=168):
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.reg_head = nn.Linear(d_model, horizon * n_features)
        self.cls_head = nn.Linear(d_model, horizon * n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        last = x[:, -1, :]

        reg = self.reg_head(last).view(-1, self.horizon, self.n_features)
        cls = self.cls_head(last).view(-1, self.horizon, self.n_classes)
        return reg, cls


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class WeatherAutoencoder(nn.Module):
    """Denoising autoencoder for city weather profiles (96-dim).

    Concepts: Ch.6 (encoder-decoder with corruption for regularization).
    """

    def __init__(self, input_dim=96, latent_dim=8, noise_frac=0.2):
        super().__init__()
        self.noise_frac = noise_frac

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def corrupt(self, x):
        """Masking corruption (Ch.6, Eq. 6.15): randomly zero out features."""
        if self.training and self.noise_frac > 0:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.noise_frac))
            return x * mask
        return x

    def encode(self, x):
        return self.encoder(self.corrupt(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class MoGHead(nn.Module):
    """Mixture of Gaussians output head for regression with uncertainty.

    Concepts: Ch.9 (Eq. 9.7-9.8, mixture density network).
    Predicts K components, each with (mu, log_sigma, log_pi).
    """

    def __init__(self, input_dim, output_dim=1, n_components=3):
        super().__init__()
        self.n_components = n_components
        self.output_dim = output_dim
        # For each component: mu, log_sigma, log_pi
        self.head = nn.Linear(input_dim, n_components * (output_dim + output_dim + 1))

    def forward(self, x):
        params = self.head(x)
        K = self.n_components
        D = self.output_dim

        mu = params[:, :K * D].view(-1, K, D)
        log_sigma = params[:, K * D:2 * K * D].view(-1, K, D)
        log_pi = params[:, 2 * K * D:]  # (batch, K)
        log_pi = F.log_softmax(log_pi, dim=-1)

        return mu, log_sigma, log_pi

    @staticmethod
    def nll_loss(y, mu, log_sigma, log_pi):
        """Negative log-likelihood under MoG (Ch.9, Eq. 9.8).

        Uses log-sum-exp trick (Ch.9, Eq. 9.11-9.12) for numerical stability.
        """
        # y: (batch, D), mu: (batch, K, D), log_sigma: (batch, K, D), log_pi: (batch, K)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        y = y.unsqueeze(1)  # (batch, 1, D)

        sigma = torch.exp(log_sigma)
        log_probs = -0.5 * (((y - mu) / (sigma + 1e-8)) ** 2 + 2 * log_sigma + math.log(2 * math.pi))
        log_probs = log_probs.sum(dim=-1)  # (batch, K)
        log_probs = log_probs + log_pi  # (batch, K)

        loss = -torch.logsumexp(log_probs, dim=-1)  # (batch,)
        return loss.mean()


class QuantileHead(nn.Module):
    """Quantile regression head with pinball loss.

    Concepts: Ch.9 (Eq. 9.21, asymmetric loss for quantile prediction).
    """

    def __init__(self, input_dim, output_dim=1, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        self.heads = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in quantiles])

    def forward(self, x):
        return torch.stack([h(x) for h in self.heads], dim=1)  # (batch, n_quantiles, output_dim)

    def pinball_loss(self, y, y_pred):
        """Pinball loss (Ch.9, Eq. 9.21)."""
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        y = y.unsqueeze(1)  # (batch, 1, D)

        losses = []
        for i, q in enumerate(self.quantiles):
            error = y - y_pred[:, i:i + 1, :]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        return sum(losses) / len(losses)
