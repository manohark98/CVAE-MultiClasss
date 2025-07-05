import os
import argparse
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm might not be available
    def tqdm(iterable, *args, **kwargs):
        return iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image


class CVAE(nn.Module):
    def __init__(self, img_channels: int, num_classes: int, latent_dim: int, embedding_dim: int = 16):
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),            # 64x16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),           # 128x8x8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),          # 256x4x4
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4 + embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4 + embedding_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + embedding_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128x8x8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64x16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # Cx64x64
            nn.Sigmoid(),
        )

    def encode(self, x, y):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        y_emb = self.embedding(y)
        h = torch.cat([h, y_emb], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.embedding(y)
        z = torch.cat([z, y_emb], dim=1)
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)


def get_dataloaders(data_dir, img_size=64, batch_size=64, test_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    num_test = int(len(dataset) * test_split)
    num_train = len(dataset) - num_test
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, len(dataset.classes)


def train(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(dataloader, desc='Training', leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x, y)
        loss = loss_function(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(dataloader.dataset)


def test(model, dataloader, device, epoch, output_dir):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dataloader, desc='Testing', leave=False)):
            x = x.to(device)
            y = y.to(device)
            recon, mu, logvar = model(x, y)
            loss = loss_function(recon, x, mu, logvar)
            running_loss += loss.item() * x.size(0)
            if i == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n], recon[:n]])
                save_image(comparison.cpu(), os.path.join(output_dir, f'reconstruction_{epoch}.png'), nrow=n)
    return running_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description='Conditional VAE for multiple classes')
    parser.add_argument('--data-dir', default='dataset', type=str, help='Path to dataset root folder')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs to train')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--latent-dim', default=128, type=int, help='Latent dimension')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--output-dir', default='outputs', type=str, help='Directory to save reconstructions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader, num_classes = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    model = CVAE(img_channels=3, num_classes=num_classes, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        test_loss = test(model, test_loader, device, epoch, args.output_dir)
        print(f'Epoch {epoch}: train loss {train_loss:.4f}, test loss {test_loss:.4f}')
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'cvae_epoch_{epoch}.pth'))


if __name__ == '__main__':
    main()
