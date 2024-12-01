import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# DEFINING Discriminator AND Generator

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256), 
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim), # img_dim will 784 for MNIST dataset (28*28*1 --> 784)
            nn.Tanh(), # we used tanh here to compress outputs b/w -1 and 1 (same as inputs)
        )

    def forward(self, x):
        return self.gen(x)
    
# DEFINING HYPERPARAMETERS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4
noise_dim = 64
img_dim = 28 * 28 * 1 # 784
batch_size = 50
num_epochs = 1

# INITIALIZATION

disc = Discriminator(img_dim).to(device)
gen = Generator(noise_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, noise_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root='./GAN/data', transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        batch_size = real.shape[0]
        real = real.view(-1, img_dim).to(device)

        ### Training Discriminator max log(D(real)) + log(1 - D(fake)) (fake = G(z))
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake.detach()).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        ### Training Generator max log(D(fake)) <--> min log(1 - D(fake)) (fake = G(z))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

print('Working Fine Till Here')