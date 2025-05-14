import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used: ", device)
print("\n")
print("The training will begin now.\n")


i_s = 64
batch_size = 128
nz = 100
num_epochs = 80

lr = 0.00005  
b1 = 0.5

transform = transforms.Compose([
    transforms.Resize(i_s),
    transforms.CenterCrop(i_s),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])


class FlatImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

dataset_path = "/kaggle/input/animefacedataset/images"
dataset = FlatImageDataset(dataset_path, transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

print(f"Using {len(dataset)} images.")

def wit(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(wit)
discriminator.apply(wit)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, 0.999))
optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, 0.999))

fixed_noise = torch.randn(5, nz, 1, 1, device=device)

for epoch in range(1, num_epochs + 1):
    for i, real_images in enumerate(dataloader):
        b_size = real_images.size(0)
        real_images = real_images.to(device)        
        real_labels = torch.full((b_size,), 0.9, device=device)
        fake_labels = torch.zeros(b_size, device=device)        
        real_images += 0.05 * torch.randn_like(real_images)        
        discriminator.zero_grad()
        output_real = discriminator(real_images).view(-1)
        lossD_real = criterion(output_real, real_labels)
        lossD_real.backward()
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach()).view(-1)
        lossD_fake = criterion(output_fake, fake_labels)
        lossD_fake.backward()
        optimizerD.step()

        
        for _ in range(2):  
            generator.zero_grad()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = generator(noise)
            output = discriminator(fake_images).view(-1)
            lossG = criterion(output, real_labels)
            lossG.backward()
            optimizerG.step()

        if (i+1) % 100 == 0:
            print(f"(Epoch {epoch} out of {num_epochs}), (Batch {i+1}/{len(dataloader)}), Discriminator Loss -> {lossD_real.item()+lossD_fake.item():.4f}, Generator Loss -> {lossG.item():.4f}")

    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake, nrow=5, normalize=True)
        plt.figure(figsize=(10, 2))
        plt.axis("off")
        plt.title(f"After epoch {epoch}")
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

torch.save(generator.state_dict(), "generator.pth")