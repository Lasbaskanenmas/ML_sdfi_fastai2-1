import torch.nn as nn




#a simple conv classifier
class CNN(nn.Module):
    def __init__(self,n_classes,n_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output n_classes classes
        self.out = nn.Linear(32 * 7 * 7, n_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output   # return x for visualization


def create_custom_model(arch=None, n_out=10, img_size=None, pretrained=False, cut=None, n_in=3, **kwargs):
    "Create custom unet architecture for use in fastai"
    #meta = model_meta.get(arch, _default_meta)
    #body = create_body(arch, n_in, pretrained, ifnone(cut, meta['cut']))
    #model = CustomUnet(body, n_out, img_size, **kwargs)
    model = CNN(n_classes=n_out,n_channels=n_in)
    return model
