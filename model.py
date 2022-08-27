import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from custom_audio_dataset import CustomDataset


BATCH_SIZE = 1
EPOCHS = 12
LEARNING_RATE = 0.001

ANNOTATIONS_FILE_TRAIN = 'dataset/audio_labels_train.csv'
ANNOTATIONS_FILE_TEST = 'dataset/audio_labels_test.csv'
AUDIO_DIR = 'dataset/audio/'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

class CNNAudioModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=265,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(265 * 9, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        prediction = model(input)
        loss = loss_fn(prediction, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f'loss: {loss.item()}')


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i+1}')
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print('---------------------------')
    print('Finished training')


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using {device}')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    train_samples = CustomDataset(ANNOTATIONS_FILE_TRAIN,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    train_dataloader = create_data_loader(train_samples, BATCH_SIZE)
    cnn = CNNAudioModel().to(device)
    print(cnn)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), 'audio_model.pth')