import torch
import torchaudio
from model import CNNAudioModel
from custom_audio_dataset import CustomDataset
from model import AUDIO_DIR, ANNOTATIONS_FILE_TEST, SAMPLE_RATE, NUM_SAMPLES


class_mapping = [
    'no_monkey_word',
    'contain_monkey_word'
]

def predict_single(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

def get_accuracy(model, test_data):
    model.eval()
    total_acc_val = 0
    for sample in test_data:
        input, target = sample[0], sample[1]
        input.unsqueeze_(0)
        with torch.no_grad():
            predictions = model(input)
            predicted_index = predictions[0].argmax(0)
            total_acc_val += (predicted_index == target).sum().item()
    
    return total_acc_val/len(test_data)

if __name__ == '__main__':
    cnn = CNNAudioModel()
    state_dict = torch.load('audio_model.pth')
    cnn.load_state_dict(state_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    test_samples = CustomDataset(ANNOTATIONS_FILE_TEST,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            'cpu')


    input, target = test_samples[0][0], test_samples[0][1]
    input.unsqueeze_(0)
    predicted, expected = predict_single(cnn, input, target, class_mapping)
    acc = get_accuracy(cnn, test_samples)
    print(f'Predicted: {predicted}, expected: {expected}')
    print(f'Metric evaluated: {acc}')