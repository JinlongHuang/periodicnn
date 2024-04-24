from icecream import ic
import torch

if __name__ == '__main__':
    weights = dict()
    for i in range(1, 10):
        checkpoint_file = f'checkpoints/seed_{i}_epoch_200.pt'
        model = torch.load(checkpoint_file)['model_state_dict']
        weights[f'seed_{i}'] = model['weight'].squeeze().sign()

    for i in range(1, 10):
        for j in range(i+1, 10):
            print(torch.sum(
                weights[f'seed_{i}'] == weights[f'seed_{j}']).item())
