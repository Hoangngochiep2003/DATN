import torch.nn as nn


def l1(output, target):
    return torch.mean(torch.abs(output - target))


def l1_wav(output_dict, target_dict):
	return l1(output_dict['segment'], target_dict['segment'])


def get_loss_function(loss_type):
    if loss_type == 'l1_loss':
        return nn.L1Loss()
    elif loss_type == 'mse_loss':
        return nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss type '{loss_type}' is not implemented!")
