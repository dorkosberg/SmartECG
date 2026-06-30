import torch
from models import cnn1D


def generate_model(opt):
    """Build the general 1D CNN classifier.

    Patient-specific fine-tuning and frozen-layer transfer learning were removed.
    One shared model is trained from scratch for all patients.
    """
    assert opt.mode in ['general-training', 'test']

    model = cnn1D.cnn_1D(
        num_blocks=opt.num_blocks,
        block_channels=opt.block_channels,
        kernel_size=opt.kernel_size,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    return model, model.parameters()
