import hydra
import torch
from tqdm import tqdm
from hydra.utils import instantiate
from accelerate.logging import get_logger
from safetensors.torch import load_file

from model import fetch_model
from dataset import fetch_dataloader
from util.padder import InputPadder


@hydra.main(version_base=None, config_path='config', config_name='evaluate_stereo')
def main(cfg):
    logger = get_logger(__name__)
    accelerator = instantiate(cfg.accelerator)

    # -------------------------
    # Dataset
    # -------------------------
    dataloader = fetch_dataloader(cfg, cfg.dataset, cfg.dataloader, logger)

    # -------------------------
    # Model
    # -------------------------
    model = fetch_model(cfg, logger)

    logger.info(f'Loading checkpoint from {cfg.checkpoint}')
    state_dict = load_file(cfg.checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model = accelerator.prepare_model(model)

    for name in dataloader:
        dataloader[name] = accelerator.prepare_data_loader(dataloader[name])

    # -------------------------
    # Evaluation
    # -------------------------
    for name in dataloader:
        model.eval()
        total_elem = 0
        total_epe = 0.0
        total_out = 0.0

        outlier = cfg.dataset[name].outlier

        for data in tqdm(
            dataloader[name],
            dynamic_ncols=True,
            disable=not accelerator.is_main_process
        ):
            left, right, disp_gt, valid = data

            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)

            with torch.no_grad():
                if cfg.model.name == 'RAFTStereo':
                    _, disp_pred = model(
                        left, right,
                        iters=cfg.model.valid_iters,
                        test_mode=True
                    )
                    disp_pred = -disp_pred
                elif cfg.model.name == 'IGEVStereo':
                    disp_pred = model(
                        left, right,
                        iters=cfg.model.valid_iters,
                        test_mode=True
                    )
                else:
                    raise ValueError(f'Invalid model name: {cfg.model.name}')

            disp_pred = padder.unpad(disp_pred)
            assert disp_pred.shape == disp_gt.shape

            # -------------------------
            # CORRECT METRIC COMPUTATION
            # -------------------------
            epe_map = torch.abs(disp_pred - disp_gt)
            valid_mask = valid >= 0.5

            epe_valid = epe_map[valid_mask]
            out_valid = (epe_valid > outlier).float()

            epe_valid, out_valid = accelerator.gather_for_metrics(
                (epe_valid, out_valid)
            )

            total_elem += epe_valid.numel()
            total_epe += epe_valid.sum().item()
            total_out += out_valid.sum().item()

        accelerator.print(
            f'{name}/EPE: {total_epe / total_elem:.2f}, '
            f'{name}/Bad {outlier}px: {100 * total_out / total_elem:.2f}'
        )

    accelerator.end_training()


if __name__ == '__main__':
    main()
