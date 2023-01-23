import os
import numpy as np
import torch

from detectron2.utils.logger import setup_logger
setup_logger()
import detectron2.utils.comm as comm
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import default_writers
from detectron2.data import build_detection_train_loader
from detectron2.utils.events import EventStorage
from eval import do_test
from logger import logger
from config import get_model_config


def train_model(cfg):

    PATIENCE = 500
    BEST_LOSS = np.inf 

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = ( checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1)
    prev_iter = start_iter
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    data_loader = build_detection_train_loader(cfg)

    logger.info("Starting training from iteration {}".format(start_iter))

    patience_counter = 0

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
            
            if iteration > prev_iter:
                prev_iter = iteration
                if losses_reduced < BEST_LOSS:
                    BEST_LOSS = losses_reduced
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter % 100 == 0:
                        print(f"Loss has not improved for {patience_counter} iterations")
                    if patience_counter >= PATIENCE:
                        print(f"EARLY STOPPING")
                        break
    return model

def main():
    cfg = get_model_config()
    trained_model = train_model(cfg)
    do_test(cfg, trained_model)

if __name__ == "__main__":
    main()                
