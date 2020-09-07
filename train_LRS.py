import argparse
import os
import torch
import model.model as module_arch
from parse_config import ConfigParser
from trainer.trainer import Trainer
from torch.utils.data.dataloader import default_collate
from data_loader.datasets import DatasetV
from torch.utils.data import DataLoader

def collate_fn(batch):
  if True:
    return batch
  return default_collate(batch)

def main(config):
    logger = config.get_logger('train')
    num_words = config["dataset"]["args"]["num_words"]
    num_phoneme_thr = config["dataset"]["args"]["num_phoneme_thr"]
    split = config["dataset"]["args"]["split"]
    cmu_dict_path = config["dataset"]["args"]["cmu_dict_path"]
    data_struct_path = config["dataset"]["args"]["data_struct_path"]
    p_field_path = config["dataset"]["args"]["field_vocab_paths"]["phonemes"]
    g_field_path = config["dataset"]["args"]["field_vocab_paths"]["graphemes"]
    vis_feat_dir = config["dataset"]["args"]["vis_feat_dir"]
    batch_size = config["data_loader"]["args"]["batch_size"]
    shuffle = config["data_loader"]["args"]["shuffle"]
    drop_last = config["data_loader"]["args"]["drop_last"]
    pin_memory = config["data_loader"]["args"]["pin_memory"]
    num_workers = config["data_loader"]["args"]["num_workers"]

    train_dataset = DatasetV(num_words, num_phoneme_thr, cmu_dict_path,
        vis_feat_dir, "train", data_struct_path, p_field_path, g_field_path, True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True, collate_fn = collate_fn) 
   
    val_dataset = DatasetV(num_words, num_phoneme_thr, cmu_dict_path,
        vis_feat_dir, "val", data_struct_path, p_field_path, g_field_path, False)

    val_dataloader =torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory, shuffle=False, drop_last=True, collate_fn = collate_fn) 

    model = config.init('arch', module_arch)
    logger.info(model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init('optimizer', torch.optim, trainable_params) 
    lr_scheduler = config.init('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(model, optimizer,
                      config=config,
                      lr_scheduler=lr_scheduler,
                      num_words = num_words,
                      logger = logger,
                      train_dataset = train_dataset,
                      train_dataloader = train_dataloader,
                      val_dataset = val_dataset,
                      val_dataloader = val_dataloader,
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', required=True, help='config file path (default: None)')
    args.add_argument('--resume', help='path to latest checkpoint (default: None)')
    args.add_argument('--device', type=str, help="indices of GPUs to enable")
    args.add_argument('--mini_train', action="store_true")
    args.add_argument('--disable_workers', action="store_true")
    args.add_argument('--train_single_epoch', action="store_true")
    args.add_argument('--seeds', default="0", help="comma separated list of seeds")
    args.add_argument("--dbg", default="ipdb.set_trace")
    args.add_argument('--purge_exp_dir', action="store_true",
                      help="remove all previous experiments with the given config")
    args = ConfigParser(args)
    os.environ["PYTHONBREAKPOINT"] = args._args.dbg

    if args._args.disable_workers:
        print("Disabling data loader workers....")
        args["data_loader"]["args"]["num_workers"] = 0

    if args._args.train_single_epoch:
        print("Restring training to a single epoch....")
        args["trainer"]["epochs"] = 1
        args["trainer"]["save_period"] = 1
        args["trainer"]["skip_first_n_saves"] = 0

    msg = (f"Expected the number of training epochs ({args['trainer']['epochs']})"
           f"to exceed the save period ({args['trainer']['save_period']}), otherwise"
           " no checkpoints will be saved.")
    assert args["trainer"]["epochs"] >= args["trainer"]["save_period"], msg

    print("Launching experiment with config:")
    print(args)
    main(config=args)

