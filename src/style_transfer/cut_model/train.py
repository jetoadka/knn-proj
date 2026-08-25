import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import wandb
from util.util import tensor2im

torch.backends.cudnn.benchmark = True

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        print("Not using distributed mode")
        return False, 0, 0, 1

    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    dist.barrier()
    return True, gpu, rank, world_size

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    
    # --- DDP Initialization ---
    is_distributed, gpu_id, rank, world_size = init_distributed_mode()
    
    # Override gpu_ids based on DDP environment
    if is_distributed:
        opt.gpu_ids = [gpu_id]
        opt.is_distributed = True
    else:
        opt.is_distributed = False

    # Initialize W&B only on the main process (rank 0)
    if rank == 0:
        wandb.init(
            entity="knn-proj",
            project="style-transfer",
            name="contrastive-CUT-50M-param",
            config=vars(opt)
        )

    # --- Dataset Creation ---
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.

    # --- Model Creation ---
    model = create_model(opt)      # create a model given opt.model and other options
    if rank == 0:
        print('The number of training images = %d' % dataset_size)

    # Only visualizer on main process
    visualizer = Visualizer(opt) if rank == 0 else None
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations
    optimize_time = 0.1

    # Ensure all processes start training at the same time
    if is_distributed:
        dist.barrier()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        if rank == 0:
            visualizer.reset()

        # IMPORTANT: Set epoch for distributed sampler to ensure shuffling works across epochs
        dataset.set_epoch(epoch)
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            
            # Since data is split across GPUs, effective batch size per iteration is higher
            batch_size = data["A"].size(0) 
            total_iters += batch_size * world_size 
            epoch_iter += batch_size * world_size

            if is_distributed:
                torch.cuda.synchronize()
                
            optimize_start_time = time.time()
            
            # Initialization step
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               
                model.parallelize() # will be redefined in BaseModel

            model.set_input(data) 
            model.optimize_parameters()
            
            if is_distributed:
                torch.cuda.synchronize()
                
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            # --- Logging (Only on Rank 0) ---
            if rank == 0:
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                    wandb.log(model.get_current_losses())
                    
                if total_iters % opt.display_freq == 0:
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                    
                    visuals = model.get_current_visuals()
                    wandb.log({
                        "1_Realna_Cista_Tvar": wandb.Image(tensor2im(visuals['real_A'])),
                        "2_Nasa_Vygenerovana_Novina": wandb.Image(tensor2im(visuals['fake_B'])),
                        "3_Vzorova_Realna_Novina": wandb.Image(tensor2im(visuals['real_B']))
                    })

                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                    if opt.display_id is None or opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)


            iter_data_time = time.time()

            if total_iters % opt.save_latest_freq == 0:
                if rank == 0:
                    print(f'saving the latest model (epoch {epoch}, total_iters {total_iters})')
                    for name in model.model_names:
                        net = getattr(model, 'net' + name)
                        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                            net = net.module
                        
                        save_filename = f'iter_{total_iters}_net_{name}.pth'
                        save_path = os.path.join(model.save_dir, save_filename)
                        torch.save(net.state_dict(), save_path)
                        
                        latest_filename = f'latest_net_{name}.pth'
                        latest_path = os.path.join(model.save_dir, latest_filename)
                        torch.save(net.state_dict(), latest_path)

                if is_distributed:
                    torch.distributed.barrier()


        # In the end of each epoch
        if epoch % opt.save_epoch_freq == 0:
            # Logging, only rank 0
            if int(os.environ.get("RANK", 0)) == 0:
                print(f'saving the model at the end of epoch {epoch}, iters {total_iters}')
                
                # Saving via real underlying models (for saving clean weights)
                for name in model.model_names:
                    net = getattr(model, 'net' + name)
                    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        net = net.module
                    save_filename = f'{epoch}_net_{name}.pth'
                    save_path = os.path.join(model.save_dir, save_filename)
                    torch.save(net.state_dict(), save_path)
                    
                # Saving latest
                for name in model.model_names:
                    net = getattr(model, 'net' + name)
                    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        net = net.module
                    save_filename = f'latest_net_{name}.pth'
                    save_path = os.path.join(model.save_dir, save_filename)
                    torch.save(net.state_dict(), save_path)

            # 2. Bezpečné počkanie pre ostatné karty bez deadlocku:
            # Save waiting, to not cause deadlock
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if rank == 0:
                print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
                
            model.update_learning_rate()