import argparse
import os, time, yaml
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from utils import *
from get_instances import *

def setup(args):
    config_path = args.config
    with open(config_path, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)
    device = 'cuda'

    #read configs =================================
    
    epochs = configs['epochs']

    dataset_name = configs['dataset_name']
    dataset_params = configs['dataset_params']
    val_data = configs['val_data']
    phases = ['train', 'val'] if val_data else ['train']

    batch_size = configs['batch_size']

    model_name = configs['model_name']
    model_params = configs.get('model_params', {}) 
    model_params['cascades'] = args.cascades
    
    restore_weights = configs['restore_weights']
    resume_model = configs['resume_model']

    loss_image_name = configs['loss_image_name']
    loss_edge_name = configs['loss_edge_name']
    score_names = configs['score_names']
    optim_name = configs['optim_name']
    optim_params = configs.get('optim_parmas', {})
    scheduler_name = configs.get('scheduler_name', None)
    scheduler_params = configs.get('scheduler_params', {})

    config_name = configs['config_name']
   
    #dirs, logger, writers, saver =========================================
    workspace = os.path.join(args.workspace, config_name) 
    checkpoints_dir, log_dir = get_dirs(workspace, remake=False) 
    tensorboard_dir = os.path.join(args.tensorboard_dir, config_name) 
    logger = Logger(log_dir)
    writers = get_writers(tensorboard_dir, phases)
    saver = CheckpointSaver(checkpoints_dir)

    #dataloaders, model, loss f, score f, optimizer, scheduler================================
    dataloaders = get_loaders(dataset_name, dataset_params, batch_size, phases, model_name)
    model = get_model(model_name, model_params, device)
    loss_image_f = get_loss(loss_image_name)
    loss_edge_f = get_loss(loss_edge_name)
    score_fs = get_score_fs(score_names)
    val_score_name = score_names
    optim_params['params'] = model.parameters()
    optimizer, scheduler = get_optim_scheduler(optim_name, optim_params, scheduler_name, scheduler_params)

    #load weights ==========================================
    if resume_model:
        model = saver.load(restore_weights, model)
    
    start_epoch = 0

    return configs, device, epochs, start_epoch, phases, workspace, logger, writers, saver, dataloaders, model, loss_image_f, loss_edge_f, score_fs, val_score_name, optimizer, scheduler

def main(args):
    configs, device, epochs, start_epoch, phases, workspace, logger, writers, saver, \
        dataloaders, model, loss_image_f, loss_edge_f, score_fs, val_score_name, optimizer, scheduler = setup(args)


    logger.write('config path: ' + args.config)
    logger.write('workspace: ' + workspace)
    logger.write('description: ' + configs['description'])
    logger.write('\n')
    logger.write('train start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.write('-----------------------')

    start = time.time()
    if args.seed:
        set_seeds(args.seed)
    
    for epoch in range(start_epoch, epochs):
        for phase in phases: #['train', 'val'] or ['train']
            running_score = defaultdict(int)

            if phase == 'train': model.train()
            else: model.eval()

            for i, (x, y, csm, mask, coeffs_x0, coeffs_gt) in enumerate(tqdm(dataloaders[phase])):
                
                x, y, csm, mask, coeffs_x0, coeffs_gt = x.to(device), y.to(device), csm.to(device), mask.to(device), coeffs_x0.to(device), coeffs_gt.to(device)
    
                with torch.set_grad_enabled(phase=='train'):
                    
                    if args.exp == 'ueomri':

                        csm = c2r(csm, axis=-1).float() 
                        mask = mask.unsqueeze(1).unsqueeze(-1).float()
                        y_pred, y_pred_edge = model(x, mask, csm, coeffs_x0)

        
                    
                    loss_image = loss_image_f(y_pred, y) 
                    loss_edge = loss_edge_f(y_pred_edge, coeffs_gt)
                    
                    img_w = 1.0
                    edge_w = .1
                    loss = loss_image * img_w + loss_edge * edge_w

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    if configs['gradient_clip']:
                        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                    optimizer.step()

                running_score['loss'] += loss.item() * y_pred.size(0)
                running_score['loss_image'] += loss_image.item() * y_pred.size(0)
                running_score['loss_edge'] += loss_edge.item() * y_pred.size(0)

                # when calculating metric, based on abs of complex value
                y = np.abs(r2c(y.detach().cpu().numpy(), axis=1))
                y_pred = np.abs(r2c(y_pred.detach().cpu().numpy(), axis=1))
                coeffs_gt = coeffs_gt.detach().cpu()
                y_pred_edge = y_pred_edge.detach().cpu()

                for score_name, score_f in score_fs.items():
                    running_score[score_name] += score_f(y, y_pred) * y_pred.shape[0]

            #scheduler
            if phase == 'train' and scheduler:
                scheduler.step()
            
            #write log
            epoch_score = {score_name: score / len(dataloaders[phase].dataset) for score_name, score in running_score.items()}
            for score_name, score in epoch_score.items():
                writers[phase].add_scalar(score_name, score, epoch)
            
            mask = mask.squeeze(1).squeeze(-1)
            writers[phase].add_figure('image', display_img(np.abs(r2c(x[-1].detach().cpu().numpy())), mask[-1].detach().cpu().numpy(), \
                y[-1], y_pred[-1], epoch_score[val_score_name[0]]), epoch)
                
            writers[phase].add_figure('edge', display_edge(y_pred_edge[0,0], y_pred_edge[0,1], y_pred_edge[0,2], y_pred_edge[0,3], \
                y_pred_edge[0,4], y_pred_edge[0,5],coeffs_gt[0, 0], coeffs_gt[0, 1], coeffs_gt[0, 2], coeffs_gt[0, 3], coeffs_gt[0, 4], coeffs_gt[0, 5]), epoch)

            logger.write('epoch {}/{} {} {}:{:.4f}\t{}:{:.4f}\tloss:{:.4f}\tloss_image:{:.4f}\tloss_edge:{:.4f}'.format(epoch, epochs, phase, val_score_name[0], epoch_score[val_score_name[0]], \
                 val_score_name[1], epoch_score[val_score_name[1]], epoch_score['loss'], epoch_score['loss_image'] * img_w, epoch_score['loss_edge'] * edge_w))

        #save model
        if phase == 'val':
            saver.save_checkpoints(model, epoch_score[val_score_name[0]])

    for phase in phases:
        writers[phase].close()
        
    logger.write('-----------------------')
    logger.write('total train time: {:.2f} min'.format((time.time()-start)/60))
    logger.write('best score: {:.4f}'.format(saver.best_score))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=False,help="config file path")
    parser.add_argument("--exp", type=str)
    parser.add_argument("--workspace", type=str)
    parser.add_argument("--tensorboard_dir", type=str)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--write_lr", type=bool, default=False)
    parser.add_argument("--write_image", type=int, default=0)
    parser.add_argument("--cascades", type=int, default=7)
    parser.add_argument("--write_lambda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)
    

    args = parser.parse_args()

    main(args)
