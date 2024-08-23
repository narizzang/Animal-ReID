import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import cv2

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True)
            print(text_feature.shape)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()
        print(text_features.shape)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                # score, feat, image_features, text_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                print("i_features 크기: ", image_features.shape, "t_features 크기: ", text_features.shape)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view, get_feat = True)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 3, 5]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view, get_feat = True)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 3, 5]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)
def add_border(image, color, border_size=5):
    """Adds a border around an image."""
    return cv2.copyMakeBorder(
        image, border_size, border_size, border_size, border_size, 
        cv2.BORDER_CONSTANT, value=color
    )
def visualize_ranked_results(distmat, img_tensors, img_path_list, pids, num_query, save_dir='.', topk=5):
    """Helper function to display query image and ranked images"""
    num_q, num_g = distmat.shape
    assert num_q == num_query
    indices = np.argsort(distmat, axis=1)
    print(img_path_list)

    for q_idx in range(num_q):
        query_image = img_tensors[q_idx].cpu().numpy().transpose(1, 2, 0)
        query_image = np.clip(query_image, 0, 1)
        query_image = (query_image * 255).astype(np.uint8)
        query_path = img_path_list[q_idx]
        query_image_real_path = os.path.join('/content/drive/MyDrive/data/Animals/query', query_path)
        # query_image_real_path = os.path.join('/content/drive/MyDrive/data/Market-1501-v15.09.15/query', query_path)
        query_image_real = cv2.imread(query_image_real_path)
        print(query_image_real_path)
        query_image_real_real = cv2.cvtColor(query_image_real, cv2.COLOR_BGR2RGB)
        
        query_pid = pids[q_idx]
        query_height, query_width = query_image_real_real.shape[:2]
        
        fig, axes = plt.subplots(1, topk + 1, figsize=(15, 5))

        axes[0].imshow(query_image_real_real)
        axes[0].imshow(add_border(query_image_real_real, (0, 0, 255), border_size=7))
        axes[0].set_title("Query")
        axes[0].axis('off')

        rank_idx = 1
        for g_idx in indices[q_idx, :topk]:
            rank_image = img_tensors[num_query + g_idx].cpu().numpy().transpose(1, 2, 0)
            rank_image = np.clip(rank_image, 0, 1)
            rank_path = img_path_list[num_query + g_idx]
            rank_image_real_path = os.path.join('/content/drive/MyDrive/data/Animals/gallery', rank_path)
            # rank_image_real_path = os.path.join('/content/drive/MyDrive/data/Market-1501-v15.09.15/bounding_box_test', rank_path)
            rank_image_real = cv2.imread(rank_image_real_path)
            rank_image_real_real = cv2.cvtColor(rank_image_real, cv2.COLOR_BGR2RGB)
            rank_image_resized = cv2.resize(rank_image_real_real, (query_width, query_height))
            rank_pid = pids[num_query + g_idx]

            if rank_pid == query_pid:
                border_color = (0, 255, 0)  # Blue for correct match
            else:
                border_color = (255, 0, 0)  # Red for incorrect match
            
            rank_image_with_border = add_border(rank_image_resized, border_color, border_size=7)
            axes[rank_idx].imshow(rank_image_with_border)
            axes[rank_idx].set_title(f"Rank {rank_idx}")
            axes[rank_idx].axis('off')
            
            rank_idx += 1
        
        save_path = os.path.join(save_dir, f'ranked_results_{q_idx}.jpg')
        plt.savefig(save_path)
    # plt.figure(figsize=(15, 5))
    
    # # Show query image
    # plt.subplot(1, 6, 1)
    # query_image = query_image.cpu().numpy().transpose(1, 2, 0)
    # plt.imshow(query_image)
    # plt.title("Query")
    # plt.axis('off')
    
    # # Show rank images
    # for i, (rank_image, rank_path) in enumerate(zip(rank_images, rank_paths)):
    #     plt.subplot(1, 6, i+2)
    #     rank_image = rank_image.cpu().numpy().transpose(1, 2, 0)
    #     plt.imshow(rank_image)
    #     plt.title(f"Rank {i+1}")
    #     plt.axis('off')
    # plt.savefig(f'output_{num}.png')
    # plt.show()
def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    features = []
    pids = []
    camids = []
    img_path_list = []
    img_tensors = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view, get_feat = True)
            evaluator.update((feat, pid, camid))
            features.append(feat)
            pids.extend(pid)
            # camids.extend(camid)
            img_path_list.extend(imgpath)
            img_tensors.extend(img)

    features = torch.cat(features, dim=0).cpu().numpy()
    dist_matrix = cosine_distances(features[:num_query], features[num_query:])
    save_dir = '/content/drive/MyDrive/output/figs'
    visualize_ranked_results(dist_matrix, img_tensors, img_path_list, pids, num_query, save_dir)
    # for i in range(5):
    #     query_image = img_tensors[i]
    #     query_path = img_path_list[i]
    #     print("이미지: ",query_image, "경로: ", query_path)
    #     # Get indices of the 5 smallest distances (most similar images)
    #     rank_indices = np.argsort(dist_matrix[i])[:5]
    #     rank_images = [img_tensors[num_query + idx] for idx in rank_indices]
    #     rank_paths = [img_path_list[num_query + idx] for idx in rank_indices]
    #     print("랭크 이미지: ",rank_images)
    #     print(rank_paths)
    #     # Show query image and ranked images
    #     show_images(query_image, rank_images, query_path, rank_paths, i)
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 3, 5]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]