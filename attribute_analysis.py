import os

import numpy as np
import PIL.Image
import PIL.ImageDraw, PIL.ImageFont
import torch
from torch.nn.functional import normalize


from stylegan.wrapper import Generator
from stylegan.embedding import get_delta_t
from stylegan.manipulator import Manipulator
from stylegan.mapper import get_delta_s, get_delta_s_top

import class_labels
import pandas as pd


from torchvision.models import resnet
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm


import clip
import time
import argparse

from scripts.histogram import histogram, filter_string
from scripts.draw_label import draw_label

exp_list = ['ffhq', 'afhqdog', 'afhqcat', 'church', 'car']
draw_logit = True


def process_experiment(args):
    exp = args.experiment 
    sg_path = {
        x : f'./pretrained/{x}.pkl' for x in exp_list
    }
    fs3_path = {
        x : f'./pretrained/fs3{x}.npy' for x in exp_list

    }

    neutral = {
        'ffhq': 'a face',
        'afhqcat': 'a cat',
        'church': 'a church',
        'car': 'a car',
        'afhqdog': 'a dog'
    }

    args.stylegan_path = sg_path[exp]
    args.fs3_path = fs3_path[exp]
    args.neutral = neutral[exp]
    return args


def findMaxS(style, dic):
    s_list = np.array([])
    for a in style:
        s_list= np.append(s_list, style[a].cpu().detach().numpy())
    
    top = max(s_list)
    
    ind = np.argmax(np.array(s_list))
    top_ind = np.where(np.abs(s_list) > 1.5)

    for dd in top_ind[0]:
        d = int(dd)
        if d in dic:

            dic[d] = dic[d] + 1
        else:
            dic[d] = 1

    dic_list = sorted(dic.items(), key=lambda item: item[1])
    dic_list.reverse()
    print({k: v for k, v in dic_list[:30]})
    return dic




def prepare_label(args, device , beta_threshold=0.6):
    
    if args.target_attr == None:
        labels, label_beta =  getattr(class_labels, args.experiment)()
    else:
        labels, label_beta =  args.target_attr.split(','), {}
    # breakpoint()

    labels = labels[:args.num_attr]

    neutral = args.neutral

    ckpt = args.stylegan_path
    G = Generator(ckpt, device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    fs3 = np.load(args.fs3_path)
    manipulator = Manipulator(G, device)

    avg = 0

    delta_s_dict = {}
    for target in labels:
        classnames=[neutral, target]
        delta_t = get_delta_t(classnames, model)

        if beta_threshold < 5:
            if target not in label_beta:
                delta_s, num_channel = get_delta_s(fs3, delta_t, manipulator, beta_threshold=beta_threshold)
            else:
                delta_s, num_channel = get_delta_s(fs3, delta_t, manipulator, beta_threshold=label_beta[target])

            d_s_array = convert(delta_s)
            d_s_array =  normalize(torch.tensor(d_s_array), dim = 0)
            delta_s = convert(d_s_array, delta_s)

        else:
            delta_s, num_channel = get_delta_s_top(fs3, delta_t, manipulator, num_top=int(beta_threshold))
            d_s_array = convert(delta_s)
            d_s_array =  normalize(torch.tensor(d_s_array), dim = 0)
            delta_s = convert(d_s_array, delta_s)

        print(f"{target}:{num_channel}")
        avg += num_channel

        delta_s_dict[target] = delta_s

    print(f"average channel is {avg/len(labels)}")

    return delta_s_dict, {a:torch.tensor([0.], device=device, requires_grad=True) for a in labels}
    



def convert(s,sample_dict = {}):

    if type(s) is dict:
        output = []
        for x in s:
            output += list(s[x].detach().cpu().numpy())
        return np.array(output)
    else:
        dic = dict()
        ind = 0
        for layer in sample_dict: # 26
            dim = sample_dict[layer].shape[-1]

            dic[layer] = s[ind:ind+dim].to(sample_dict[layer].device)
            ind += dim
        return dic



custom_transform_clip = transforms.Compose([transforms.Resize(size=224), 
                                    transforms.CenterCrop(size=(224, 224)),
                                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
custom_transform_cls = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

custom_transform_resize = transforms.Compose([transforms.Resize(size=224)])

def main(args):

    args = process_experiment(args)
    read_labels = getattr(class_labels, args.experiment)

    print(args)

    np.random.seed(args.seed)
    device = torch.device(args.device)

    victim_model = resnet.resnet50(pretrained=False)
    num_fc_in_features = victim_model.fc.in_features
    victim_model.fc = torch.nn.Linear(num_fc_in_features, 1)
    victim_model.load_state_dict(torch.load(args.target_model_path, map_location='cpu'))
    victim_model.to(device)
    victim_model.eval()

    G = Generator(args.stylegan_path, device)

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    bcelogit_loss = nn.BCEWithLogitsLoss().to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)

    clip_text_consistent = clip.tokenize(args.clip_token).to(device)

    attr = args.target_model_path[args.target_model_path.find("resnet50_")+9 :args.target_model_path.find("_trainfull")]
    if args.outdir == None:
        outdir = f'./{args.experiment}/{attr}/'
    else:
        outdir = args.outdir
    print(f"output folder is {outdir}")

    if args.mode == "single":
        os.makedirs(f'{outdir}/single/', mode=0o777, exist_ok=True)
    else:
        os.makedirs(f'{outdir}/multiple/', mode=0o777,exist_ok=True)


    cls_weight = args.cls_weight
    clip_weight = args.clip_weight
    tv_weight = args.tv_weight
    l2_weight = args.l2_weight


    ds_label_ori, ds_weights_ori = prepare_label(args, device, args.style_beta_or_channel)
    overall = {a:0 for a in ds_weights_ori}
    overall_changes = {a:0 for a in ds_weights_ori}
    overall_logit_changes = {a:0 for a in ds_weights_ori}

 

    def attack( target, styles , index , combination=[], text=False, save_img = True):
        ds_label = ds_label_ori.copy()
        ds_weights = ds_weights_ori.copy()
        ds_weights_adv =  ds_weights_ori.copy()
        # global confusion_matrix

        single_mode =  target != ""  
        if type(target) is list:
            ds_label = { your_key: ds_label[your_key] for your_key in target }
            ds_weights = { your_key: ds_weights[your_key] for your_key in target }
            ds_weights_adv = { your_key: ds_weights_adv[your_key] for your_key in target }

        for attack_iter in range(args.attack_iter):
            s = styles.copy()

            if single_mode:
                ds_weights[target].requires_grad=True
                for x in s:
                    ds = ds_label[target][x]
                    s[x] = s[x] + ds_weights[target] * ds
            else:
                for target in ds_label:
                    ds_weights[target].requires_grad=True
                    for x in s:
                        ds = ds_label[target][x]
                        s[x] = s[x] + ds_weights[target] * ds


            img = G.synthesis_from_stylespace(w, s)
            img_resize224 = custom_transform_resize(img)
            img_resize224 = (img_resize224 * 127.5 + 128).clamp(0, 255) / 255.0
            img_resize224 = custom_transform_cls(img_resize224)
            
            victim_logit = victim_model(img_resize224).type(torch.float64)
            # breakpoint()
            clip_image = custom_transform_clip((img+1)/2)
            logits_per_image, logits_per_text = clip_model(clip_image, clip_text_consistent)
            clip_prob = logits_per_image.softmax(dim=-1)


            if attack_iter == 0:
                original_image = img.clone().detach()
                clip_pred = clip_prob.argmax(dim=-1).to(device)
                clip_pred.requires_grad = False
                ground_truth = (torch.sigmoid(victim_logit) > 0.5).type(torch.float64).to(device)
                ground_truth_inv = (torch.sigmoid(victim_logit) <= 0.5).type(torch.float64).to(device)
                gt = float(torch.sigmoid(victim_logit))
                # print("ground truth: ", float(gt))
                ground_truth.requires_grad = False

                img_normalize = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                output_img = PIL.Image.fromarray(img_normalize[0].cpu().numpy(), 'RGB')
                output_img = output_img.resize((512,512))
                if draw_logit:
                    output_img = draw_label(output_img, f"{gt:.2f}" )
                output_img_ori = output_img


            # ================================ Backward ===================================


            cls_cost = bcelogit_loss(victim_logit, ground_truth_inv)
            clip_cost = ce_loss(logits_per_image, clip_pred)

            victim_model.zero_grad()
            G.G.zero_grad()
            
            final_cost = cls_weight * cls_cost + clip_weight * clip_cost # + tv_weight * tv_cost
            final_cost.backward()

            # ================================ Draw Image ===================================
            def draw_text(output_img):
                output_img_draw = PIL.ImageDraw.Draw(output_img)
                output_img_string = "Iter: {}, Model Logit: {}, Pred: {}".format(attack_iter, victim_logit.cpu().detach().numpy(), (torch.sigmoid(victim_logit) > 0.5).cpu().detach().numpy())
                output_img_font = PIL.ImageFont.load_default()
                output_img_loc = (10, 20)
                output_img_draw.rectangle((output_img_loc[0], output_img_loc[1] - output_img_font.getsize(output_img_string)[1], output_img_loc[0] + output_img_font.getsize(output_img_string)[0], output_img_loc[1] + output_img_font.getsize(output_img_string)[1]), fill="#000000")
                output_img_draw.text((output_img_loc[0], output_img_loc[1]), output_img_string, font=output_img_font, fill="#ffffff")
                return output_img
            
            if attack_iter == args.attack_iter - 1:
                if save_img and abs(float(torch.sigmoid(victim_logit)) - gt) > 0.7:

                    img_normalize = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    output_img = PIL.Image.fromarray(img_normalize[0].cpu().numpy(), 'RGB')
                    output_img = output_img.resize((512,512))
                    
                    # if text:
                    #     output_img = draw_text(output_img)
                    if draw_logit:
                        output_img = draw_label(output_img, f"{float(torch.sigmoid(victim_logit)):.2f}" )
                    if single_mode:
                        output_img_ori.save(f'{outdir}/single/{index}--original.png')
                        output_img.save(f'{outdir}/single/{index}-{attack_iter:04d}-{target}.png')
                    else:
                        output_img_ori.save(f'{outdir}/multiple/{index}--original.png')
                        output_img.save(f'{outdir}/multiple/{index}-{attack_iter:04d}.png')

                    output_img.close()
                attacked_s = s.copy()

            # ================================ Next Iter ===================================

            s = styles.copy()
            if single_mode:
                ds_weights_adv[target] = ds_weights[target] - args.attack_step_size * ds_weights[target].grad
                ds_weights_adv[target] = ds_weights_adv[target].clamp(-args.attack_bound, args.attack_bound)
                ds_weights[target] = ds_weights_adv[target].detach()

            else:
                for target in ds_label:
                    ds_weights_adv[target] = ds_weights[target] - args.attack_step_size * ds_weights[target].grad.sign()
                    ds_weights_adv[target] = ds_weights_adv[target].clamp(-args.attack_bound, args.attack_bound)
                    ds_weights[target] = ds_weights_adv[target].detach()




        # ================================= ATTACK FINISHED =================================#####

        if single_mode:

            logit_changes[target] = float(torch.sigmoid(victim_logit)) - gt
            print(f"attr: {target} | logit changes: {float(logit_changes[target])} |   weight {float(ds_weights[target])}")
            
        else:
            print( f"logit changes: {float(torch.sigmoid(victim_logit)) - gt}")
            print(f"final logit: {float(torch.sigmoid(victim_logit))}")
            for target in ds_label:
                logit_changes[target] = float(torch.sigmoid(victim_logit)) - gt
                # print(f"attr: {target} |  weight {float(ds_weights[target])}")

            print({k:f"{float(ds_weights[k]):.6f}" for k in ds_weights})
        flipped = (float(torch.sigmoid(victim_logit)) > 0.5) != (gt > 0.5)
        psnr_current = 0
        ssim_current = 0 
        del original_image
        return ds_weights, logit_changes, attacked_s, flipped, psnr_current, ssim_current


    all_flipped = 0
    all_psnr = 0
    all_ssim = 0
    for i in range(args.num_sample):
        
        truncation_psi = 0.7
        noise_mode = 'const' # 'const', 'random', 'none'

        label = torch.zeros([1, 1], device=device)
        z_ori = np.random.randn(1, G.G.z_dim)

        z_ori = torch.from_numpy(z_ori).to(device)
        z = z_ori.clone().detach()
        w_ori = G.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=8)
        styles = G.mapping_stylespace(w_ori)

        s = styles.copy()
        for x in s: 
            s[x].detach()
        # img_ori = G.synthesis_from_stylespace(w_ori, styles)
        w = w_ori.clone().detach()

        print(f"===========================Sample Index {i}==========================")


        ds_label, ds_weights = ds_label_ori.copy(), ds_weights_ori.copy()
        logit_changes = {}
        
        us_loc = [pos for pos, char in enumerate(args.target_model_path) if char == '_']
        classifier_name = args.target_model_path[us_loc[-2]: us_loc[-1]]
        if args.mode in "single_attribute":
            for target in ds_label:
                ds_weights, logit_changes, attacked_s, flipped, _, _  = attack(target, styles, i, text=True)
                overall_logit_changes[target] += abs( float(logit_changes[target]) )
                overall[target] += float(ds_weights[target])
                overall_changes[target] += abs(float(ds_weights[target]))

            print("Expectation of Abosulte Logit change:", {k:f"{v/(i+1):.15f}" for k,v in sorted(overall_logit_changes.items(), key=lambda item: abs(item[1]))})
            
            histogram(overall_logit_changes, f"{outdir}/barchart-logit-single.png", f"{args.experiment} on {classifier_name}" )
        
        else:
            ds_weights, logit_changes, attacked_s, flipped, psnr_current, ssim_current = attack( "", styles, i, text=True)
            for a in overall:
                overall[a] += float(ds_weights[a])
                overall_changes[a] += abs(float(ds_weights[a]))
            if flipped:
                all_flipped += 1

            # breakpoint()
            all_psnr += psnr_current
            all_ssim += ssim_current
            print(f"current flip rate: {all_flipped /(i+1)}")


        print(f"current average psnr: {all_psnr /(i+1)}")
        print(f"current average ssim: {all_ssim /(i+1)}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--device', type=str, default='cuda:4', help='cuda device')

    parser.add_argument('--stylegan_path', type=str, default='./pretrained/afhqcat.pkl', help='StyleGAN parameter file')

    parser.add_argument('--target_model_path', type=str, default='./pretrained/resnet50_Dog_trainfull.pkl', help='Target model')

    parser.add_argument('--clip_token', default=["a cat", "a dog"], help='Clip token')

    parser.add_argument('--cls_weight', type=float, default=1, help='Classification loss weight')

    parser.add_argument('--clip_weight', type=float, default=0.005, help='Clip loss weight')

    parser.add_argument('--tv_weight', type=float, default=0.001, help='Total Variation loss weight')

    parser.add_argument('--l2_weight', type=float, default=0.003, help='L2 loss weight')

    parser.add_argument('--num_sample', type=int, default=1000, help='Number of images sampled from StyleGAN')

    parser.add_argument('--attack_iter', type=int, default=100, help='Number of attack iterations per image')
    
    parser.add_argument('--num_attr', type=int, default=10000, help='Number of a first attr used in attr list, used in controlling attributes(table1)')

    parser.add_argument('--attack_step_size', type=float, default=1, help='Attack step size')

    parser.add_argument('--attack_bound', type=float, default=20, help='Attack bound')

    parser.add_argument('--style_beta_or_channel', type=float, default=0.1, help='if this < 5, then it is beta, else it is num_channel ')

    parser.add_argument('--experiment', type=str, default='ffhq', choices=exp_list, help='experiment name')

    parser.add_argument('--outdir', type=str, default=None,help='output image place')

    parser.add_argument('--target_attr', type=str, default=None, help='single target')

    parser.add_argument('--mode', type=str, default='single', choices=["single", "multiple"], help='experiment name')

    args = parser.parse_args()
    main(args)
