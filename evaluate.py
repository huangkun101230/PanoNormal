import os
import argparse
import torch
import torch.nn.functional as F
import utils.file_utils as fu
import numpy as np
import tqdm

import datasets.dataset3D60 as dataset3D60
# import datasets.structured3d as sturctured3d

import PanoNormal

from Metrics.norm_metrics import Norm_Evaluator

def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda:", use_cuda)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    np.random.seed(args.seed)

    test_set = dataset3D60.Dataset3D60('./datasets/3d60_examples.txt', 256, 512)
    # test_set = dataset3D60.Dataset3D60('./datasets/3d60_test.txt', 256, 512)
    # test_set = dataset3D60.Dataset3D60('./datasets/stanford2d3d_test.txt', 256, 512)
    # test_set = dataset3D60.Dataset3D60('./datasets/Matterport3D_test.txt', 256, 512)
    # test_set = dataset3D60.Dataset3D60('./datasets/SunCG_test.txt', 256, 512)
    # test_set = sturctured3d.Structured3d('./datasets/structured3d_test.txt', 256, 512)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=False, **kwargs)

    model = PanoNormal.DeepNet()

    if torch.cuda.is_available():
        model.cuda()
                                                
    checkpoint = torch.load(model_path,map_location='cuda:0')
    pretrained_dict = checkpoint['model']
    model.load_state_dict(pretrained_dict)    
    start_epoch = checkpoint['epoch']

    print('loading epoch {} successfully！'.format(start_epoch))

    norm_evaluator = Norm_Evaluator()
    norm_evaluator.reset_eval_metrics()

    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Evaluating")

    with torch.no_grad():
        for batch_idx, test_sample in enumerate(pbar):

            rgb = test_sample["ori_rgb"].to(device)
            norm = test_sample["gt_surface"]
            mask = test_sample["mask"]

            outputs = model(rgb)
            output_norm = outputs["pred_normal"]

            output_norm = F.normalize(output_norm, p = 2, dim = 1)

            output_norm = output_norm.detach().cpu()*mask
            norm_evaluator.compute_eval_metrics(norm, output_norm,mask)
            fu.save_norm_tensor_as_float(output_path, test_sample['surface_filename'][0]+'_pred', output_norm[0])
            fu.save_norm_tensor_as_float(output_path, test_sample['surface_filename'][0]+'_gt', norm[0])

    norm_evaluator.print(eva_path)


if __name__ == "__main__":
    folder = 'saved_models'

    model_path = './'+folder+'/models/model.pkl'
    print(model_path)
    output_path = './results/'+folder+'/results/'
    eva_path = './results/'+folder+'/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    main()