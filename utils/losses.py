import numpy as np
import torch
import torch.nn as nn
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def padding_right(ary):
    right_padding = ary[:,:,:,-1:].repeat(1,1,1,1)
    padded = torch.cat((ary, right_padding), axis=3)
    return padded

def padding_bottom(ary):
    bot_padding = ary[:,:,-1:,:].repeat(1,1,1,1)
    padded = torch.cat((ary, bot_padding), axis=2)
    return padded

# image gradient computations
'''
    Image gradient x-direction
    \param
        input_tensor
    \return 
        input_tensor's x-direction gradients
'''
def grad_x(input_tensor):
    input_tensor = padding_right(input_tensor)
    gx = input_tensor[:, :, :, :-1] - input_tensor[:, :, :, 1:]
    return gx

'''
    Image gradient y-direction
    \param
        input_tensor
    \return 
        input_tensor's y-direction gradients
'''
def grad_y(input_tensor):
    input_tensor = padding_bottom(input_tensor)
    gy = input_tensor[:, :, :-1, :] - input_tensor[:, :, 1:, :]
    return gy

'''
    Cosine Similarity loss (vector dot product)
    \param
        input       input tensor (model's prediction)
        target      target tensor (ground truth)
        use_mask    set True to compute masked loss
        mask        Binary mask tensor
    \return
        Cosine similarity loss mean between target and input
        Cosine similarity loss map betweem target and input
'''

class NormalMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        errors = torch.norm(pred - gt, dim=1)  # Calculate the L2 norm (Euclidean distance) for each data point
        # errors = torch.sqrt((pred[:,0]-gt[:,0])**2+(pred[:,1]-gt[:,1])**2+(pred[:,2]-gt[:,2])**2)
        count = torch.sum(mask).item()
        mse_loss = torch.sum(errors ** 2)/count  # Calculate the mean squared error
        return mse_loss

'''
    Quaternion loss
    \param
        input       input tensor (model's prediction)
        target      target tensor (ground truth)
        use_mask    set True to compute masked loss
        mask        Binary mask tensor
    \return
        Quaternion loss mean between target and input
        Quaternion loss map betweem target and input
'''
class quaternion_loss(nn.Module):
    def __init__(self):
        super(quaternion_loss, self).__init__()

    def forward(self, input, target, mask):
        q_pred = -input
        loss_x = target[:, 1, :, :] * q_pred[:, 2, :, :] - target[:, 2, :, :] * q_pred[:, 1, :, :]
        loss_y = target[:, 2, :, :] * q_pred[:, 0, :, :] - target[:, 0, :, :] * q_pred[:, 2, :, :]
        loss_z = target[:, 0, :, :] * q_pred[:, 1, :, :] - target[:, 1, :, :] * q_pred[:, 0, :, :]
        loss_re = -target[:, 0, :, :] * q_pred[:, 0, :, :] - target[:, 1, :, :] * q_pred[:, 1, :, :] - target[:, 2, :, :] * q_pred[:, 2, :, :]
        loss_x = loss_x.unsqueeze(1)
        loss_y = loss_y.unsqueeze(1)
        loss_z = loss_z.unsqueeze(1)
        # loss_xyz = torch.cat((loss_x, loss_y, loss_z), 1)
        
        dot = loss_x * loss_x + loss_y * loss_y + loss_z * loss_z
        eps = torch.ones_like(dot) * 1e-8

        vec_diff = torch.sqrt(torch.max(dot, eps))
        real_diff = torch.sign(loss_re) * torch.abs(loss_re)
        real_diff = real_diff.unsqueeze(1)
        
        loss = torch.atan2(vec_diff, real_diff) / (np.pi)
        
        if mask is not None:
            count = torch.sum(mask)
            # mask = mask[:, 0, :, :].unsqueeze(1)
            masked_loss = loss * mask
            return torch.sum(masked_loss) / count
        return torch.mean(loss)
    

'''
    Smoothness loss
    \param
        input   input tensor (model's prediction)
'''
def smoothness_loss(input, mask = None):
    grads_x = grad_x(input)
    grads_y = grad_y(input)
    loss = torch.abs(grads_x) + torch.abs(grads_y)
    if mask is not None:
        count = torch.sum(mask).item()
        masked_loss = mask * loss
        return torch.sum(masked_loss) / count
        # return torch.sum(masked_loss) / count, masked_loss
    return torch.mean(loss)
    # return torch.mean(loss), loss