"""

@author: Pavithra Vijayakrishnan

"""

import torch
import torch.nn as nn


class deep_channel_estimator(nn.Module):

        def __init__(self,in_channel,out_channel,layers = 6,bs_state  = False):
            super(deep_channel_estimator, self).__init__()

            padd  = 1
            ksize = 3
            intermediate_layer = []
            first_layer = 1
            while(first_layer < layers):
                        #intermediate_layer.append(nn.ReflectionPad2d(padd))
                        intermediate_layer.append(nn.Conv2d(out_channel, out_channel, kernel_size=ksize, stride=1,bias = bs_state, padding = padd))
                        intermediate_layer.append(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True))
                        intermediate_layer.append(nn.ReLU())
                        intermediate_layer.append(nn.BatchNorm2d(out_channel))
                        first_layer += 1
            #intermediate_layer.append(nn.ReflectionPad2d(padd))
            intermediate_layer.append(nn.Conv2d(out_channel, out_channel, kernel_size=ksize, stride=1,bias = bs_state, padding = padd))
            intermediate_layer.append(nn.ReLU())
            intermediate_layer.append(nn.BatchNorm2d(out_channel))

            #intermediate_layer.append(nn.ReflectionPad2d(padd))
            intermediate_layer.append(nn.Conv2d(out_channel,in_channel, kernel_size=ksize, stride=1,bias = bs_state, padding = padd))
            self.model = nn.Sequential(*intermediate_layer)
            
            return



        def forward(self,x):

            x = torch.unsqueeze(x, 0)
            x = self.model(x)
            return x
