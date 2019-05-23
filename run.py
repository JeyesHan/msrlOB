import torch
import torchvision
from models.Q_net import Q_zoom, Q_refine
from data import load_images_names_in_data_set, get_bb_of_gt_from_pascal_xml_annotation
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from PIL import Image, ImageDraw
from utils import cal_iou, reward_func
import time

# hyper-parameters
BATCH_SIZE = 100
LR = 1e-6
GAMMA = 0.9
MEMORY_CAPACITY = 1000
Q_NETWORK_ITERATION = 100
epochs = 50
NUM_ACTIONS = 6
his_actions = 4
subscale = 1/2
NUM_STATES = 7*7*512+his_actions*NUM_ACTIONS
path_voc = "/home/hanj/dataset/VOCdevkit/VOC2007/"

class DQN():
    """docstring for DQN"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.eval_net, self.target_net = Q_zoom(), Q_zoom()
        self.eval_net.to(self.device)
        self.target_net.to(self.device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPISILO):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device) # get a 1D array
        if np.random.randn() <= EPISILO:# random policy
            action = np.random.randint(0, NUM_ACTIONS)
        else: # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].cpu().item()
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int)).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:]).to(self.device)

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target_unterminated = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = torch.where(batch_action!=5,q_target_unterminated,batch_reward)
        loss = self.loss_func(q_eval, q_target)
        print("step loss is {:.3f}".format(loss.cpu().detach().item()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def init_process(image, transform=None):
    # image.show()
    # time.sleep(5)
    if transform:
        image = transform(image)
    return image.unsqueeze(0)

def inter_process(image, bbx, transform=None):
    (left,upper,right,lower)=(bbx[0],bbx[2],bbx[1],bbx[3])
    image_crop = image.crop((left,upper,right,lower))
    # image_crop.show()
    # time.sleep(5)
    if transform:
        image_crop = transform(image_crop)
    return image_crop.unsqueeze(0)

def update_bbx(bbx, action):
    new_bbx = np.zeros(4)
    if action == 0:         #top left
        new_bbx[0] = bbx[0] #x1
        new_bbx[1] = bbx[0] + (bbx[1]-bbx[0]) * subscale    #x2
        new_bbx[2] = bbx[2]  # y1
        new_bbx[3] = bbx[2] + (bbx[3]-bbx[2]) * subscale    #y2
    elif action == 1:       #top right
        new_bbx[0] = bbx[1] - (bbx[1]-bbx[0]) * subscale    #x1
        new_bbx[1] = bbx[1]     #x2
        new_bbx[2] = bbx[2]# y1
        new_bbx[3] = bbx[2] + (bbx[3]-bbx[2]) * subscale    #y2
    elif action == 2:       #lower left
        new_bbx[0] = bbx[0]#x1
        new_bbx[1] = bbx[0] + (bbx[1]-bbx[0]) * subscale    #x2
        new_bbx[2] = bbx[3] - (bbx[3]-bbx[2]) * subscale # y1
        new_bbx[3] = bbx[3]#y2
    elif action == 3:        #lower right
        new_bbx[0] = bbx[1] - (bbx[1]-bbx[0]) * subscale    #x1
        new_bbx[1] = bbx[1] #x2
        new_bbx[2] = bbx[3] - (bbx[3]-bbx[2]) * subscale    #y1
        new_bbx[3] = bbx[3] #y2
    elif action == 4:        #center
        new_bbx[0] = (bbx[0]+bbx[1])/2-(bbx[1]-bbx[0]) * subscale/2 #x1
        new_bbx[1] = (bbx[0]+bbx[1])/2+(bbx[1]-bbx[0]) * subscale/2 #x2
        new_bbx[2] = (bbx[2]+bbx[3])/2-(bbx[3]-bbx[2]) * subscale/2 #y1
        new_bbx[3] = (bbx[2]+bbx[3])/2+(bbx[3]-bbx[2]) * subscale/2 #y2
    elif action == 5:
        new_bbx = bbx
    return new_bbx


def main(args):

    # Class category of PASCAL that the RL agent will be searching
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    image_names = np.array(load_images_names_in_data_set('aeroplane_trainval', path_voc))
    feature_exactrator = torchvision.models.vgg16(pretrained=True).features.to(device)
    single_plane_image_names = []
    single_plane_image_gts = []
    dqn = DQN(device)
    EPISILO = args.EPISILO


    for image_name in image_names:
        annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
        if(len(annotation)>1):
            continue
        single_plane_image_names.append(image_name)
        single_plane_image_gts.append(annotation[0][1:])        #[[x1,x2,y1,y2] ...]

    trans = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
    ])


    for i in range(epochs):
        ep_reward = 0
        for index, image_name in enumerate(single_plane_image_names):
            image_path = os.path.join(path_voc + "JPEGImages", image_name + ".jpg")
            image_original = Image.open(image_path)
            width, height = image_original.size
            #image_original = image_original.resize((224,224))
            bbx_gt = single_plane_image_gts[index]
            #draw = ImageDraw.Draw(image_original)
            #draw.rectangle([bbx_gt[0],bbx_gt[2],bbx_gt[1],bbx_gt[3]],outline='red')
            #image_original.show()
            #return

            image = init_process(image_original, trans).to(device)
            #print(image.shape)
            bbx = [0, width, 0, height]
            history_action = np.zeros(his_actions*NUM_ACTIONS)
            with torch.no_grad():
                vector = feature_exactrator(image).cpu().detach().numpy().reshape(7*7*512)
            state = np.concatenate([history_action, vector])
            step = 0
            while(step<10):
                iou = cal_iou(bbx, bbx_gt)
                if iou>0.5:
                    action = 5
                else:
                    action = dqn.choose_action(state, EPISILO)
                #print(action)

                #execute action and step to new bbx
                new_bbx = update_bbx(bbx, action)
                reward = reward_func(bbx, new_bbx, bbx_gt, action)

                #get new state
                action_vec = np.zeros(NUM_ACTIONS)
                action_vec[action] = 1.0
                history_action = np.concatenate([history_action[NUM_ACTIONS:], action_vec])

                with torch.no_grad():
                    vector = feature_exactrator(inter_process(image_original,new_bbx,trans).to(device)).cpu().detach().numpy().reshape(7*7*512)
                next_state = np.concatenate([history_action,vector])

                #store transition
                dqn.store_transition(state, action, reward, next_state)

                ep_reward += reward

                if dqn.memory_counter >= MEMORY_CAPACITY:
                    print("episode: {},".format(i),end=' ')
                    dqn.learn()

                #termation
                if action==5:
                    break

                state = next_state
                bbx = new_bbx
                step += 1

        if (EPISILO>0.1):
            EPISILO -= 0.1
        print("episode: {} , this epoch reward is {}".format(i, round(ep_reward, 3)))  # 0.001 precision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical Object Detection with Deep Reinforcement Learning')
    parser.add_argument('--gpu-devices', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use_gpu', default=True, action='store_true')
    parser.add_argument('--EPISILO', type=int, default=1.0)

    main(parser.parse_args())