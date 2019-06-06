import tensorflow as tf 
import numpy as np 
import cv2
import os
import random
import scipy.io
import scipy.ndimage
from scipy.misc import imread, imresize
from utilities import preprocess_fixmaps, preprocess_images, preprocess_maps
from config import *
from Net import Net

# test时视频路径
VideoName= "/Users/yanhang/Downloads/animal_alpaca01.mp4"






#显著度视频的batch提取，batch_shape: (batch_size, frame_num, width, height, RGB)
#对一个视频随机抽取一段连续帧为frame_num的一组实例
def _VideoBatch_Generator(phase_gen='train'):
    '''
    Args:
    VideoCap: cv2预读取视频的操作
    batch_size: batch的大小
    frame_num: 一个视频的连续帧
    resize_shape: 输出视频帧的大小
    Out: video_batch: (batch_size, frame_num, width, height, RGB)
    '''
    VideoCap = cv2.VideoCapture(VideoName)
    
    #frame_width = VideoCap.get(3)  #3:CV_CAP_PROP_FRAME_HEIGHT
    #frame_height = VideoCap.get(4) #4:CV_CAP_PROP_FRAME_HEIGHT
    if phase_gen == 'test':
        Video_frames = VideoCap.get(7)  #7:CV_CAP_PROP_FRAME_COUNT
        print(Video_frames)
        
        batch_size = Video_frames // frame_num
        if not Video_frames % frame_num == 0:
            batch_size = batch_size + 1
        print(batch_size)
        
        start_frame = 0
        Video_Batch = np.zeros(shape=[1, 1, 720, 1080, 3])
        
        
        for j in range(int(batch_size)):
            Video_Slice = np.zeros(shape=[1, 1, 720, 1080, 3])
            if start_frame + frame_num > Video_frames:
                start_frame = Video_frames - frame_num
            stop_frame = start_frame + frame_num 
            for i in range(start_frame, stop_frame):
                VideoCap.set(1, i)          #1:CV_CAP_PROP_POS_FRAMES
                _, frame = VideoCap.read()   #（frame_width, frame_height, BGR）
                frame = cv2.resize(frame, input_shape) 
                frame = frame.astype(np.float32)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#（resize_width, resize_height, RGB）
                frame = frame[np.newaxis, np.newaxis, ...]
                if np.shape(Video_Slice)[1] == 1 and i == start_frame:
                    Video_Slice = frame
                else:
                    Video_Slice = np.concatenate((Video_Slice, frame), axis=1)                
                print(np.shape(Video_Slice))
            if np.shape(Video_Batch)[0] == 1 and start_frame == 0:
                Video_Batch = Video_Slice
            else:
                Video_Batch = np.concatenate((Video_Batch, Video_Slice), axis=0)
            print(np.shape(Video_Batch))
            start_frame = stop_frame
    #training process 读取的是文件夹里的照片
    # video_trainpaths里包含的是按数字标号保存的video，每个video文件家里三个文件夹 “images”（） “maps”
    if phase_gen == 'train':
        videos = [Video_train_paths + '/' + video_train_path for video_train_path 
                  in os.listdir(Video_train_paths) if os.path.isdir(Video_train_paths + '/' + video_train_path)]
        videos.sort()
        random.shuffle(videos)
        
        range_num = 0
        while True:
            Xims = np.zeros((batch_size, frame_num, input_shape[0], input_shape[1], 3))

            Ymaps = np.zeros((batch_size, frame_num, output_shape[0], output_shape[1], 1)) + 0.01
            Yfixs = np.zeros((batch_size, frame_num, output_shape[0], output_shape[1], 1)) + 0.01

            for i in range(batch_size):
                video = videos[(range_num * batch_size + i) % len(videos)]
                images = [video + frames_path + f for f in os.listdir(video + frames_path) if f.endswith((".jpg", ".jpeg", ".png"))]
                maps = [video + maps_path + f for f in os.listdir(video + maps_path) if f.endswith((".jpg", ".jpeg", "png"))]
                fixs = [video + fixs_path + f for f in os.listdir(video + fixs_path) if f.endswith(".mat")]
                
                start = np.random.choice(max(1, len(images) - frame_num))
                X = preprocess_images(images[start:min((start + frame_num), len(images))], input_shape[0], input_shape[1])
                Y = preprocess_maps(maps[start:min((start + frame_num), len(images))], output_shape[0], output_shape[1])
                Y_fix = preprocess_fixmaps(fixs[start:min((start + frame_num), len(images))], output_shape[0], output_shape[1])
                
                Xims[i, 0:np.shape(X)[0], :] = np.copy(X)
                Ymaps[i, 0:np.shape(Y)[0], :] = np.copy(Y)
                Yfixs[i, 0:np.shape(Y_fix)[0], :] = np.copy(Y_fix)
                
                Xims[i, X.shape[0]:frame_num, :] = np.copy(X[-1, :, :])
                Ymaps[i, Y.shape[0]:frame_num, :] = np.copy(Y[-1, :, :])
                Yfixs[i, Y_fix.shape[0]:frame_num, :] = np.copy(Y_fix[-1, :, :])
            yield Xims, Ymaps, Yfixs
            range_num = range_num + 1



if __name__ == '__main__':
    phase = "train"

    My_model = Net()
    
    if phase == "train":
        x = tf.placeholder(tf.float32, shape = [batch_size, frame_num, input_shape[0], input_shape[1], 3])
        ymaps = tf.placeholder(tf.float32, shape = [batch_size, frame_num, output_shape[0], output_shape[1], 1])
        yfixs = tf.placeholder(tf.float32, shape = [batch_size, frame_num, output_shape[0], output_shape[1], 1])

        My_model.inference(x)
        y_ = My_model.out
        loss_kl = My_model.kl_divergence(ymaps, y_)
        loss_cc = My_model.correlation_coefficient(ymaps, y_)
        loss_nss = My_model.nss(yfixs, y_)
        loss = loss_kl + 0.1 * loss_cc + 0.1 * loss_nss
        tf.summary.scalar('loss', loss)

        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


    if phase == "test":
        x = tf.placeholder(tf.float32, shape = [1, None, input_shape[0], input_shape[1], 3])

    if phase == "train":

        My_model.inference(x)
        y_pred = My_model.out

        Xims, Ymaps, Yfixs = _VideoBatch_Generator(phase_gen='train').__next__()

    