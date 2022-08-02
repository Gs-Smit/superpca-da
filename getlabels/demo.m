clc;clear;close all
addpath('.\libsvm-3.21\matlab');
addpath(genpath(cd));


database         =   'Indian';

%% load the HSI dataset
if strcmp(database,'Indian')
    load Indian_pines_corrected;load Indian_pines_gt;load Indian_pines_randp 
    data3D = indian_pines_corrected;        label_gt = indian_pines_gt;
    num_Pixel        =   100;
elseif strcmp(database,'PaviaU')    
    load PaviaU;load PaviaU_gt;load PaviaU_randp; 
    data3D = paviaU;        label_gt = paviaU_gt;
    num_Pixel        =   20;
end
data3D = data3D./max(data3D(:));

%% super-pixels segmentation
labels = cubseg(data3D,num_Pixel);

save('Indian_labels_num_Pixels_100.mat',  'labels');
