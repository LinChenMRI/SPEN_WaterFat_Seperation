% main function of comparing WFSR-CG, WFSR-SWAF and WFSR-DL
% The demo for conjugate gradient (CG), super-resolved water/fat image reconstruction (SWAF), and deep learning methods for water-fat separation and 
% super-resolved reconstruction (WFSR) in SPEN MRI
% WFSR-CG:
% [1] Schmidt R, Frydman L. In vivo 3D spatial/1D spectral imaging by spatiotemporal encoding: a new single-shot experimental and processing approach. 
% Magn Reson Med. 2013;70(2):382-391.
% WFSR-SWAF:
% [2] Huang J, Chen L, Chan KWY, Cai C, Cai S, Chen Z. Super-resolved water/fat image reconstruction based on single-shot spatiotemporally encoded MRI. 
% J Magn Reson. 2020;314:106736.
% sorted out by ChenXinran, 11/08/2021
clear all; close all; clear class; clc
addpath(strcat(pwd,'/Toolbox'));
%% load in experiment data
direction.fid_dir = ['example data',filesep,'rat.fid']; % The directory of FID file
Procpar = readprocpar(direction.fid_dir);   % load the experiment parameters
direction.savedatadir = ['reconstruction_result'];
if exist(direction.savedatadir,'file')==0
    mkdir(direction.savedatadir);
end
%% WFSR-CG
[water_CG,fat_CG,time_CG] = WFSR_CG_func(direction,Procpar);% obtain CG water/fat reconstruction results
a1 = sprintf('   CG reconstruction has done in %0.4f s',time_CG);
disp(a1)
disp('------------------------------------------------')
%% WFSR-SWAF
[water_SWAF,fat_SWAF,time_SWAF] = WFSR_SWAF_func(direction,Procpar);% obtain SWAF water/fat reconstruction results
a2 = sprintf('   SWAF reconstruction has done in %0.4f s',time_SWAF);
disp(a2)
%% WFSR-DL
% % step1:codes for DL pre-processing
[time_DL_pre] = WFSR_DL_preprocessing_func(direction,Procpar);% preprocessing for subsequent DL reconstruction
% % step2:put data into trained U-net to get inference (this step can also
% % be done in Pycharm)
%%%%%% attention!! please check or define a avaliable pyversion in MATLAB 
%%%%%% (recommend creating a new virtual environment and install requirements.txt)

% % for Linux system
% % example:
% pyversion /data1/oyby/PycharmProjects/venv/bin/python
% % for Windows system
% % example:
% pyversion C:\Anaconda3\python.exe

% % add current directory to the Python search path
% if count(py.sys.path,'') == 0
%     insert(py.sys.path,int32(0),'');
% end
% % inference part
% py.My_Interface_WFSR.test('water', 'UNet_trained_for_water') 
% py.My_Interface_WFSR.test('fat', 'UNet_trained_for_fat') 

% % step3: load in DL inference results and show
load(['reconstruction_result',filesep,'DL_result',filesep,'WFSR_DL_water1.mat'])
water_DL = rot90(output_image);
time_water = running_time;
load(['reconstruction_result',filesep,'DL_result',filesep,'WFSR_DL_fat1.mat'])
fat_DL = rot90(output_image);
time_fat = running_time;
time_DL = time_DL_pre+time_water+time_fat;% total time for DL reconstruction
disp('------------------------------------------------')
a3 = sprintf('   DL reconstruction has done in %0.4f s',time_DL);
disp(a3)
%% results show
figure(1);subplot(2,3,1);imshow(abs(water_CG),[],'InitialMagnification' ,'fit');title('CG water');set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',14)
subplot(2,3,4);imshow(abs(fat_CG),[],'InitialMagnification' ,'fit');title('CG fat');set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',14)
subplot(2,3,2);imshow(abs(water_SWAF),[],'InitialMagnification' ,'fit');title('SWAF water');set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',14)
subplot(2,3,5);imshow(abs(fat_SWAF),[],'InitialMagnification' ,'fit');title('SWAF fat');set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',14)
subplot(2,3,3);imshow(abs(water_DL),[],'InitialMagnification' ,'fit');title('DL water');set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',14)
subplot(2,3,6);imshow(abs(fat_DL),[],'InitialMagnification' ,'fit');title('DL fat');set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',14)



