function [water_roh,fat_roh,time_CG] = WFSR_CG_func(direction,Procpar)% 
% codes for CG reconstruction
% [1] Schmidt R, Frydman L. In vivo 3D spatial/1D spectral imaging by spatiotemporal encoding: a new single-shot experimental and processing approach. 
% Magn Reson Med. 2013;70(2):382-391.
%% Get the experiment parameters
tic
param.gama = 2.6752e8;
param.segment = 1;  % The number of segments used in the experiment
% * Extract the paramters which are needed in the reconstruction process* 
param.Lpe = Procpar.lpe.Values*1e-2; % The FOV of phase encoded dimension
param.pw = Procpar.p1_bw.Values; % The bandwidth of the chirp pulse
param.Texc = Procpar.pw1.Values; % The duration of the chirp pulse
param.Tacq = 20e-3;
param.nphase = Procpar.nphase.Values; % The number of points in the phase encoded dimension
param.nread = Procpar.nread.Values/2; % The number of points in the frequency encoded dimension
param.chemshift = 1014*pi*2;% define the chemical shift here
param.chirp = 1; % 1 for 90 chirp and 2 for 180 chirp. 
param.sign = 1; % choose 1 or -1 according to the sweep direction of chirp
param.fft_number = 128; % The digital resolution of SR image
param.SVDpreprocessing = 0; % SVD is used to reduce the condition number of linear equations to make the algorithm more robust
% * Calculate the other paramters* 
param.Gexc = 2*pi*param.sign*param.pw/(param.gama*param.Lpe);
param.Gacq = -param.Gexc*param.Texc/param.Tacq;
param.R = param.gama*param.Gexc*param.Lpe/param.Texc;
% *Create the directory for saving the CG results*
direction.savedatadir_CG=[direction.savedatadir, filesep,'CG_result'];% data save folder for CG
if exist(direction.savedatadir_CG,'file')==0
    mkdir(direction.savedatadir_CG);
end
%%
% *Load the original data in the SPEN sampling domain*
[RE,IM,NP,NB,NT,HDR]=load_fid(direction.fid_dir,param.segment); % load fid data,choose the wanting segment
fid=RE+1i*IM;   % translate into complex form
fid=fid(1+param.nread*2:end,:); % discard the front useless data
zerofillnum=(param.fft_number-param.nread)/2;% the number of zero-filling points 
for n = 1:1:NT
    fid_temp = reshape(fid(:,n),param.nread,param.nphase).'; % Translate into 2D
    fid_temp(2:2:end,:) =  fliplr(fid_temp(2:2:end,:)); % Rearrange the even lines
    fid_temp_zerofill = [zeros(param.nphase,zerofillnum),fid_temp,zeros(param.nphase,zerofillnum)]; % Zero-fill the fid
    fid_temp_fftshift  = fftshift(fid_temp_zerofill,2);
    st_temp_fftshift = fft(fid_temp_fftshift,[],2);
    st_temp = fftshift(st_temp_fftshift,2);
    param.st(:,:,n) = st_temp;
end
% imwrite(abs(param.st)/max(abs(param.st(:))),[direction.savedatadir_CG,filesep,'Blurred image.tiff'],'tiff')% output
% figure;imshow(abs(param.st),[]);title('Blurred image');
% figure;imshow(angle(param.st),[]);title('Blurred angle');
%%------------------Construct P matrix here----------------------------------------------%
a=param.gama*param.gama*param.Gexc*param.Gexc/(2*param.R);
b=param.gama*param.gama*param.Gexc*param.Gexc*param.Lpe/(2*param.R);
c=param.gama*param.gama*param.Gexc*param.Gexc*param.Lpe*param.Lpe/(8*param.R);
d=param.gama*param.Gacq;
%-------define steps------------%
step_number=40;
y_step=param.Lpe/param.nphase*2;
y_step_step=y_step/step_number;
t_step=param.Tacq/param.nphase;
t_step_step=t_step/step_number;
for x=1:param.nphase
    current_t=(x)*t_step;
    for y=1:param.nphase/2
        current_water_y=-param.Lpe/2+(y-1)*y_step;
        temp_water_re=0;
        temp_water_im=0;
        %fat
        current_fat_y=-param.Lpe/2+(y-1)*y_step-param.chemshift/(param.gama*param.Gexc);
        temp_fat_re=0;
        temp_fat_im=0;
        for z=1:step_number
            current_water_angle=-a*current_water_y*current_water_y+b*current_water_y-c-param.sign*pi/2+d*current_t*current_water_y;
            temp_water_re=temp_water_re+y_step_step*cos(current_water_angle);
            temp_water_im=temp_water_im+y_step_step*sin(current_water_angle);
            current_water_y=current_water_y+y_step_step;
            %fat
            current_fat_angle=-a*current_fat_y*current_fat_y+b*current_fat_y-param.Texc*param.chemshift*current_fat_y/param.Lpe-c-param.sign*pi/2+d*current_t*current_fat_y+param.chemshift*current_t;
            temp_fat_re=temp_fat_re+y_step_step*cos(current_fat_angle);
            temp_fat_im=temp_fat_im+y_step_step*sin(current_fat_angle);
            current_fat_y=current_fat_y+y_step_step;
        end
        P_water_re(x,y)=temp_water_re/y_step;
        P_water_im(x,y)=temp_water_im/y_step;
        %fat
        P_fat_re(x,y)=temp_fat_re/y_step;
        P_fat_im(x,y)=temp_fat_im/y_step;
    end
end 
param.P_matrix_water=P_water_re+1i*P_water_im;
param.P_matrix_fat=P_fat_re+1i*P_fat_im;
param.P_matrix_mix=[param.P_matrix_water,param.P_matrix_fat];

[U,S,V] = svd(param.P_matrix_mix,'econ');
param.P_matrix_mix = U*V';
% figure,imagesc(abs(param.P_matrix_mix));title('P abs');

result_without_CS = param.P_matrix_mix'*param.st;
% figure,imshow(abs(result_without_CS),[]);title('SR without CG');

[N,M]=size(param.st);

for m=1:1:M
     [roh(:,m),FLAG,RELRES,ITER]=bicg(param.P_matrix_mix,param.st(:,m),5e-12,400);
end
% figure,imshow(abs(roh),[]);title('roh with CG');
result_with_CS=roh;
water_result=result_with_CS(1:32,:);
fat_result=result_with_CS(33:end,:);
L=[param.Lpe,param.Lpe];
num_x=round(128/min(L)*param.Lpe);
num_y=round(128/min(L)*param.Lpe);

res_mix = result_with_CS(1:32,:)+result_with_CS(33:end,:);
res_mix=flipud(imresize(res_mix,[num_x num_y],'cubic'));
% figure,imshow(abs(res_mix),[],'InitialMagnification' ,'fit');title('mix result');


water_roh=flipud(imresize(water_result,[num_x num_y],'cubic'));
water_roh = water_roh/max(max(abs(water_roh)));
% figure,imshow(abs(water_roh),[],'InitialMagnification' ,'fit');title('water CG result');
% imwrite(abs(water_roh)/max(abs(water_roh(:))),[direction.savedatadir_CG,filesep,'CG_water_result.tiff'],'tiff')% output

fat_roh=flipud(imresize(fat_result,[num_x num_y],'cubic'));
fat_roh = fat_roh/max(max(abs(fat_roh)));
time_CG=toc;
% figure,imshow(abs(fat_roh),[],'InitialMagnification' ,'fit');title('fat CG result');
% imwrite(abs(fat_roh)/max(abs(fat_roh(:))),[direction.savedatadir_CG,filesep,'CG_fat_result.tiff'],'tiff')% output
% save reconstruction results
save([direction.savedatadir_CG,filesep,'CG_fat_result.mat'],'fat_roh')
save([direction.savedatadir_CG,filesep,'CG_water_result.mat'],'water_roh')

end

