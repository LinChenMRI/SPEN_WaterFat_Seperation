function [water_res,fat_res,time_SWAF] = WFSR_SWAF_func(direction,Procpar)
% writen by Jianpan Huang
% codes for SWAF reconstruction
% [1] Huang J, Chen L, Chan KWY, Cai C, Cai S, Chen Z. Super-resolved water/fat image reconstruction based on single-shot spatiotemporally encoded MRI. 
% J Magn Reson. 2020;314:106736.
%% Define parameters for reconstruction
tic
param.sign = 1;
param.fft_number = 128; 
param.gama = 2.6752e8;
param.segment = 1;
param.chirp = 1; 
param.ifSVD = 0;
param.wcs = 1014*2*pi; % define chemical shift
param.Tacq = 20e-3;
%% Get the experiment parameters
param.Lpe = Procpar.lpe.Values*1e-2;
param.pw = Procpar.p1_bw.Values;
param.Texc = Procpar.pw1.Values;
param.nphase = Procpar.nphase.Values;
param.nread = Procpar.nread.Values/2; %
% * Calculate the other paramters* 
param.Gexc = param.sign*2*pi*param.pw/(param.gama*param.Lpe);
param.Gacq = -param.Gexc*param.Texc/param.Tacq;
param.R = param.gama*param.Gexc*param.Lpe/param.Texc;

% *Create the directory for saving the SWAF results*
direction.savedatadir_SWAF=[direction.savedatadir, filesep,'SWAF_result'];% data save folder for CG
if exist(direction.savedatadir_SWAF,'file')==0
    mkdir(direction.savedatadir_SWAF);
end
% *Load the original data in the SPEN sampling domain*
[RE,IM,NP,NB,NT,HDR]=load_fid(direction.fid_dir,param.segment); 
fid=RE+1i*IM;
fid=fid(1+param.nread*2:end,:);
zerofillnum=(param.fft_number-param.nread)/2;
for n = 1:1:NT
    fid_temp = reshape(fid(:,n),param.nread,param.nphase).'; 
    fid_temp(2:2:end,:) =  fliplr(fid_temp(2:2:end,:));
    fid_temp_zerofill = [zeros(param.nphase,zerofillnum),fid_temp,zeros(param.nphase,zerofillnum)];
    fid_temp_fftshift  = fftshift(fid_temp_zerofill,2);
    st_temp_fftshift = fft(fid_temp_fftshift,[],2);
    st_temp = fftshift(st_temp_fftshift,2);
    param.st(:,:,n) = st_temp;
end
% figure,imshow(abs(param.st),[]);title('Blurred image');drawnow
% imwrite(abs(param.st)/max(abs(param.st(:))),[direction.savedatadir_SWAF,filesep,'Blurred image.tiff'],'tiff')% output
[param.stw_coarse,param.stf_coarse] = SpectrumSeparation(param);% filter operation

%% ——————————————————————————————————————Construct P matrix here—————————————————————————————————————————————%
a = param.gama*param.Gexc*param.Texc/(2*param.Lpe);
b = param.gama*param.Gexc*param.Texc/2;
c = param.gama*param.Gexc*param.Texc*param.Lpe/8;
d = param.gama*param.Gacq;
%-------define steps------------%
step_number=40;
y_step=param.Lpe/param.fft_number;
y_step_step=y_step/step_number;
t_step=param.Tacq/param.nphase;
for x=1:param.nphase
    current_t=(x)*t_step;
    for y=1:param.fft_number
        current_water_y=-param.Lpe/2+(y-1)*y_step;
        temp_water_re=0;
        temp_water_im=0;
        %fat
        current_fat_y=-param.Lpe/2+(y-1)*y_step-param.wcs/(param.gama*param.Gexc);
        temp_fat_re=0;
        temp_fat_im=0;
        for z=1:step_number
            current_water_angle=-a*current_water_y*current_water_y+b*current_water_y-c-param.sign*pi/2+d*current_t*current_water_y;
            temp_water_re=temp_water_re+y_step_step*cos(current_water_angle);
            temp_water_im=temp_water_im+y_step_step*sin(current_water_angle);
            current_water_y=current_water_y+y_step_step;
            %fat
            current_fat_angle=-a*current_fat_y*current_fat_y+b*current_fat_y-param.Texc*param.wcs*current_fat_y/param.Lpe-c-param.sign*pi/2+d*current_t*current_fat_y+param.wcs*current_t;
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
param.P_matrix_w=P_water_re+1i*P_water_im;
P_water=param.P_matrix_w;
[U,S,V] = svd(P_water);
param.P_matrix_w=U*eye(size(S))*V';
% figure,subplot(1,2,1),imagesc(abs(P_water));title('P water abs');
% subplot(1,2,2),imagesc(angle(P_water));title('P water angle');
%fat
param.P_matrix_f=P_fat_re+1i*P_fat_im;
P_fat=param.P_matrix_f;
[U,S,V] = svd(P_fat);
param.P_matrix_f=U*eye(size(S))*V';
% figure,subplot(1,2,1),imagesc(abs(P_fat));title('P fat abs');
% subplot(1,2,2),imagesc(angle(P_fat));title('P fat angle');

param.P_matrix=[param.P_matrix_w, param.P_matrix_f];
%% —————————————————————————————————————————————Construct artifact matrix of water————————————————————————————————————%
param.undsam_fac=param.pw*param.Texc/param.nphase;
param.a=-param.gama*param.Gexc*param.Texc/(2*param.Lpe);
param.b=param.gama*param.Gexc*param.Texc/2;
param.dlt_y=param.Lpe/param.undsam_fac;
param.keymatrix_w=diag(ones(1,param.fft_number));
for number=1:1:round(param.undsam_fac-1)
    param.dlt_y=param.Lpe/param.undsam_fac*number;
    for yindex=1:param.fft_number
        current_y=-param.Lpe/2+(yindex-1)*y_step;
        temp_left_re=0;
        temp_left_im=0;
        %right
        temp_right_re=0;
        temp_right_im=0;
        for zindex=1:step_number
            current_left=-a*param.dlt_y^2-b*param.dlt_y+2*a*current_y*param.dlt_y;
%             current_left=-a*(param.dlt_y^2-current_y^2)+b*(param.dlt_y-current_y)
            temp_left_re=temp_left_re+y_step_step*cos(current_left);
            temp_left_im=temp_left_im+y_step_step*sin(current_left);
            %right
            current_right=-a*param.dlt_y^2+b*param.dlt_y-2*a*current_y*param.dlt_y;
%             current_right=-a*(param.dlt_y^2-current_y^2)+b*(-param.dlt_y-current_y)
            temp_right_re=temp_right_re+y_step_step*cos(current_right);
            temp_right_im=temp_right_im+y_step_step*sin(current_right);
            %
            current_y=current_y+y_step_step;
        end
        left_re(yindex)=temp_left_re/y_step;
        left_im(yindex)=temp_left_im/y_step;
        %right
        right_re(yindex)=temp_right_re/y_step;
        right_im(yindex)=temp_right_im/y_step;
    end
    left_roh=left_re+1i*left_im;
    right_roh=right_re+1i*right_im;
    
    param.als_num=round(param.fft_number*(1-number/param.undsam_fac));
    right=[ones(1,param.als_num),zeros(1,(param.fft_number-param.als_num))];
    left=[zeros(1,(param.fft_number-param.als_num)),ones(1,param.als_num)];
    param.dltroh_left=diag(left_roh)*circshift(diag(left),[0,param.als_num]);
    param.dltroh_right=diag(right_roh)*circshift(diag(right),[0,-param.als_num]);
    param.keymatrix_w=param.keymatrix_w+param.dltroh_left+param.dltroh_right;
end
%% ——————————————————————————————————————————————————————————Construct artifact matrix of fat——————————————————————%
param.keymatrix_f=diag(ones(1,param.fft_number));
for number=1:1:round(param.undsam_fac-1)
    param.dlt_y=param.Lpe/param.undsam_fac*number;
    for yindex=1:param.fft_number
        current_y=-param.Lpe/2+(yindex-1)*y_step-param.wcs/(param.gama*param.Gexc);
        temp_left_re=0;
        temp_left_im=0;
        %right
        temp_right_re=0;
        temp_right_im=0;
        for zindex=1:step_number
            current_left=-a*param.dlt_y^2-b*param.dlt_y+2*a*current_y*param.dlt_y + param.wcs*param.Texc*param.dlt_y/param.Lpe;
%             current_left=-a*param.dlt_y^2-b*param.dlt_y+2*a*current_y*param.dlt_y-(-a*param.dlt_y^2-b*param.dlt_y+2*a*current_y*param.dlt_y);
            temp_left_re=temp_left_re+y_step_step*cos(current_left);
            temp_left_im=temp_left_im+y_step_step*sin(current_left);
            %right
            current_right=-a*param.dlt_y^2+b*param.dlt_y-2*a*current_y*param.dlt_y - param.wcs*param.Texc*param.dlt_y/param.Lpe;
            temp_right_re=temp_right_re+y_step_step*cos(current_right);
            temp_right_im=temp_right_im+y_step_step*sin(current_right);
            %
            current_y=current_y+y_step_step;
        end
        left_re(yindex)=temp_left_re/y_step;
        left_im(yindex)=temp_left_im/y_step;
        %right
        right_re(yindex)=temp_right_re/y_step;
        right_im(yindex)=temp_right_im/y_step;
    end
    left_roh=left_re+1i*left_im;
    right_roh=right_re+1i*right_im;
    
    param.als_num=round(param.fft_number*(1-number/param.undsam_fac));
    right=[ones(1,param.als_num),zeros(1,(param.fft_number-param.als_num))];
    left=[zeros(1,(param.fft_number-param.als_num)),ones(1,param.als_num)];
    param.dltroh_left=diag(left_roh)*circshift(diag(left),[0,param.als_num]);
    param.dltroh_right=diag(right_roh)*circshift(diag(right),[0,-param.als_num]);
    param.keymatrix_f=param.keymatrix_f+param.dltroh_left+param.dltroh_right;
end
param.keymatrix=zeros(2*param.fft_number, 2*param.fft_number);
param.keymatrix(1:param.fft_number,1:param.fft_number)=param.keymatrix_w;
param.keymatrix(param.fft_number+1:end,param.fft_number+1:end)=param.keymatrix_f;

%% water fat separation——————————————————————————————————————————————————————————————————————————————%
FSR = p1FSR(param);
result_without_CS = FSR'*param.st;
% figure,imshow(abs(result_without_CS),[]);title('Super-resovled result');

FSRw = p1FSRw(param);
water_without_CS = FSRw'*param.stw_coarse/norm(param.st);
% figure,imshow(flipud(abs(water_without_CS)),[]);title('water result')
% imwrite(abs(water_without_CS)/max(abs(water_without_CS(:))),[param.savedatadir, filesep, 'water_prior.tiff'],'tiff')

FSRf = p1FSRf(param);
fat_without_CS = FSRf'*param.stf_coarse/norm(param.st);
% figure,imshow(flipud(abs(fat_without_CS)),[]);title('fat result');
% imwrite(flipud(abs(fat_without_CS)/max(abs(fat_without_CS(:))))/1.15,[param.savedatadir, filesep, 'fat_prior.tiff'],'tiff')

w_factor = 1;
water_w_ini = FSRf'*param.stw_coarse;
% figure,imshow(abs(water_w_ini),[]);
water_w_ini = abs(water_w_ini);
water_w = water_w_ini/max(water_w_ini(:));
% imwrite(flipud(abs(water_w)/max(abs(water_w(:)))*w_factor),[param.savedatadir, filesep, 'water_w.tiff'],'tiff')

water_f_ini = FSRw'*param.stf_coarse;
% figure,imshow(abs(water_f_ini),[]);
water_f_ini = abs(water_f_ini);
water_f = water_f_ini/max(water_f_ini(:));
% imwrite(flipud(abs(water_f)/max(abs(water_f(:)))*w_factor),[param.savedatadir, filesep, 'fat_w.tiff'],'tiff')

roh_weight = [water_f;water_w];
% figure,imshow(abs(roh_weight),[]);

%%
% adjust parameters
nmse1=1e-3; % regularization parameter
nmse2=1.5e-3; % regularization parameter
w_factor2 = 0.01; % scaling factor
%%%%%
CSparam=init;
CSparam.BW=roh_weight;
% CSparam.BW(param.fft_number:end,:)=CSparam.BW(param.fft_number:end,:)*1;
gaussianFilter = fspecial('gaussian', [3, 3], 10);
CSparam.BW = imfilter(abs(CSparam.BW), gaussianFilter, 'circular', 'conv');
CSparam.BW = abs(CSparam.BW);
CSparam.BW=CSparam.BW/max(CSparam.BW(:))*w_factor2;
% figure(500),imshow(abs(CSparam.BW),[]);title('W');
% CSparam.BW=1;
 
%%
CSparam.FT=FSR;
CSparam.data=param.st/norm(param.st);
CSparam.Itnlim=20;
CSparam.TV=TVOP;
CSparam.XFM=1;
CSparam.xfmWeight=nmse1;
CSparam.TVWeight=nmse2;
x=[water_without_CS;fat_without_CS];

for n=1:1:13
    x=fnlCg_SEED(x,CSparam);
%     figure(100), imshow(abs(x),[]), drawnow
end

%%
result_with_CS = x;
% figure,imshow(flipud(abs(result_with_CS)),[]);title('Edge deghosting result');

fat_res=result_with_CS(param.fft_number+1:end,:);
% figure,imshow(flipud(abs(fat_res)),[]);title('SWAF fat result');
fat_res=flipud(fat_res);
% imwrite(abs(fat_res)/max(abs(fat_res(:))),[direction.savedatadir_SWAF, filesep, 'fat_wcoeff',num2str(w_factor2),'_TV',num2str(nmse2),'.tiff'],'tiff')

water_res=result_with_CS(1:param.fft_number,:);
% figure,imshow(flipud(abs(water_res)),[]);title('SWAF water result')
water_res=flipud(water_res);
% imwrite(abs(water_res)/max(abs(water_res(:))),[direction.savedatadir_SWAF, filesep, 'water_wcoeff',num2str(w_factor2),'_TV',num2str(nmse2),'.tiff'],'tiff')
% 
mix_res=result_with_CS(param.fft_number+1:end,:)+result_with_CS(1:param.fft_number,:);
% figure,imshow(flipud(abs(mix_res)),[]);title('SWAF mix result')
% mix_save=flipud(mix_res);
% maxi=max(max(abs(water_res)));
time_SWAF=toc;
% save reconstruction results
save([direction.savedatadir_SWAF,filesep,'SWAF_fat_result.mat'],'fat_res')
save([direction.savedatadir_SWAF,filesep,'SWAF_water_result.mat'],'water_res')
end

