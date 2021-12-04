function [time_DL_pre] = WFSR_DL_preprocessing_func(direction,Procpar)
% written by Xinran Chen, 11/21/2020
% this program is used to pre-process experiment data acquired from Varian system(.fid) to
% test sample for network to reconstruct (.Charles)
%% Get the experiment parameters�
tic
param.sign = 1;
param.fftNumber = 128; 
param.gama = 2.6752e8;
param.segment = 1;
param.chirp = 1; 
param.ifSVD = 0;
param.Tacq = 27.98e-3;
expand_num=128;
phase_num=64;
fre_num=64;
fftNumber=128;
type='rat_data';
% * Calculate the other paramters* �
param.Lpe = Procpar.lpe.Values*1e-2;
param.pw = Procpar.p1_bw.Values;
param.Texc = Procpar.pw1.Values;
param.nphase = Procpar.nphase.Values;
param.nread = Procpar.nread.Values/2; 
param.Gexc = param.sign*2*pi*param.pw/(param.gama*param.Lpe);
param.Gacq = -param.Gexc*param.Texc/param.Tacq;
param.R = param.gama*param.Gexc*param.Lpe/param.Texc;

% *Create the directory for saving the DL pre-processing results*
direction.savedatadir_DL = [direction.savedatadir, filesep,'DL_result'];
if exist(direction.savedatadir_DL,'file')==0
    mkdir(direction.savedatadir_DL);
end
%% pre-processing part
% *Load the original data in the SPEN sampling domain*��
[RE,IM,NP,NB,NT,HDR]=load_fid(direction.fid_dir,param.segment); 
fid=RE+1i*IM;
fid=fid(1+param.nread*2:end,:);
zerofillnum=(param.fftNumber-param.nread)/2;
for n = 1:1:NT
    fid_temp = reshape(fid(:,n),param.nread,param.nphase).'; 
    fid_temp(2:2:end,:) =  fliplr(fid_temp(2:2:end,:));
    fid_temp_zerofill = [zeros(param.nphase,zerofillnum),fid_temp,zeros(param.nphase,zerofillnum)];
    fid_temp_fftshift  = fftshift(fid_temp_zerofill,2);
    st_temp_fftshift = fft(fid_temp_fftshift,[],2);
    st_temp = fftshift(st_temp_fftshift,2);
    param.st(:,:,n) = st_temp;
end
% figure(1);imshow(angle(param.st),[]);axis image; title('Initial phase')
% flipud
param.st=flipud(param.st);
 % dimension
        st_zeros_fill = param.st;
        % remove the quadratic phase
        SPP_position = linspace(param.Lpe/2,-param.Lpe/2,param.nphase).';% the position of stationary phase points
        interpolation_position=linspace(param.Lpe/2,-param.Lpe/2,param.fftNumber).';% the position of interpolated points
        add_term = -param.sign*param.gama*param.chirp*param.Gexc*param.Texc/param.Lpe;
        add_phase = exp(1i*add_term*SPP_position.^2/2);
        st_remove = st_zeros_fill.*repmat(add_phase,[1,param.fftNumber]);
%         figure(2);imshow(angle(st_remove),[]);title('remove quadratic phase') % remove quadratic phase
        % interpolation between SPEN diemnsion
        for mm=1:1:param.fftNumber
            I1_interpolation(:,mm)=interp1(SPP_position,st_remove(:,mm),interpolation_position);
        end
        % normalize
        max_amp=max(max(abs(I1_interpolation(:))));
        I1_interpolation = I1_interpolation/max_amp;
%         imwrite(abs(I1_interpolation)/max(abs(I1_interpolation(:))),[direction.savedatadir_DL,filesep,'Input_for_Unet.tiff'],'tiff')% output

%         figure(3);imshow(abs(I1_interpolation),[]);axis image;title('Input for U-net')
%%     define output
        output(1,:,:) = real(I1_interpolation);% input 1 (real part)
        output(2,:,:) = imag(I1_interpolation);% input 2 (imaginary part)
        output(3,:,:) = zeros(expand_num,expand_num);% output 1: water 
        output(4,:,:) = zeros(expand_num,expand_num);% output 2: fat 
        filename=[direction.savedatadir_DL,filesep,type,'.Charles'];
        [fid,msg]=fopen(filename,'wb');
        fwrite(fid,output,'single');
        fclose(fid);    
        time_DL_pre = toc;
end

