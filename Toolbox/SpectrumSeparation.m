function [stw,stf] = SpectrumSeparation(param)
% One-dimensional spectrum separation for SPEN signal
SPP_position = linspace(param.Lpe/2,-param.Lpe/2,param.nphase).';
add_term = -param.sign*param.gama*param.chirp*param.Gexc*param.Texc/param.Lpe;
add_phase = exp(1i*add_term*SPP_position.^2/2);% remove the quadratic phase for separation
st = param.st.*repmat(add_phase,[1,param.fft_number]);
% figure;imshow(angle(param.st),[]);title('remove st');
% figure;imshow(angle(st),[]);title('remove');

% 1D FFT
st_fft = fft(st,[],1);
st_fftshift = fftshift(st_fft,1);
% normalized_st_fftshift=st_fftshift/(max(st_fftshift(:)));
% figure(101);
% for m = 1:1:size(normalized_st_fftshift,2)
% 	plot(abs(normalized_st_fftshift(:,m)),'LineWidth',2);
% 	hold on
% end
% axis([1,64,0,1]);title('1D mixed spectra')
% set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',20)
% set(gca,'linewidth',3);
%     box off;
% grid on;
%     axis off;
% axis([0 1 0 70])
% print(gcf,'-dtiff',[param.savedatadir, filesep, 'mix spectrum.tif']);
    
%     set(gca,'FontSize',20);
% define the position of fat peak (also can use 'ginput' manually selected
% for different samples)
peak_index = 57;
% [peak_index,my] = ginput(1);% manually selected
move_num = round(peak_index - param.nphase/2);
% filter operation along frequency domain
lpf = lpfilter('btw',64,1,10,1.5);
lpf = fftshift(lpf);
lpf = circshift(lpf,[move_num 0]);
% figure, plot(lpf,'LineWidth',5);title('filter for fat')
% box off;
% axis([1,64,0,1]);
% set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',20)
% set(gca,'linewidth',3);
% print(gcf,'-dtiff',[param.savedatadir, filesep, 'filter.tif']);
filter_mat = repmat(lpf,[1,param.fft_number]);
st_filtered = st_fftshift.*filter_mat; 
% figure(102);
% for m = 1:1:size(st_filtered,2)
% 	plot(abs(st_filtered(:,m)),'LineWidth',2);
% 	hold on
% end
% box off;
% grid on;title('fat spectra')
% xlim([1, 64]);
% axis([1,64,0,1]);
% set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',20)
% set(gca,'linewidth',3);
% print(gcf,'-dtiff',[param.savedatadir, filesep, 'fat spectra.tif']);
% 1D IFFT to get blurry fat signal (before SR reconstruction)
stf_ifftshift = ifftshift(st_filtered,1);
stf_ifft = ifft(stf_ifftshift,[],1);
stf = stf_ifft;
stw = st - stf;

stw_fft = fft(stw,[],1);
stw_fftshift = fftshift(stw_fft,1);
% figure(103);
% for m = 1:1:size(stw_fftshift,2)
% 	plot(abs(stw_fftshift(:,m)),'LineWidth',2);
% 	hold on
% end
% box off;
% grid on;title('water spectra')
% xlim([1, 64]);
% % axis([1,64,0,1]);
% set(gca,'Fontname','Times New Roman','FontWeight','bold','FontSize',20)
% set(gca,'linewidth',3);
% % print(gcf,'-dtiff',[param.savedatadir, filesep, 'water spectra.tif']);

stf = stf./repmat(add_phase,[1,param.fft_number]);
% figure;imshow(abs(stf),[]);title('fat st');
% get blurry water signal (before SR reconstruction)
stw = stw./repmat(add_phase,[1,param.fft_number]);
% figure;imshow(abs(stw),[]);title('water ori st');
end