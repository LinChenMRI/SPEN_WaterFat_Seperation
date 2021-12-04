function res = NCGDA(P_matrix,st,xfmW,TVW,fft_number)
CSparam=init;
CSparam.FT=Pmatrix(P_matrix);
CSparam.data=st;
% tempnorm = norm(param.stw);
CSparam.Itnlim=20;
CSparam.TV=TVOP;
CSparam.XFM=1;
% CSparam.XFM=Wavelet('Daubechies',6,2);
CSparam.xfmWeight=xfmW; % The regularization parameters may need to be fine-tuned
CSparam.TVWeight=TVW;
x=zeros([fft_number,fft_number]);

for n=1:1:5
    x=fnlCg(x,CSparam);
%     figure(20), imshow(abs(x),[]), drawnow
end
res = x;
end