function [H,D] = lpfilter(type, M, N, D0, n)
%   LPFILTER Compute frequency domain lowpass filters.
%   D0 must be positive.
%   Valid values for TYPE, D0, and n are:
%   'ideal'     Ideal lowpass filter with cutoff frequency D0. n need not be
%               supplied.
%   'btw'       Butterworth lowpass filter of order n and cutoff D0. The
%               default value for n is 1.
%   'gaussian'  Gaussian lowpass filter with cutoff (standard deviation)
%               D0. n need not be supplied.

%   Compute the distances D(U, V).
[U, V] = dftuv(M, N);
D = sqrt(U.^2 + V.^2);

%   Filter computations.
switch type
    case 'ideal'
        H = double(D <= D0);
    case 'btw'
        if nargin == 4
            n = 1;
        end
        H = 1./(1 + (D./D0).^(2*n));
    case 'gaussian'
        H = exp(-(D.^2)./(2*(D0^2)));
    otherwise
        error('Unknown filter typ.')
end