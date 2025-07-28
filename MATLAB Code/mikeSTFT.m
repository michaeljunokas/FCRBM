function f = mikeSTFT( data, fftSz, hop, taperingWindow )
% Performs an STFT: a forward transform if the input is real or an inverse
% transform otherwise
%
% data = samples to be transformed (i.e. audioread('a.wav')
% fftSz = number of samples over which FFT is computed; fft window size
% hop = how  frequently to perform FFT (i.e. fftSz = 256 and hop = 
%   256, no overlap; fftSz = 256 and hop = 128, 50% overlap
% taperingWindow = type of tapering window to use; should be same size as
%   fftSz (i.e. fftSz = 256, taperingWindow could be hamming(256))

if isreal(data)
    
    y = zeros(fftSz, ceil((length(data)-fftSz)/hop));
    si = 1:fftSz;
    
    for i = 0:hop:length(data)-fftSz-1
        y(:,i/hop+1) = data(i+si);
    end
    % apply window
    y = bsxfun(@times,y,taperingWindow); 
    % perform STFT
    f = fft(y,[],1);
    % only need bottom half of frequencies, 0hz to Nyquist
    f = f(1:end/2+1,:); 
    imagesc(abs(f).^.35), axis xy % why .35?
    
else
    % add back frequencies that were removed
    f = [data;conj(data(end-1:-1:2,:))]; 
    % perform IFFT
    data = ifft(f,[],1); % adding 'real'....
    % overlap and add
    f = zeros(1,size(data,2)*hop+fftSz);
    for i = 1:size(data,2)
        f((i-1)*hop+(1:fftSz)) = f((i-1)*hop+(1:fftSz)) + (taperingWindow.*data(:,i))';
    end
end

