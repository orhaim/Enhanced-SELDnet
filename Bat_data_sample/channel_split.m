clear all; clc;

%Fs for SledNet data
Fs_orig = 44100;

%% wav file read
[x,fs] = audioread('merged_all_20.wav');
[m, n] = size(x); %gives dimensions of array where n is the number of channels

%% Channel 1

c1 = x(:,7)+x(:,44)+x(:,10)+x(:,43)+x(:,16)+x(:,11)+x(:,13)+x(:,14)+...
    x(:,30)+x(:,42)+x(:,29)+x(:,41)+x(:,12)+x(:,31)+x(:,15)+x(:,40) + x(:,8);
[m_c1, n_c1] = size(c1); 
peakAmp_c1 = max(abs(c1)); 
c1 = c1/peakAmp_c1;

peak1 = max(abs(x(:,7)));    
peak2 = max(abs(x(:,44)));
peak3 = max(abs(x(:,10)));
peak4 = max(abs(x(:,43)));
peak5 = max(abs(x(:,16)));
peak6 = max(abs(x(:,11)));
peak7 = max(abs(x(:,13)));
peak8 = max(abs(x(:,14)));
peak9 = max(abs(x(:,30)));
peak10 = max(abs(x(:,42)));
peak11 = max(abs(x(:,29)));
peak12 = max(abs(x(:,41)));
peak13 = max(abs(x(:,12)));
peak14 = max(abs(x(:,31)));
peak15 = max(abs(x(:,15)));
peak16 = max(abs(x(:,40)));
peak17 = max(abs(x(:,8)));

maxPeak_c1 = max([peak1 peak2 peak3 peak4 peak5 peak6 peak7 peak8 peak9 peak10...
                peak11 peak12 peak13 peak14 peak15 peak16 peak17]);
c1 = c1*maxPeak_c1;

%% Channel 2

c2 = x(:,28)+x(:,39)+x(:,27)+x(:,17)+x(:,22)+x(:,26)+x(:,38)+x(:,3)+...
    x(:,18) + x(:,8);
[m_c2, n_c2] = size(c2); 
peakAmp_c2 = max(abs(c2)); 
c2 = c2/peakAmp_c2;

peak1 = max(abs(x(:,28)));    
peak2 = max(abs(x(:,39)));
peak3 = max(abs(x(:,27)));
peak4 = max(abs(x(:,17)));
peak5 = max(abs(x(:,22)));
peak6 = max(abs(x(:,26)));
peak7 = max(abs(x(:,38)));
peak8 = max(abs(x(:,3)));
peak9 = max(abs(x(:,18)));
peak10 = max(abs(x(:,8)));

maxPeak_c2 = max([peak1 peak2 peak3 peak4 peak5 peak6 peak7 peak8 peak9 peak10]);
c2 = c2*maxPeak_c2;

%% Channel 3

c3 = x(:,1)+x(:,2)+x(:,23)+x(:,35)+x(:,21)+x(:,24)+x(:,25)+x(:,20)+...
    x(:,36)+x(:,37)+x(:,19) + x(:,8);
[m_c3, n_c3] = size(c3); %gives dimensions of array where n is the number of channels
peakAmp_c3 = max(abs(c3)); 
c3 = c3/peakAmp_c3;

peak1 = max(abs(x(:,1)));    
peak2 = max(abs(x(:,2)));
peak3 = max(abs(x(:,23)));
peak4 = max(abs(x(:,35)));
peak5 = max(abs(x(:,21)));
peak6 = max(abs(x(:,24)));
peak7 = max(abs(x(:,25)));
peak8 = max(abs(x(:,20)));
peak9 = max(abs(x(:,36)));
peak10 = max(abs(x(:,37)));
peak11 = max(abs(x(:,19)));
peak12 = max(abs(x(:,8)));

maxPeak_c3 = max([peak1 peak2 peak3 peak4 peak5 peak6 peak7 peak8 peak9 peak10...
    peak11 peak12]);
c3 = c3*maxPeak_c3;

%% Channel 4

c4 = x(:,9)+x(:,34)+x(:,4)+x(:,33)+x(:,5)+x(:,46)+x(:,32)+x(:,45)+...
    x(:,6) + x(:,8);
[m_c4, n_c4] = size(c4); %gives dimensions of array where n is the number of channels
peakAmp_c4 = max(abs(c4)); 
c4 = c4/peakAmp_c4;

peak1 = max(abs(x(:,9)));    
peak2 = max(abs(x(:,34)));
peak3 = max(abs(x(:,4)));
peak4 = max(abs(x(:,33)));
peak5 = max(abs(x(:,5)));
peak6 = max(abs(x(:,46)));
peak7 = max(abs(x(:,32)));
peak8 = max(abs(x(:,45)));
peak9 = max(abs(x(:,6)));
peak10 = max(abs(x(:,8)));

maxPeak_c4 = max([peak1 peak2 peak3 peak4 peak5 peak6 peak7 peak8 peak9 peak10]);
c4 = c4*maxPeak_c4;

%%
%% wav file edit
% cut 4 channels from the original data
wv_4ch = [c1 c2 c3 c4];

% resample the 4 channels data
wv_4ch_resamp = resample(wv_4ch,Fs_orig,fs);

%% new wav file write
audiowrite('wv_4c_resampled.wav',wv_4ch_resamp,Fs_orig);
