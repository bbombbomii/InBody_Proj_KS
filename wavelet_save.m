function func = wavelet_save(Data,sbp,dbp)
%% Parameters
seg_len = 0.4;
ks_samf = 4000;
os_samf = 100;

%% Korotkoff Sound signal
ks = Data(:, 1)/4096*3.3;
ks = ks - mean(ks);
time_ks = (1:length(ks))/ks_samf;

%% Oscillometric signal
os = Data(:, 2)/100;
os = downsample(os, 40);
os = smooth(os);
%os = bandpass(os, [4 6.67], os_samf);
time_os = (1:length(os))/os_samf;
plot(time_os,os)
%% Segmentation based on oscillometric signal
[os_pks, os_locs] = findpeaks(os, 'MinPeakDistance', 60);
pk_n = length(os_locs);
t_sbp = mean(time_os(find(abs(os-sbp) <= 0.02)))
t_dbp = mean(time_os(find(abs(os-dbp) <= 0.02)))
seg_len/2*ks_samf
for a = 1:pk_n-1
    b = os_locs(a); % location of the middle peak in OS signal
    c = b/os_samf*ks_samf; % convert in KS signal
    if b-(seg_len/2*os_samf-1) < 0
        continue
    end
    os_seg = os(b-(seg_len/2*os_samf-1):b+seg_len/2*os_samf);
    time_os_seg = (1:length(os_seg))/os_samf;
    if a == 30
        figure()
        plot(time_os_seg,os_seg)
    end
    ks_seg = ks(c-(seg_len/2*ks_samf-1):c+seg_len/2*ks_samf);
    time_ks_seg = (1:length(ks_seg))/ks_samf;
    p= os(b); % pressure
    %% Wavelet Transform
    fb = cwtfilterbank;
    [cfs, f] = cwt(ks_seg, 'amor', ks_samf); % morse, amor, bump
    stack_time_cwt = [];
    stack_frq_cwt = [];
    
    for i = 1:size(cfs,1)
        stack_time_cwt(i, :) = time_ks_seg;
        stack_frq_cwt(i, 1:length(cfs)) = f(i);
    end
    %% Labeling
    if time_os(b) < t_sbp - 1
        label(a) = 0;
    elseif t_sbp - 1 <= time_os(b) && time_os(b) < t_sbp
            label(a) = time_os(b) - t_sbp + 1;
    elseif t_sbp <= time_os(b) && time_os(b) < t_dbp - 1
            label(a) = 1;
    elseif t_dbp - 1 <= time_os(b) && time_os(b) < t_dbp + 1
            label(a) = -1/2*time_os(b) + 1/2*t_dbp + 1/2;
    else
        label(a) = 0;
    end
    %% Save Image
    I(:,:,a) = mat2gray(abs(cfs));
    
    
end
size(I)
figure()
plot(label);
save('testdata.mat','I','label');
