clear
clc
close all

%% Data Loading
dataDir = 'C:\Users\Human\kaist.ac.kr\Bomi Lee - InBody_Project\임상데이터\3. KAIST Korotkoff Sound\1. Data\Valid';
cd (dataDir);
addpath(genpath(dataDir));
filename='v2.csv';
sbp = 125; % auscultatory value
dbp = 90.5;
Data = load(filename); % one for the pressure and another for the sound

wavelet_save(Data,sbp,dbp)