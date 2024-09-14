clear all
clc

%% Load Input Image
input_img = im2double(imread('070T.png'));  % Changed to imread for PNG

%% Enhance Image Using MPA
tech_21 = mpaenhancement(input_img);

%% Display Input and Enhanced Images Side by Side
figure()

% Display Input Image
subplot(1, 2, 1);  % 1 row, 2 columns, plot 1
imshow(input_img, []);
title('Input Image');

% Display Enhanced Image
subplot(1, 2, 2);  % 1 row, 2 columns, plot 2
imshow(tech_21, []);
title('Enhanced Image');

%% Evaluate Metrics
% Call the evaluation function from a separate file
evaluate(input_img, tech_21);
