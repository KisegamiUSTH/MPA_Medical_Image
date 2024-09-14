% main2.m
clc;
clear;
close all;

% Read DICOM Image
[dicomFile, dicomPath] = uigetfile('*.dcm', 'Select a DICOM image');
dicom_img = dicomread(fullfile(dicomPath, dicomFile));

% Convert DICOM image to grayscale if needed
if size(dicom_img, 3) == 3  % Check if it's RGB
    dicom_img = rgb2gray(dicom_img);
end

% Normalize the image to [0, 1] for processing
dicom_img = mat2gray(dicom_img);

% Call the enhance_images function to enhance the image using different methods
enhanced_images = enhance_images(dicom_img);

% Initialize a struct to hold evaluation metrics
evaluation_metrics = struct('MLI', [], 'CI', [], 'Entropy', [], 'AG', []);

% Calculate evaluation metrics for each enhancement method
methods = fieldnames(enhanced_images);
for i = 1:numel(methods)
    method_name = methods{i};
    enhanced_img = enhanced_images.(method_name);
    
    % Ensure the image is of type double for calculations
    enhanced_img = im2double(enhanced_img);

    % Mean Light Intensity (MLI)
    evaluation_metrics(i).Method = method_name;
    evaluation_metrics(i).MLI = mean(enhanced_img(:));
    
    % Contrast Index (CI) as Standard Deviation of Intensities
    evaluation_metrics(i).CI = std(enhanced_img(:));
    
    % Entropy
    evaluation_metrics(i).Entropy = entropy(enhanced_img);
    
    % Average Gradient (AG)
    [Gx, Gy] = imgradientxy(enhanced_img);
    evaluation_metrics(i).AG = mean(sqrt(Gx.^2 + Gy.^2), 'all');
end

% Display Evaluation Results
disp('Evaluation Metrics:');
disp(struct2table(evaluation_metrics));

% Plot each enhancement result for comparison
figure;
for i = 1:numel(methods)
    subplot(2, 3, i);
    imshow(enhanced_images.(methods{i}), []);
    title(methods{i});
end
