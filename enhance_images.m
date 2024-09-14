function enhanced_images = enhance_images(input_img)
    % Convert input image to double precision
    input_img = im2double(input_img);

    % Store the original image (NE - No Enhancement)
    enhanced_images.NE = input_img;

    % CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    enhanced_images.CLAHE = adapthisteq(input_img);

    % EFF (Exposure Fusion Framework) - Simplified with single image
    % Typically, EFF works with multiple exposure images. Here, we simulate it.
    % Assuming a fake multi-exposure fusion example (for illustration)
    exposure_imgs = {input_img * 0.8, input_img * 1.2};  % Simulating multiple exposure levels
    enhanced_images.EFF = imfuse(exposure_imgs{1}, exposure_imgs{2}, 'blend', 'Scaling', 'joint');

    % EGIF (Effective Guided Image Filtering)
    radius = 5;  % Define the filter radius
    epsilon = 0.01^2;  % Define the regularization parameter
    enhanced_images.EGIF = imguidedfilter(input_img, 'NeighborhoodSize', radius, 'DegreeOfSmoothing', epsilon);

    % FFM (Fractional-order Fusion Model) - Simulated in a simplified form
    % Assuming the fractional-order fusion could be related to edge detection combined with contrast enhancement
    edge_img = edge(input_img, 'canny');
    enhanced_images.FFM = imfuse(input_img, edge_img, 'blend', 'Scaling', 'joint');

    % Display results in figure
    figure;
    subplot(2, 3, 1), imshow(enhanced_images.NE, []), title('Original (NE)');
    subplot(2, 3, 2), imshow(enhanced_images.CLAHE, []), title('CLAHE');
    subplot(2, 3, 3), imshow(enhanced_images.EFF, []), title('EFF');
    subplot(2, 3, 4), imshow(enhanced_images.EGIF, []), title('EGIF');
    subplot(2, 3, 5), imshow(enhanced_images.FFM, []), title('FFM');

end
