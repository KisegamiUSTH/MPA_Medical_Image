function gmsd_value = gmsd(input_img, enhanced_img)
    % Convert images to grayscale if they are not already
    if size(input_img, 3) == 3
        input_img = rgb2gray(input_img);
    end
    
    if size(enhanced_img, 3) == 3
        enhanced_img = rgb2gray(enhanced_img);
    end
    
    % Define a Gaussian kernel for local similarity
    gaussian_window = fspecial('gaussian', 3, 1.0);

    % Calculate gradient magnitude for both input and enhanced images
    [input_gradient_x, input_gradient_y] = gradient(double(input_img));
    [enhanced_gradient_x, enhanced_gradient_y] = gradient(double(enhanced_img));

    input_magnitude = sqrt(input_gradient_x.^2 + input_gradient_y.^2);
    enhanced_magnitude = sqrt(enhanced_gradient_x.^2 + enhanced_gradient_y.^2);

    % Apply Gaussian filter
    input_magnitude = imfilter(input_magnitude, gaussian_window, 'replicate');
    enhanced_magnitude = imfilter(enhanced_magnitude, gaussian_window, 'replicate');

    % Gradient magnitude similarity calculation
    quality_map = (2 * input_magnitude .* enhanced_magnitude + 0.0001) ./ ...
                  (input_magnitude.^2 + enhanced_magnitude.^2 + 0.0001);

    % Calculate standard deviation of the quality map (GMSD)
    gmsd_value = std(quality_map(:));
end
