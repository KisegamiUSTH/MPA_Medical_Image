function evaluate(input_img, tech_21)

    %% Mean Light Intensity (MLI)
    MLI = mean(tech_21(:));

    %% Contrast Index (CI)
    contrast_matrix = (tech_21 - mean(tech_21(:))).^2;
    CI = sqrt(sum(contrast_matrix(:)) / numel(tech_21));

    %% Entropy (E)
    E = entropy(tech_21);

    %% Average Gradient (AG)
    [Gx, Gy] = gradient(tech_21);
    AG = mean(mean(sqrt(Gx.^2 + Gy.^2)));

    %% Mutual Information (MI) between input and enhanced image
    MI = mutual_information(input_img, tech_21); 
      %% PSNR
    psnr_value = psnr(tech_21, input_img);
    
    %% SSIM
    ssim_value = ssim(tech_21, input_img);
    
    
    %% Variance of Laplacian
    laplacian_img = del2(tech_21);
    laplacian_var = var(laplacian_img(:));
    
    
    %% SNR
    signal_power = mean(tech_21(:).^2);
    noise_power = mean((tech_21(:) - input_img(:)).^2);
    snr_value = 10 * log10(signal_power / noise_power);
    
    
    %% GMSD (if applicable)
    gmsd_value = gmsd(input_img, tech_21);
    %% Display the evaluation results in the Command Window
    fprintf('Evaluation Metrics for Enhanced Image:\n');
    fprintf('Mean Light Intensity (MLI): %.4f\n', MLI);
    fprintf('Contrast Index (CI): %.4f\n', CI);
    fprintf('Entropy (E): %.4f\n', E);
    fprintf('Average Gradient (AG): %.4f\n', AG);
    fprintf('Mutual Information (MI): %.4f\n', MI);
    fprintf('Peak Signal-to-Noise Ratio (PSNR): %.4f dB\n', psnr_value);
    fprintf('Structural Similarity Index (SSIM): %.4f\n', ssim_value);
    fprintf('Variance of Laplacian: %.4f\n', laplacian_var);
    fprintf('Signal-to-Noise Ratio (SNR): %.4f dB\n', snr_value);
    fprintf('Gradient Magnitude Similarity Deviation (GMSD): %.4f\n', gmsd_value);

end
