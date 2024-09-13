function enhanced_img = mpaenhancement(input_img)

    input_img = im2double(input_img);
    input_img(isinf(input_img) | isnan(input_img)) = 0;

    % Apply CLAHE
    clahe_img = adapthisteq(input_img);

    % Visualize CLAHE result in a new figure window
    figure();

    subplot(1, 2, 1);
    imshow(input_img, []);
    title('Input Image (Before CLAHE)');
    
    subplot(1, 2, 2);
    imshow(clahe_img, []);
    title('Enhanced Image (CLAHE)');

    % Apply Laplacian edge detection
    laplacianFilter = [0 1 0; 1 -4 1; 0 1 0];
    edge_img = imfilter(input_img, laplacianFilter, 'replicate');
    
    % Visualize Edge Detection result in a new figure window
    figure();
    subplot(1, 2, 1);
    imshow(input_img, []);
    title('Input Image (Before Edge Detection)');
    
    subplot(1, 2, 2);
    imshow(edge_img, []);
    title('Enhanced Image (Edge Detection)');

    % Apply DN-CNN denoising
    net = denoisingNetwork('dncnn');
    denoised_img = denoiseImage(input_img, net);

    % Visualize DN-CNN Denoising result in a new figure window
    figure();
    subplot(1, 2, 1);
    imshow(input_img, []);
    title('Input Image (Before DN-CNN Denoising)');
    
    subplot(1, 2, 2);
    imshow(denoised_img, []);
    title('Enhanced Image (DN-CNN Denoising)');

    [H, W] = size(input_img);

    % Define MPA parameters
    populationSize = 200;
    numGenerations = 100;
    lowerBound = 0;
    upperBound = 5;

    % Initialize MPA variables
    bestSolution = [];
    bestFitness = Inf;

    % Main MPA loop
    for generation = 1:numGenerations
        % Generate random solutions for beta parameters
        population = lowerBound + (upperBound - lowerBound) * rand(populationSize, 3);

        % Evaluate the fitness of each solution
        fitness = evaluateFitness(population, clahe_img, edge_img, denoised_img, input_img);

        % Find the best solution and its fitness
        [currentBestFitness, bestIndex] = min(fitness);

        % Display the best fitness for the current generation
        fprintf('Generation %d: Best Fitness = %f\n', generation, currentBestFitness);

        % Update the best solution if a better one is found
        if currentBestFitness < bestFitness
            bestFitness = currentBestFitness;
            bestSolution = population(bestIndex, :);
        end
    end

    % Apply the optimal parameters to enhance the image
    beta_1 = bestSolution(1);
    beta_2 = bestSolution(2);
    beta_3 = bestSolution(3);
    enhanced_img = beta_1 * clahe_img + beta_2 * edge_img + beta_3 * denoised_img;
end

function fitness = evaluateFitness(population, clahe_img, edge_img, denoised_img, input_img)
    numSolutions = size(population, 1);
    fitness = zeros(numSolutions, 1);
    [H, W] = size(input_img);
    M = max(input_img(:));

    for i = 1:numSolutions
        beta_1 = population(i, 1);
        beta_2 = population(i, 2);
        beta_3 = population(i, 3);

        I_T = beta_1 * clahe_img + beta_2 * edge_img + beta_3 * denoised_img;

        % Calculate fitness function components
        V = var(I_T(:));
        M_T = mean(I_T(:));
        E_1 = entropy(input_img);
        E_2 = entropy(I_T);
        G_1 = sum(sum(abs(input_img - mean(input_img(:)))))/(H*W);
        G_2 = sum(sum(abs(I_T - mean(I_T(:)))))/(H*W);
        PSNR = 10 * log10(M^2 / (sum(sum((I_T - input_img).^2)) / (H*W)));

        fitness(i) = V/M * ((E_1 - E_2) + (G_1 - G_2) / PSNR);
    end
end
