function enhanced_img = mpaenhancement(input_img)

    input_img = im2double(input_img);
    input_img(isinf(input_img) | isnan(input_img)) = 0;

    % Apply CLAHE with adjusted parameters for higher contrast
    clahe_img = adapthisteq(input_img);

    % Apply Laplacian edge detection
    laplacianFilter = [0 1 0; 1 -4 1; 0 1 0];
    edge_img = imfilter(input_img, laplacianFilter, 'replicate');

    % Apply DN-CNN denoising
    net = denoisingNetwork('dncnn');
    denoised_img = denoiseImage(input_img, net);

    [H, W] = size(input_img);

    % Define MPA parameters with increased range
    populationSize = 50;
    numGenerations = 50;
    lowerBound = 0;
    upperBound = 1.5; % Should always be smaller than 1.7

    % Initialize MPA variables
    bestSolution = [];
    bestFitness = -Inf; % Set to negative infinity to prefer higher values
    bestMetrics = []; % Store the metrics (E1, E2, V, G1, G2, PSNR) for the best solution

    % Main MPA loop
    for generation = 1:numGenerations
        % Generate random solutions for beta parameters
        population = lowerBound + (upperBound - lowerBound) * rand(populationSize, 3);

        % Evaluate the fitness of each solution
        [fitness, metrics] = evaluateFitness(population, clahe_img, edge_img, denoised_img, input_img, generation);

        % Find the best solution and its fitness
        [currentBestFitness, bestIndex] = max(fitness); % Use max to prefer higher fitness

        % Update the best solution if a better one is found
        if currentBestFitness > bestFitness % Change to greater than
            bestFitness = currentBestFitness;
            bestSolution = population(bestIndex, :);
            bestMetrics = metrics(bestIndex, :); % Store the metrics for the best solution

            % Print the best solution for the current generation
            fprintf('Generation %d: Best Fitness = %f, Best Solution = [beta_1: %f, beta_2: %f, beta_3: %f]\n', ...
                generation, bestFitness, bestSolution(1), bestSolution(2), bestSolution(3));
        end
    end

    % Print the final best solution and its metrics after all generations
    fprintf('Final Best Solution: [beta_1: %f, beta_2: %f, beta_3: %f] with Best Fitness = %f\n', ...
        bestSolution(1), bestSolution(2), bestSolution(3), bestFitness);
    fprintf('Final Metrics: E1 = %f, E2 = %f, V = %f, G1 = %f, G2 = %f, PSNR = %f\n', ...
        bestMetrics(1), bestMetrics(2), bestMetrics(3), bestMetrics(4), bestMetrics(5), bestMetrics(6));

    % Apply the optimal parameters to enhance the image with adjusted weights
    beta_1 = bestSolution(1);
    beta_2 = bestSolution(2);
    beta_3 = bestSolution(3);
    enhanced_img = beta_1 * clahe_img + beta_2 * edge_img + beta_3 * denoised_img;
end
function [fitness, metrics] = evaluateFitness(population, clahe_img, edge_img, denoised_img, input_img, generation)
    numSolutions = size(population, 1);
    fitness = zeros(numSolutions, 1);
    metrics = zeros(numSolutions, 6); % Store E1, E2, V, G1, G2, PSNR for each solution
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

        % Original fitness function
        fitness(i) = V/M * ((E_1 - E_2) + (G_1 - G_2) / PSNR);

        % Store the metrics for later use
        metrics(i, :) = [E_1, E_2, V, G_1, G_2, PSNR];

        % Print all the fitness function components
        %fprintf('Generation %d, Solution %d: V = %f, M = %f, E1 = %f, E2 = %f, G1 = %f, G2 = %f, PSNR = %f\n', ...
        %    generation, i, V, M, E_1, E_2, G_1, G_2, PSNR);
    end
end
