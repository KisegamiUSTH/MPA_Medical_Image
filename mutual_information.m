function MI = mutual_information(image1, image2)
    % Convert images to 256 gray levels (if not already)
    image1 = im2uint8(image1);
    image2 = im2uint8(image2);

    % Joint histogram
    jointHist = joint_histogram(image1, image2);
    jointHist = jointHist / numel(image1);  % Normalize to get probabilities

    % Marginal histograms
    hist1 = sum(jointHist, 2);  % Sum over columns
    hist2 = sum(jointHist, 1);  % Sum over rows

    % Compute the mutual information
    MI = 0;
    for i = 1:256
        for j = 1:256
            if jointHist(i, j) > 0
                MI = MI + jointHist(i, j) * log2(jointHist(i, j) / (hist1(i) * hist2(j)));
            end
        end
    end
end

function jointHist = joint_histogram(image1, image2)
    % Joint histogram for 256 intensity levels
    jointHist = zeros(256, 256);

    for i = 1:numel(image1)
        x = image1(i) + 1;
        y = image2(i) + 1;
        jointHist(x, y) = jointHist(x, y) + 1;
    end
end
