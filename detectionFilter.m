%% Filter detections

function [filteredIdx, filteredCentroids] = detectionFilter (maxRatio, maxArea, bbox, centroids)

%Calculate Ratio, Area
w = bbox(:,3);
h = bbox(:,4);

ratio = double(w) ./ double(h);
area = h .* w;

%Filter out the bboxes

badBbox = ratio > maxRatio;
badBbox = badBbox | area > maxArea;

% disp(bbox);
% disp(badBbox);

filteredIdx = bbox(logical(~badBbox), :);
filteredCentroids = centroids(logical(~badBbox), :);

% disp(filteredIdx);

end

