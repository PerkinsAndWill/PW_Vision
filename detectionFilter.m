%% Filter detections

function [filteredIdx, filteredCentroids] = detectionFilter (ROI, maxRatio, maxArea, bbox, centroids)

%Calculate Ratio, Area
w = bbox(:,3);
h = bbox(:,4);
x = int8(centroids(:,1));
y = int8(centroids(:,2));

ratio = double(w) ./ double(h);
area = h .* w;

%Filter out the bboxes

badBbox = ratio > maxRatio;
badBbox = badBbox | area > maxArea;

%Filter out centroids if not in ROI
badCentroid = int32.empty();

for i = 1:length(centroids)
    if ROI(x(i),y(i))  == 1
        badCentroid = [badCentroid 1];
    else
        badCentroid = [badCentroid 0];
    end
end

filter = badBbox | badCentroid';
% disp(bbox);
% disp(badBbox);

filteredIdx = bbox(logical(~filter), :);
filteredCentroids = centroids(logical(~filter), :);

% disp(filteredIdx);

end

