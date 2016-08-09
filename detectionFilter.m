%% Filter detections

function filteredIdx = detectionFilter (maxRatio, maxArea, bbox)

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

% disp(filteredIdx);

end

