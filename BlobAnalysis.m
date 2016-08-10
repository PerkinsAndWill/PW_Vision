%% Setup Video Player, Foreground detector

video = vision.VideoFileReader('Test_1.avi');

foregroundDetector = vision.ForegroundDetector('NumTrainingFrames', 75, ...
        'LearningRate', .01, ...
        'NumGaussians', 3);

blobs = vision.BlobAnalysis('CentroidOutputPort', true, ...
       'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobArea', 100);
   
   
frame = step(video);
   
%% Global Variables

% create an empty array of tracks
tracks = struct(...
    'id', {}, ...
    'color', {}, ...
    'bbox', {}, ...
    'score', {}, ...
    'kalmanFilter', {}, ...
    'age', {}, ...
    'totalVisibleCount', {}, ...
    'consecutiveInvisibleCount', {}, ...
    'confidence', {}, ...
    'predPosition', {});

% Set other global options

% dogRatio = .8;

peopleRatio = .75;

ROI = roipoly (frame);

vPlayer = vision.VideoPlayer();
fPlayer = vision.VideoPlayer();
   
% opticFlow = opticalFlowLK();

%% Loop through the video

while ~isDone(video)
    
   frame = step(video);
   mask = foregroundDetector.step(frame);
   
%% Morphological Image Cleaning
   
%    cleanMask = imopen(mask, strel('Disk',1));
   cleanMask = imopen(mask, strel('rectangle', [3,3]));
   cleanMask = imclose(cleanMask, strel('rectangle', [15, 15])); 
   cleanMask = imfill(cleanMask, 'holes');
   
%    flow = estimateFlow(opticFlow, cleanMask);

%% Blob detection
   [centroids, bboxes] = blobs.step(cleanMask);
   
%    [peoples, scores] = detectPeopleACF(frame,...
%             'Model','caltech',...
%             'NumScaleLevels', 32);
   
   if ~isempty(bboxes)

%% Filter detections based off of size, aspect ratio, location
%         [bboxDogs, centDogs] = detectionFilter(dogRatio, 400, bboxes, centroids);
        [bboxPeople, centPeople] = detectionFilter(ROI, peopleRatio, 1500, bboxes, centroids);
%    frame = insertShape(frame, 'rectangle', bboxDogs, 'Color', 'green');
        frame = insertShape(frame, 'rectangle', bboxPeople, 'Color', 'red');
%    cleanMask = insertShape(cleanMask, 'rectangle', bboxPeople, 'Color', 'green');
   


%% Predict track Locations
        trackPredictions(tracks);

%% Assign Detections to Tracks
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        
        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);



%    imshow(frame);
%    hold on;
%    plot(flow,'DecimationFactor', [5,5], 'ScaleFactor', 25);
%    hold off;
   end
   
   %% Update assigned tracks
   numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);
            
            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;
            
            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
   
   %% update Unassigned Tracks
   for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
   end
   
   %% delete tracks
   if isempty(tracks)
            return;
        end
        
        invisibleForTooLong = 20;
        ageThreshold = 8;
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % Delete lost tracks.
        tracks = tracks(~lostInds);
   
   %% Create New Tracks
   
   centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        
        for i = 1:size(centroids, 1)
            
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            
            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [50, 200], [25, 250], 500);
            
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;
            
            % Increment the next id.
            nextId = nextId + 1;
        end
%Continue to next frame
   step(vPlayer, frame);
   step(fPlayer, cleanMask);
end

%% Release the videoplayers

release(vPlayer);
release(fPlayer);