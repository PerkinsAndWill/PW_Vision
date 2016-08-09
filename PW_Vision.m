
% PW Computer Vision 

function PW_Vision()

clear all;
close all
% set(0, 'DefaultFigureWindowStyle','docked');


%% Main Program
videoFile = 'Test_1.avi';

obj = setupSystemObjects(videoFile);

% Create an empty array of tracks.
tracks = initializeTracks(); 

% ID of the next track.
nextId = 1; 


% Set the global parameters.
option.ROI                  = [90, 100, 510, 320];   % A rectangle [x, y, w, h] that limits the processing area
option.scThresh             = 0.3;                   % A threshold to control the tolerance of error in estimating the scale of a detected pedestrian. 
option.gatingThresh         = 0.9;                   % A threshold to reject a candidate match between a detection and a track.
option.gatingCost           = 100;                   % A large value for the assignment cost matrix that enforces the rejection of a candidate match.
option.costOfNonAssignment  = 10;                    % A tuning parameter to control the likelihood of creation of a new track.
option.timeWindowSize       = 16;                    % A tuning parameter to specify the number of frames required to stabilize the confidence score of a track.
option.confidenceThresh     = 2;                     % A threshold to determine if a track is true positive or false alarm.
option.ageThresh            = 1;                     % A threshold to determine the minimum length required for a track being true positive.
option.visThresh            = 0.6;                   % A threshold to determine the minimum visibility value for a track being true positive.
option.invisibTooLong       = 5;                     % A threshold to determine if a track has gone off frame

while ~isDone(obj.reader);
    frame = step(obj.reader);
    
    [centroids, bboxes, mask, scores] = detectObjects(frame);
    
    predictNewLocationsOfTracks();    
    
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();    
    updateUnassignedTracks();    
    deleteLostTracks();    
    createNewTracks();
    
    displayTrackingResults();

    % Exit the loop if the video player figure is closed by user.     
    cont = ~isDone(obj.reader) && isOpen(obj.videoPlayer);
end

%% Gaussian Filter
% 
% hsizeh = 30
% sigmah = 6
% h = fspecial('log', hsizeh, sigmah)
% subplot(121); imagesc(h)
% subplot(122); mesh(h)
% colormap(jet)

%% Create System Objects
% Create System objects used for reading the video frames, detecting
% foreground objects, and displaying results.

    function obj = setupSystemObjects(videoFile)
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.
        
        % Create a video file reader.
        obj.reader = vision.VideoFileReader(videoFile, 'VideoOutputDataType', 'uint8');
        
        % Create two video players, one to display the video,
        % and one to display the foreground mask.
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        
        % Create System objects for foreground detection and blob analysis
        
        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background. 
        
        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 30, 'MinimumBackgroundRatio', 0.75);
        
        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.
        
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
           'AreaOutputPort', true, 'CentroidOutputPort', true, ...
           'MinimumBlobArea', 100);
    end

%% Initialize Tracks

    function tracks = initializeTracks()
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
    end    


%% Detect objects

 function [centroids, bboxes, mask, scores] = detectObjects(frame)

        % Detect foreground.
        mask = obj.detector.step(frame);
        
        % Resize the image to increase the resolution of the pedestrian.
        % This helps detect people further away from the camera.
%         resizeRatio = 1.5;
%         frame = imresize(frame, resizeRatio, 'Antialiasing',false);

        % Run ACF people detector within a region of interest to produce
        % detection candidates.
        [bboxes, scores] = detectPeopleACF(frame, option.ROI,...
            'Model','caltech',...
            'NumScaleLevels', 32);
%             'SelectStrongest', false);
        
         % Apply non-maximum suppression to select the strongest bounding boxes.
%         [bboxes, scores] = selectStrongestBbox(bboxes, scores, ...
%                             'RatioType', 'Min', 'OverlapThreshold', 0.6);
                        
        % Compute the centroids
        if isempty(bboxes)
            centroids = [];
        else
            centroids = [(bboxes(:, 1) + bboxes(:, 3) / 2), ...
                (bboxes(:, 2) + bboxes(:, 4) / 2)];
        end

        % Apply morphological operations to remove noise and fill in holes.
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');

        % Perform blob analysis to find connected components.
%         [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
 end

%% Predict New Locations for Existing tracks
% 
% function predictNewLocationsOfTracks()
%         for i = 1:length(tracks)
%             bbox = tracks(i).bbox;
%             
%             % Predict the current location of the track.
%             predictedCentroid = predict(tracks(i).kalmanFilter);
%             
%             % Shift the bounding box so that its center is at the predicted location.
%             tracks(i).predPosition = [predictedCentroid - bbox(3:4)/2, bbox(3:4)];
%        %%%%%%%%%% ^^^ I still don't understand this data structure
%        %%%%%%%%%% manipulation
%         end
%     end

% %% Assign Detections to Tracks
% 
% function [assignments, unassignedTracks, unassignedDetections] = ...
%             detectionToTrackAssignment()
% 
%         % Compute the overlap ratio between the predicted boxes and the
%         % detected boxes, and compute the cost of assigning each detection
%         % to each track. The cost is minimum when the predicted bbox is
%         % perfectly aligned with the detected bbox (overlap ratio is one)
%         predBboxes = reshape([tracks(:).predPosition], 4, [])';
%         cost = 1 - bboxOverlapRatio(predBboxes, bboxes);
% 
%         % Force the optimization step to ignore some matches by
%         % setting the associated cost to be a large number. Note that this
%         % number is different from the 'costOfNonAssignment' below.
%         % This is useful when gating (removing unrealistic matches)
%         % technique is applied.
%         cost(cost > option.gatingThresh) = 1 + option.gatingCost;
% 
%         % Solve the assignment problem.
%         [assignments, unassignedTracks, unassignedDetections] = ...
%             assignDetectionsToTracks(cost, option.costOfNonAssignment);
%     end
% 
% % function [assignments, unassignedTracks, unassignedDetections] = ...
% %         detectionToTrackAssignment()
% %         
% %         nTracks = length(tracks);
% %         nDetections = size(centroids, 1);
% %         
% %         % Compute the cost of assigning each detection to each track.
% %         cost = zeros(nTracks, nDetections);
% %         for i = 1:nTracks
% %             cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
% %         end
% %         
% %         % Solve the assignment problem.
% %         costOfNonAssignment = 20;
% %         [assignments, unassignedTracks, unassignedDetections] = ...
% %             assignDetectionsToTracks(cost, costOfNonAssignment);
% %     end

% %% Update Assigned Tracks
% 
% function updateAssignedTracks()
%         numAssignedTracks = size(assignments, 1);
%         for i = 1:numAssignedTracks
%             trackIdx = assignments(i, 1);
%             detectionIdx = assignments(i, 2);
%             centroid = centroids(detectionIdx, :);
%             bbox = bboxes(detectionIdx, :);
%             
%             % Correct the estimate of the object's location
%             % using the new detection.
%             correct(tracks(trackIdx).kalmanFilter, centroid);
%             
%             % Replace predicted bounding box with detected
%             % bounding box.
%             tracks(trackIdx).bbox = bbox;
%             
%             % Update track's age.
%             tracks(trackIdx).age = tracks(trackIdx).age + 1;
%             
%             % Update track's score history
%             tracks(trackIdx).score = [tracks(trackIdx).score; scores(detectionIdx)];
%             
%             % Update visibility.
%             tracks(trackIdx).totalVisibleCount = ...
%                 tracks(trackIdx).totalVisibleCount + 1;
%             tracks(trackIdx).consecutiveInvisibleCount = 0;
%             
%             % Adjust track confidence score based on the maximum detection
%             % score in the past 'timeWindowSize' frames.
%             T = min(option.timeWindowSize, length(tracks(trackIdx).score));
%             score = tracks(trackIdx).score(end-T+1:end);
%             tracks(trackIdx).confidence = [max(score), mean(score)];
%             
%         end
% end

% %% Update Unassigned Tracks
% 
% function updateUnassignedTracks()
%         for i = 1:length(unassignedTracks)
%             ind = unassignedTracks(i);
%             tracks(ind).age = tracks(ind).age + 1;
%             tracks(ind).bbox = [tracks(ind).bbox; tracks(ind).predPosition];
%             tracks(ind).score = [tracks(ind).score; 0];
%             tracks(ind).consecutiveInvisibleCount = ...
%                 tracks(ind).consecutiveInvisibleCount + 1;
%             
%             % Adjust track confidence score based on the maximum detection
%             % score in the past 'timeWindowSize' frames
%             T = min(option.timeWindowSize, length(tracks(ind).score));
%             score = tracks(ind).score(end-T+1:end);
%             tracks(ind).confidence = [max(score), mean(score)];
%         end
% end

% %% Delete Lost Tracks
% 
% function deleteLostTracks()
%         if isempty(tracks)
%             return;
%         end
%         
%         % Compute the fraction of the track's age for which it was visible.
%         ages = [tracks(:).age];
%         totalVisibleCounts = [tracks(:).totalVisibleCount];
%         visibility = totalVisibleCounts ./ ages;
%         
%         %Check the maximum detection confidence score
%         confidence = reshape([tracks(:).confidence], 2, [])';
%         maxConfidence = confidence(:, 1);
%         
%         % Find the indices of 'lost' tracks.
%         lostInds = (ages < option.ageThresh & visibility < option.visThresh) | ...
%             ([tracks(:).consecutiveInvisibleCount] >= option.invisibTooLong);
% %             (maxConfidence <= option.confidenceThresh);
% %         disp(lostInds);
%         % Delete lost tracks.
%         tracks = tracks(~lostInds);
%     end
% %% Create New Tracks
% 
% function createNewTracks()
%         unassignedCentroids = centroids(unassignedDetections, :);
%         unassignedBboxes = bboxes(unassignedDetections, :);
%         unassignedScores = scores(unassignedDetections);
%         
%         
%         for i = 1:size(unassignedBboxes, 1)
%             
%             centroid = unassignedCentroids(i,:);
%             bbox = unassignedBboxes(i, :);
%             score = unassignedScores(i);
%             
%             % Create a Kalman filter object.
%             kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
%                 centroid, [2, 1], [5, 5], 100);
%             
%             % Create a new track.
%             newTrack = struct(...
%                 'id', nextId, ...
%                 'color', 255*rand(1,3), ...
%                 'bbox', bbox, ...
%                 'score', score, ...
%                 'kalmanFilter', kalmanFilter, ...
%                 'age', 1, ...
%                 'totalVisibleCount', 1, ...
%                 'consecutiveInvisibleCount', 0, ...
%                 'confidence', [score, score], ...
%                 'predPosition', bbox);
%             
%             % Add it to the array of tracks.
%             tracks(end + 1) = newTrack;
%             % Increment the next id.
%             nextId = nextId + 1;
%         end
% end

%% Display Tracking Results

function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        
%         displayRatio = 4/3;
%         frame = imresize(frame, displayRatio);
        
        minVisibleCount = 8;
        if ~isempty(tracks)
              
            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than 
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            
            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);
                
                % Get ids.
                ids = int32([reliableTracks(:).id]);
                
                % Create labels for objects indicating the ones for 
                % which we display the predicted rather than the actual 
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);
                
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                
            end
        end
        
        frame = insertShape(frame, 'Rectangle', option.ROI, ...
            'Color', [255, 0, 0], 'LineWidth', 3);
        
    step (obj.videoPlayer, frame)
    step (obj.maskPlayer, mask)
end

%% release video reader, player
% release(videoPlayer);
% release(videoReader);
% release(fgPlayer);

end
