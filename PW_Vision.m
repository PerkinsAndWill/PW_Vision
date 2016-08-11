
% PW Computer Vision 

function PW_Vision()

%% Main Program
videoFile = 'Test_1.avi';

obj = setupSystemObjects(videoFile);

% Create an empty array of tracks.
tracks = initializeTracks(); 

% ID of the next track.
nextId = 1; 

% Object Count
count = 0;

%Get a still of the first frame
frame = step(obj.reader);

% Set the global parameters
option.peopleRatio          = 3;					 % Aspect ratio to filter out blobs.
option.peopleArea           = 5000;                  % A threshold to control the area for detecting pedestrians. 
option.costOfNonAssignment  = 10;                    % A tuning parameter to control the likelihood of creation of a new track.
option.confidenceThresh     = 2;                     % A threshold to determine if a track is true positive or false alarm.
option.ageThresh            = 4;                     % A threshold to determine the minimum length required for a track being true positive.
option.visThresh            = 0.6;                   % A threshold to determine the minimum visibility value for a track being true positive.
option.invisibTooLong       = 8;                     % A threshold to determine if a track has gone off frame.
option.ROI					= roipoly(frame);		 % User-defined Region of Interest (ROI) to help filter detections.

close all;


while ~isDone(obj.reader);
    frame = step(obj.reader);
    
    [centroids, bboxes, mask] = detectObjects(frame);
    
    [bboxPeople, centPeople] = detectionFilter (option.ROI, option.peopleRatio, option.peopleArea, bboxes, centroids);
    
    predictNewLocationsOfTracks();    
    
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();    
    updateUnassignedTracks();    
    deleteLostTracks();    
    createNewTracks();
    
    displayTrackingResults();
    
%     countTrackedObjects(tracked);

    % Exit the loop if the video player figure is closed by user.     
    if ~isDone(obj.reader) && (~isOpen(obj.videoPlayer) || ~isOpen(obj.maskPlayer))
        break
    end
    
end

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
            'NumTrainingFrames', 100, 'LearningRate', 0.01);
        
        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.
        
        obj.blobs = vision.BlobAnalysis('CentroidOutputPort', true, ...
			'AreaOutputPort', false, ...
			'BoundingBoxOutputPort', true, ...
			'MinimumBlobArea', 100);
    end

%% Initialize Tracks

    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {}, ...
            'predPosition', {},...
            'number', {},...
            'displayed', {});
    end    

%% Detect objects

	 function [centroids, bboxes, mask] = detectObjects(frame)
		
		%Create foreground mask
		mask =  obj.detector.step(frame);
		
		%Morphological Image cleaning
		cleanMask = imopen(mask, strel('square', 3));
% 		cleanMask = imclose(cleanMask, strel('disk', 5));
%         cleanMask = imerode(cleanMask, strel('disk', 2));
		cleanMask = imfill(cleanMask, 'holes');
		mask = cleanMask;
		
		%Blob detection
		[centroids, bboxes] = obj.blobs.step(mask);
	 end

%% Filter detections

	function [filteredIdx, filteredCentroids] = detectionFilter (ROI, maxRatio, maxArea, bbox, centroids)
		
        if isempty(bbox)
            filteredIdx = bbox;
            filteredCentroids = centroids;
            return
        end
        
		%Calculate Ratio, Area, Centroid Coordinates
		w = bbox(:,3);
		h = bbox(:,4);
		if ~isempty(ROI)
            x = uint16(centroids(:,1));
            y = uint16(centroids(:,2));
		end

		ratio = double(w) ./ double(h);
		area = h .* w;

		%Filter out the bboxes

		badBbox = ratio > maxRatio;
		badBbox = badBbox | area > maxArea;

		%Filter out centroids if not in ROI
		badCentroid = int8.empty();
%         disp (centroids);
		if ~isempty(ROI)
			for i = 1:size(centroids,1)
				if ROI(y(i),x(i))  == 1
					badCentroid = [badCentroid 0];
				else
					badCentroid = [badCentroid 1];
				end
			end

			%Combine filters
			filter = badBbox | badCentroid';

		else
			filter = badBbox;
		end

		%Apply the Filter
		filteredIdx = bbox(logical(~filter), :);
		filteredCentroids = centroids(logical(~filter), :);
	end
	
%% Predict New Locations of exisiting tracks

	function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            
            % Predict the current location of the track.
            predictedCentroid = vision.KalmanFilter.predict(tracks(i).kalmanFilter);
            
            % Shift the bounding box so that its center is at 
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
	end 
	
%% Assign Detections to Tracks

	function [assignments, unassignedTracks, unassignedDetections] = ...
			detectionToTrackAssignment()
   
        nTracks = length(tracks);
        nDetections = size(centPeople, 1);
        
        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centPeople);
        end
        
        % Solve the assignment problem.
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, option.costOfNonAssignment);
    end
	
%% Update Assigned Tracks

	function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centPeople(detectionIdx, :);
            bbox = bboxPeople(detectionIdx, :);
            
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
    end
	
%% Update unassigned Tracks

	function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

%% Delete Lost Tracks

	function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages < option.ageThresh & visibility < option.visThresh) | ...
            [tracks(:).consecutiveInvisibleCount] >= option.invisibTooLong;
        
        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end

%% Create New Tracks

	function createNewTracks()
        centPeople = centPeople(unassignedDetections, :);
        bboxPeople = bboxPeople(unassignedDetections, :);
        
        for i = 1:size(centPeople, 1)
            
            centroid = centPeople(i,:);
            bbox = bboxPeople(i, :);
            
            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [50, 200], [25, 300], 250);
            
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0, ...
				'predPosition', bbox, ...
                'number', 0, ...
                'displayed', false);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;
            
            % Increment the next id.
            nextId = nextId + 1;
        end
    end
	
%% Display Results

	function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        
        if ~isempty(tracks)
              
            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than 
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > option.ageThresh;
            reliableTracks = tracks(reliableTrackInds);
            
            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                
                for i = 1:length(reliableTracks)
                
                    if reliableTracks(i).displayed == false
                        count = count + 1;
                        reliableTracks(i).displayed = true;
                    end
                end
                countid = reliableTracks(:).id;
                
                for i = length(reliableTracks);
                    tracks(countid(i)).displayed = true;
                end
                
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
                isPredicted(predictedTrackInds) = {'predicted'};
                labels = strcat(labels, isPredicted);
                
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                frame = insertMarker(frame, centroids, '+', 'Color', 'green');
            end
        end
        
        frame  = insertText(frame, [10 10], count, 'BoxOpacity', 1, ...
            'FontSize', 14);

        step (obj.videoPlayer, frame)
        step (obj.maskPlayer, mask)
    end

%% Count Tracked Objects

%     function countTrackedObjects(tracked)
%% release video reader, player
release(obj.videoPlayer);
release(obj.reader);
release(obj.maskPlayer);

end
