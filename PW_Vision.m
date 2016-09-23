
% PW Computer Vision 
% 
% 
% Basic Overview
% 
% The process for detecting and counting people in this program is as
% follows:
% 
% First, using the input video, a foreground detector is setup. This system
% object detects and subtracts the background from the video per frame,
% resulting in a white on black mask. The mask is then cleaned through
% morphological operations. The mask passes through a blob analysis object
% outputting bounding boxes and centroids per detection. These are then
% filtered according to area, location, and aspect ratio. After filtering
% detections, tracking is implemented. The tracks are filtered, and the
% remaining tacks are displayed and counted.
% 
% 
% Known Issues
% 
% Video input is a timelapse with one frame taken every three seconds. As
% such, a number of issues arise:
%
% 1. People/Cars/Bikes/Dogs/Objects move extremely quickly, jumping across
% the video. Computer vision (CV) motion tracking works best with higher
% frame rates as blobs 'appear' to move smoothly. As a result, tracks in
% the timelapse struggle to maintain correct assignments with detections.
% 
% 2. As the camera is on auto-mode, the images can greatly vary between
% frames in terms of color balance and lighting (exposure, aperture, ISO,
% etc). Coupled with the low frame rate, this forces erros in the
% foreground detector which in turn translates to multiple (upwards of 50+!)
% false detections in a single frame.
% 
% 3. Dogs and bikes cannot be properly detected and categorized. For bikes,
% there are too few frames in which they are detected and shown resulting
% in failure to assign tracks. For dogs, detection works; however, as
% tracking is based off a linear model, the tracks cannot accurately
% predict and assign detections since the dogs move erratically with
% the low frame rate.
% 
% 4. Occaisionally, objects that become occluded are double or triple
% counted. This could be a result of insufficient frames to accurately
% predict an established track.



function PW_Vision()

%% Main Program
videoFile = 'C:\Users\szilasia\Desktop\Test_1.avi';
videoFWriter = vision.VideoFileWriter('Sample.avi');

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
option.peopleRatio          = 2.5;					 % Aspect ratio to filter out blobs for example if the blob is too wide or too long its not a real thing
option.peopleArea           = 55000;                 % A threshold to control the area for detecting pedestrians. 
option.costOfNonAssignment  = 12;                    % A tuning parameter to control the likelihood of creation of a new track - Robbey didnt change
option.ageThresh            = 5;                     % A threshold to determine the minimum length required for a track being true positive. A ratio of the total number of visible frames over the total age of the track
option.visThresh            = 0.5;                   % A threshold to determine the minimum visibility value for a track being true positive.
option.invisibTooLong       = 5;                     % A threshold to determine if a track has gone off frame.
option.ROI					= roipoly(frame);		 % User-defined Region of Interest (ROI) to help filter detections.

close all;

while ~isDone(obj.reader);
    
    % code is sequential here one loop of this code is one frame of the
    % video
    
    % Simply moves video player forward
    frame = step(obj.reader);
    
    % Section A: No tracking this code will simply create bounding boxes
    % around people for each frame
    
    % This detects objects that is any blob
    [centroids, bboxes, mask] = detectObjects(frame);
    
    % This filters the objects detected in the equation above to count
    % people, bboxPeople = boundingBox people, centPeople = centroid people
    [bboxPeople, centPeople] = detectionFilter (option.ROI, option.peopleRatio, option.peopleArea, bboxes, centroids);
    
    % Section B: This section attempts to track these bounding boxes passed through the detectionFilter
    % through time to produce a count of people coming and going
    
    % Runs through tracks from previous frames and attempts to predict them in the current frame
    % using the Hungarian algorithm which uses the distance between the predictation and the detection
    predictNewLocationsOfTracks();    
    
    % Attempts to assign detections to the tracks
    % assignments = A track that has a detection in this frame
    % unassignedTrack = A track with no detection in this frame
    % unassignedDections = Detections with no track in this frame
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    % Update counts like age and visibility
    updateAssignedTracks();    
    updateUnassignedTracks();  
    deleteLostTracks(); 
    
    % Creates new tracks takes unassigned detections and creates new tracks
    % only for unassigned detections
     
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
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 800, 500]);
        obj.maskPlayer = vision.VideoPlayer('Position', [840, 400, 800, 500]);
        
        % Create System objects for foreground detection and blob analysis
        
        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background. 
        
        % So white areas in the foreground mask correspond to differences
        % in the image in time therefore the trick is to get the settings
        % as such that the white appears when we want it to e.g people,
        % dogs
        
        obj.detector = vision.ForegroundDetector('NumGaussians', 5, ...
            'NumTrainingFrames', 100, 'LearningRate', 0.005, ...
            'InitialVariance', 900);
        
        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.
        
        obj.blobs = vision.BlobAnalysis('CentroidOutputPort', true, ...
			'AreaOutputPort', false, ...
			'BoundingBoxOutputPort', true, ...
			'MinimumBlobArea', 200);
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
            'displayed', {},...
            'centroidLogx',[],...
            'centroidLogy',[]); 
    end    

%% Detect objects

	 function [centroids, bboxes, mask] = detectObjects(frame)
		
		%Create foreground mask
		mask =  obj.detector.step(frame);
		
		%Morphological Image cleaning - the idea being to clean out and get
		%rid of noise
        % cleaner one
		cleanMask = imopen(mask, strel('square', 2));
        % cleaner two
        cleanMask = imerode(mask, strel('square', 4));
        % cleaner three
 		cleanMask = imdilate(cleanMask, strel('square', 3));
%         cleanMask = imerode(cleanMask, strel('disk', 2));
        % cleaner four - detects holes
		cleanMask = imfill(cleanMask, 'holes');
		mask = cleanMask;
		
		%Blob detection - blob analysis
		[centroids, bboxes] = obj.blobs.step(mask);
	 end

%% Filter detections

	function [filteredIdx, filteredCentroids] = detectionFilter (ROI, maxRatio, maxArea, bbox, centroids)
		
        if isempty(bbox)
            filteredIdx = bbox;
            filteredCentroids = centroids;
            return
        end
        
		%Calculate Ratio, Area, Centroid Coordinates of the bounding boxes
		%of the blobs which are represented as arrays BBox: [x,y,w,h]
		w = bbox(:,3);
		h = bbox(:,4);
		if ~isempty(ROI)
            x = uint16(centroids(:,1));
            y = uint16(centroids(:,2));
		end

		ratio = double(w) ./ double(h);
		area = h .* w;

		%Filter out the boxes that don't meet the requirements (too large/
		%centroid not in the right area 

		badBbox = ratio > maxRatio;
		badBbox = badBbox | area > maxArea;

		%Filter out centroids if not in region of interest (ROI)
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
            
            % Predict the current location of the track the kalman filter basically
            % makes predictions based off velocity and other data in previous detections
            % settings are configured in configureKalmanFilter
            
            predictedCentroid = predict(tracks(i).kalmanFilter);
            
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
        
        % centPeople are the centroids that have been filtered through in
        % this particular frame
        
        nDetections = size(centPeople, 1);
        
        % Compute the cost of assigning each detection to each track. The
        % cost is looking at the difference between the centroid of the
        % predictation and the centroid of the detection e.g the distance
        % between the two and the best match is the shortest distance
        % between the two or the lowest cost
        
        % Find the cost for every assignment, an assignment is a assigned
        % track in the previous frame  
        % A unassigned track is a track that had no detection in the
        % previous frame but has not yet been deleted it is within the time
        % limit
        
        % Now go through and calculate the cost for each track in this
        % frame (both assigned and unassigned)
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centPeople);
        end
        
        % Here is the hungarian algrorithm
        % The hungarian algrorithm for this frame will determine whether a
        % detection and track are the same object, 
        % thats when the kalman Filter is correct - see the first graph here http://www.mathworks.com/help/vision/ref/assigndetectionstotracks.html
        
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, option.costOfNonAssignment);
    end
	
%% Update Assigned Tracks
    
% Update the age of the assigned tracks and the visible count
% age is the number of frames that a track has existed in the video 
% the visibility count is the name 

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
            
            % Add the centroid of the track to the track's centroid log
            
            %%tracks(trackIdx).centroidLog = [tracks(trackIdx).centroidLog+centroid]
            
            i = length(tracks(trackIdx).centroidLogx)
            
            tracks(trackIdx).centroidLogx(i+1) = centroid(1);
            
            tracks(trackIdx).centroidLogy(i+1) = centroid(2);
            
            clx = tracks(trackIdx).centroidLogx
            
            cly = tracks(trackIdx).centroidLogy
            
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
% This function determines when a track has been lost - which usually means that the blob has gone off the screen 

	function deleteLostTracks()
        if isempty(tracks)
            return;
        end   
        
        % Compute the fraction of the track's age for which it was visible.
        % ages is a list of the all the ages of the tracks
        ages = [tracks(:).age];
        % Number of frames that a track was assigned a detection
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        % visibility is a ratio
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks % 
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
            
            % Create a Kalman filter object, every track has a Kalman
            % filter - the purpose of the Kalman filter is to 
            % make predictions on where the object will be
            
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 500], [500, 350], 50);
            
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0, ...
				'predPosition', bbox, ...
                'displayed', false,...
                'centroidLogx',[],...
                'centroidLogy',[]); 
            
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
               
                
                % Count the displayed tracks and update track display
                % option.
                for i = 1:length(reliableTracks)
                
                    if reliableTracks(i).displayed == false
                        count = count + 1;
                        reliableTracks(i).displayed = true;
                    end
                end
                countid = [reliableTracks(:).id];
                
                for i = 1:length(reliableTracks);
                    if countid(i) == tracks(i).id
                        tracks(i).displayed = true;
                    end
                end
                
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);
                
                % Get ids.

%                 ids = int32([reliableTracks(:).id]);
                
                % Create labels for objects indicating the ones for 
                % which we display the predicted rather than the actual 
                % location.
%                 labels = cellstr(int2str(ids'));
%                 predictedTrackInds = ...
%                     [reliableTracks(:).consecutiveInvisibleCount] > 0;
%                 isPredicted = cell(size(labels));
%                 isPredicted(predictedTrackInds) = {'predicted'};
%                 labels = strcat(labels, isPredicted);
                labels = 'tracked person';
                
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
%                 frame = insertMarker(frame, centroids, '+', 'Color', 'green');
            end
        end
        
        countLabel = strcat('count: ', num2str(count));
        frame  = insertText(frame, [10 10], countLabel, 'BoxOpacity', 1, ...
            'FontSize', 14);
        
        step (videoFWriter, frame)

        step (obj.videoPlayer, frame)
        step (obj.maskPlayer, mask)
    end


%% release video reader, player
release(obj.videoPlayer);
release(obj.reader);
release(obj.maskPlayer);

end
