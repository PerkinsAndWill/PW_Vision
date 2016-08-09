%% PW Computer Vision 

clear all
close all
set(0, 'DefaultFigureWindowStyle','docked')


%% Main Program
videoFile = 'Test_1.avi';

obj = setupSystemObjects(videoFile);

% Create an empty array of tracks.
tracks = initializeTracks(); 

% ID of the next track.
nextId = 1; 


% Set the global parameters.
option.scThresh             = 0.3;              % A threshold to control the tolerance of error in estimating the scale of a detected pedestrian. 
option.gatingThresh         = 0.9;              % A threshold to reject a candidate match between a detection and a track.
option.gatingCost           = 100;              % A large value for the assignment cost matrix that enforces the rejection of a candidate match.
option.costOfNonAssignment  = 10;               % A tuning parameter to control the likelihood of creation of a new track.
option.timeWindowSize       = 16;               % A tuning parameter to specify the number of frames required to stabilize the confidence score of a track.
option.confidenceThresh     = 2;                % A threshold to determine if a track is true positive or false alarm.
option.ageThresh            = 8;                % A threshold to determine the minimum length required for a track being true positive.
option.visThresh            = 0.6;              % A threshold to determine the minimum visibility value for a track being true positive.

while ~isDone(obj.reader);
    frame = readFrame();
    
    [centroids, bboxes, scores] = detectPeople();
    
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

hsizeh = 30
sigmah = 6
h = fspecial('log', hsizeh, sigmah)
subplot(121); imagesc(h)
subplot(122); mesh(h)
colormap(jet)
%% Create System Objects
% Create System objects used for reading the video frames, detecting
% foreground objects, and displaying results.

    function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.
        
        % Create a video file reader.
        obj.reader = vision.VideoFileReader('atrium.mp4');
        
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
            'NumTrainingFrames', 15, 'MinimumBackgroundRatio', 0.7);
        
        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.
        
        %obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
         %   'AreaOutputPort', true, 'CentroidOutputPort', true, ...
          %  'MinimumBlobArea', 400);
    end

