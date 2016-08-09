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
   
% Morphological Image Cleaning
   
%    cleanMask = imopen(mask, strel('Disk',1));
   cleanMask = imopen(mask, strel('rectangle', [3,3]));
   cleanMask = imclose(cleanMask, strel('rectangle', [15, 15])); 
   cleanMask = imfill(cleanMask, 'holes');
   
%    flow = estimateFlow(opticFlow, cleanMask);

% Blob detection
   [centroids, bboxes] = blobs.step(cleanMask);
   
%    [peoples, scores] = detectPeopleACF(frame,...
%             'Model','caltech',...
%             'NumScaleLevels', 32);
   
   if ~isempty(bboxes)

%Filter detections based off of size, aspect ratio, location
%         [bboxDogs, centDogs] = detectionFilter(dogRatio, 400, bboxes, centroids);
        [bboxPeople, centPeople] = detectionFilter(ROI, peopleRatio, 1500, bboxes, centroids);
%    frame = insertShape(frame, 'rectangle', bboxDogs, 'Color', 'green');
        frame = insertShape(frame, 'rectangle', bboxPeople, 'Color', 'red');
%    cleanMask = insertShape(cleanMask, 'rectangle', bboxPeople, 'Color', 'green');
   


% Predict track Locations
        trackPredictions(tracks);




%    imshow(frame);
%    hold on;
%    plot(flow,'DecimationFactor', [5,5], 'ScaleFactor', 25);
%    hold off;
   end
   
   
   
   
%Continue to next frame
   step(vPlayer, frame);
   step(fPlayer, cleanMask);
end

%% Release the videoplayers

release(vPlayer);
release(fPlayer);