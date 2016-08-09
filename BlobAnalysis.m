%% Foreground Testing and BlobAnalysis

video = vision.VideoFileReader('Test_1.avi');

foregroundDetector = vision.ForegroundDetector('NumTrainingFrames', 75, ...
        'LearningRate', .01, ...
        'NumGaussians', 3);

blobs = vision.BlobAnalysis('CentroidOutputPort', false, ...
       'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobArea', 100);
   
dogRatio = .8;

peopleRatio = .6;

vPlayer = vision.VideoPlayer();
fPlayer = vision.VideoPlayer();
   
% opticFlow = opticalFlowLK();

while ~isDone(video)
    
   frame = step(video);
   mask = foregroundDetector.step(frame);
   
%    cleanMask = imopen(mask, strel('Disk',1));
   cleanMask = imopen(cleanMask, strel('rectangle', [3,3]));
   cleanMask = imclose(cleanMask, strel('rectangle', [15, 15])); 
   cleanMask = imfill(cleanMask, 'holes');
   
%    flow = estimateFlow(opticFlow, cleanMask);
   
   [~, centroids, bboxes] = blobs.step(cleanMask);
   
%    [peoples, scores] = detectPeopleACF(frame,...
%             'Model','caltech',...
%             'NumScaleLevels', 32);
   
   if ~isempty(bboxes)

        [bboxDogs, centDogs] = detectionFilter(dogRatio, 400, bboxes, centroids);
        [bboxPeople, centPeople] = detectionFilter(peopleRatio, 1500, bboxes, centroids);
   frame = insertShape(frame, 'rectangle', bboxDogs, 'Color', 'green');
   frame = insertShape(frame, 'rectangle', bboxPeople, 'Color', 'red');
%    cleanMask = insertShape(cleanMask, 'rectangle', bboxPeople, 'Color', 'green');
   
%    imshow(frame);
%    hold on;
%    plot(flow,'DecimationFactor', [5,5], 'ScaleFactor', 25);
%    hold off;
   end
   
   step(vPlayer, frame);
   step(fPlayer, cleanMask);
end

release(vPlayer);
release(fPlayer);