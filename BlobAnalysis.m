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
   
while ~isDone(video)
    
   frame = step(video);
   mask = step(foregroundDetector, frame);
   
   cleanMask = imopen(mask, strel('Disk',2));
   
   
   bboxes = step(blobs, cleanMask);
   
   if ~isempty(bboxes)

        bboxDogs = detectionFilter(dogRatio, 400, bboxes);
        bboxPeople = detectionFilter(peopleRatio, 1500, bboxes);
   frame = insertShape(frame, 'rectangle', bboxDogs, 'Color', 'green');
   frame = insertShape(frame, 'rectangle', bboxPeople, 'Color', 'red');
%    cleanMask = insertShape(cleanMask, 'rectangle', bboxPeople, 'Color', 'green');
   end
   
   step(vPlayer, frame);
   step(fPlayer, cleanMask);
end

release(vPlayer);
release(fPlayer);