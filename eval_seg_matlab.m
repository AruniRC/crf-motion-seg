%% Evaluate CRF segmentations from grid search
%   A grid-search over CRF settings is pre-computed by running Python
%   script `cross_validate_crf.py`. 
%   Now, this MATLAB script iterates over the saved results (segmentations)
%   and ground-truth and computes the multiple-instance segmentation 
%   F-score for each setting.

% # Run 1
% # bilateral (colorspace)
% RUN_NUM = 1;
% range_W = [3, 5, 10];
% range_XY_STD = [40, 50, 60, 70, 80, 90, 100];
% range_RGB_STD = [3, 5, 7, 9, 10];

RUN_NUM = 2
range_W = [10, 15, 20]
range_XY_STD = [10, 20, 30, 40]
range_RGB_STD = [1, 2, 3, 4, 5, 6]

IMAGE_DATA = '/data/arunirc/Research/dense-crf-data/training_subset/';
OUT_DIR = '/data/arunirc/Research/dense-crf-data/cross-val-crf-modifiedObjPrior/';
GT_DIR = '/data2/arunirc/Research/dense-crf/data/ground-truth/FBMS/Trainingset';

% best_f = 0;
% best_w = 0;
% best_x = 0;
% best_r = 0;


for i = 1:numel(range_W)
    w = range_W(i);
    
    for j = 1:numel(range_XY_STD)
        x = range_XY_STD(j);
        
        for k = 1:numel(range_RGB_STD)
            r = range_RGB_STD(i);
            
            outDirName = fullfile(OUT_DIR, ['w-' num2str(w) '_x-' num2str(x) '_r-' num2str(r)] );
            disp(outDirName);
            assert(exist(outDirName, 'dir')==7, 'folder does not exist');
            
            listing = dir(outDirName);
            dirList = {listing(:).name};
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            precision = [];
            recall = [];
            f_score = [];
            
            % over each folder (i.e. video)
            for d = 3:numel(dirList)
                vidName = dirList{d};
                vid_dir = fullfile(outDirName, vidName);
                listing = dir([vid_dir '/*.mat']);
                matListing = {listing(:).name};
                
                vidSegs = cell(1, numel(matListing));
                vidGT = cell(1, numel(matListing));
                
                for m = 1:numel(matListing)
                    [~, fNum,~] = fileparts(matListing{m});
                    fNum = str2num(fNum);
                    
                    gtFile = fullfile(GT_DIR, vidName, 'GroundTruth', ...
                                        [sprintf('%03d',fNum) '_gt.png']);
                    segFile = fullfile(outDirName, vidName, matListing{m});
                    
                    % load ground truth (png)
                    gt = imread(gtFile);
                    [~,~,Y] = unique(gt); % image --> unique integer labels
                    Y = reshape(Y, size(gt));
                    
                    % load segmentation, transform into unique integer
                    % labels
                    dat = load(segFile);
                    [~, seg] = max(dat.objectProb, [], 3); % argmax of 3-D predictions

                    vidSegs{m} = seg;
                    vidGT{m} = Y;
                end
                
                % eval at video level
                vidGT = cat(3, vidGT{:});
                vidSegs = cat(3, vidSegs{:});
                
                [ F_final, P_final, R_final, obj_detected, obj_detected_075, obj_gt ] = ...
                        accuracy( vidGT, vidSegs );
                    
%                 % save that videos result 
%                 disp(vidName);
%                 disp(F_final);
%                 
                precision(end+1) = P_final;
                recall(end+1) = R_final;
                f_score(end+1) = F_final;
                
            end
            
            mean_f = mean(f_score);
            
            fprintf('mean F-score: %f\n', mean_f);
            
            if mean_f > best_f
                best_f = mean_f;
                best_w = w;
                best_x = x;
                best_r = r;
            end
           
            temp = sprintf('%f %d %d %d', best_f, best_w, best_x, best_r);
            fid = fopen(fullfile(OUT_DIR, ['cv-result-run-' num2str(RUN_NUM) '.txt']), 'w');
            fprintf(fid, temp);
            
        end
    end
end
        
        
        