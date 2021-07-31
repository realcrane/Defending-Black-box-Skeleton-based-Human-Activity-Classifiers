% a matlab code snippet to render the target motions and attacked motions
% in a video

dataSet = 'ntu60';

gpath = sprintf('../../results/%s/STGCN/SMART/', dataSet);
batch = 'batch0_ab_clw_0.60_pl_l2_acc-bone_plw_0.40/';
folder = 'AdExamples_maxFoolRate_batch0_AttackType_ab_clw_0.60_pl_l2_acc-bone_reCon_0.40_fr_96.88/';
dataPath = sprintf('%s%s%s/', gpath, batch,folder);
% dataPath = '../data/';

classFileId = fopen(sprintf('../../data/%s/classes.txt', dataSet));

classes = textscan(classFileId, '%s');

fclose(classFileId);

ori_motions = readNPY(sprintf('%s%s', dataPath, 'ori_motions.npy'));
ad_motions = readNPY(sprintf('%s%s', dataPath, 'ad_motions.npy'));

ori_labels = dlmread(sprintf('%s%s', dataPath, 'tlabels.txt'));
ad_labels = dlmread(sprintf('%s%s', dataPath, 'ad_plabels.txt'));

fooledMotionIndices = [];

for i = 1:size(ori_labels, 1)
    if ori_labels(i) ~= ad_labels(i)
        fooledMotionIndices = [fooledMotionIndices; i];
    end
end



% parentid = [11, 1, 2, 3, 4, ...
%             11, 6, 7, 8, 9, ...
%             0, 11, 12, 13, 14, ... 
%             14, 16, 17, 18, 19, ...
%             14, 21, 22, 23, 24];

%%ntu60
% parentid = [1, 1, 21, 3, 21,...
%             5, 6, 7, 21, 9, ...
%             10, 11, 1, 13, 14, ...
%             15, 1, 17, 18, 19, ...
%             2, 8, 8, 12, 12];
        
parentid = [2 1 21 3 21 5 6 7 21 9 10 11 1 13 14 15 1 17 18 19 2 8 8 12 12];

% motionIndices = [8];
motionIndices = 1:size(fooledMotionIndices, 1);

oriClipLength = 300;

scalar = 20; %20 , only needed for nturgdb datasets

margin = 10;

ori_motions = permute(ori_motions, [1, 3, 2, 4, 5]);
ad_motions = permute(ad_motions, [1, 3, 2, 4, 5]);

for i = 1:size(motionIndices, 2)

    motionIndex = fooledMotionIndices(motionIndices(i));
%     motionIndex = motionIndices(i);
    %temp = reshape(ori_motions(motionIndex, :, :), [oriClipLength, 75]);
    temp = reshape(ori_motions(motionIndex, :, :, :, 1), [oriClipLength, 75]);
    temp = temp * scalar;
    motion = reshape(temp', [], 25, oriClipLength);
    
    clipLength = 0;
    
    %only for nturgbd60
    for k = size(motion, 3):-1:1
        ind = find(motion(:, :, k));
        if size(ind, 1) > 60
            clipLength = k;
            break;
        end
    end
    motion = motion(:, :, 1:clipLength);
    
    acc = (motion(:, :, 3:clipLength) - 2 * motion(:, :, 2:(clipLength-1)) + motion(:, :, 1:(clipLength-2)));

    accMag = acc .* acc;

    accMag = sqrt(sum(accMag));

    figure('visible', 'off');
    for i = 1:5
        for j = 1:5
            subplot(5, 5, (i-1)*5+j);
            plot(1:(clipLength-2), reshape(accMag(1, (i-1)*5+j, :), clipLength-2, 1), 'LineWidth', 2);
        end
    end

    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 6.4 4.8]);

    print('-djpeg', sprintf('%s%d%s', dataPath, motionIndex, '_ori.jpg'), '-r100');


    temp = reshape(ad_motions(motionIndex, :, :, :, 1), [oriClipLength, 75]);
    temp = temp * scalar;
    ad_motion = reshape(temp', [], 25, oriClipLength);

    ad_motion = ad_motion(:, :, 1:clipLength);
    
    acc = (ad_motion(:, :, 3:clipLength) - 2 * ad_motion(:, :, 2:(clipLength-1)) + ad_motion(:, :, 1:(clipLength-2)));

    accMag = acc .* acc;

    accMag = sqrt(sum(accMag));

    figure('visible', 'off');

    for i = 1:5
        for j = 1:5
            subplot(5, 5, (i-1)*5+j);
            plot(1:(clipLength-2), reshape(accMag(1, (i-1)*5+j, :), (clipLength-2), 1), 'LineWidth', 2);
        end
    end

    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 6.4 4.8]);

    print('-djpeg', sprintf('%s%d%s', dataPath, motionIndex, '_ad.jpg'), '-r100');

    figure('Renderer', 'painters', 'Position', [10 10 1024 768]);

    myVideo = VideoWriter(sprintf('%s%d%s', dataPath, motionIndex, '_motion'), 'MPEG-4'); %open video file
    myVideo.Quality = 100;
    myVideo.FrameRate = 20;  %can adjust this, 5 - 10 works well for me
    open(myVideo);

    s = 1;

    motion(1, :, s) = motion(1, :, s) - motion(1, 11, s);
    motion(2, :, s) = motion(2, :, s) - motion(2, 11, s);
    motion(3, :, s) = motion(3, :, s) - motion(3, 11, s);
    
    ad_motion(1, :, s) = ad_motion(1, :, s) - ad_motion(1, 11, s);
    ad_motion(2, :, s) = ad_motion(2, :, s) - ad_motion(2, 11, s);
    ad_motion(3, :, s) = ad_motion(3, :, s) - ad_motion(3, 11, s);
    
%     motion(1, :, s) = -motion(1, :, s);
%     motion(3, :, s) = -motion(3, :, s);
% 
%     ad_motion(1, :, s) = -ad_motion(1, :, s);
%     ad_motion(3, :, s) = -ad_motion(3, :, s);
    
    
    for s=1:size(motion, 3)
        
        
        subplot(3,4,[1, 2, 5, 6]);
        
        plot3(motion(1, :, s), motion(3,:,s) ,motion(2,:,s),'*');
%         plot3(motion(1, :, s), motion(2,:,s) ,motion(3,:,s),'*');
        minx = min(min(motion(1, :, 2:end)));
        maxx = max(max(motion(1, :, 2:end)));
        miny = min(min(motion(3, :, 2:end)));
        maxy = max(max(motion(3, :, 2:end)));
        minz = min(min(motion(2, :, 2:end)));
        maxz = max(max(motion(2, :, 2:end)));
    
        set(gca,'DataAspectRatio',[1 1 1])
        %axis([0 400 0 400 0 400])
        for j=1:25
            if j == 11
                continue;
            end
            c1=parentid(j);
            c2=j;
            l = line([motion(1, c1,s) motion(1, c2,s)], [motion(3, c1,s) motion(3, c2,s)], [motion(2, c1,s) motion(2, c2,s)], 'LineWidth', 3);
%             l = line([motion(1, c1,s) motion(1, c2,s)], [motion(2, c1,s) motion(2, c2,s)], [motion(3, c1,s) motion(3, c2,s)], 'LineWidth', 2);

        end
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
%         axis([-15 15 -10 20 -20 15]);
        axis([minx-margin maxx+margin miny-margin maxy+margin minz-margin maxz+margin]); %only for nturgbd60
        title(sprintf('Ground Truth: %s', classes{1}{ori_labels(motionIndex) + 1}), 'Units', 'normalized', 'Position', [0.5, -0.1, 0]);
        
        set( gca, 'xdir', 'reverse' );
        
        subplot(3,4,[3, 4, 7, 8]);

        plot3(ad_motion(1, :, s), ad_motion(3,:,s) ,ad_motion(2,:,s),'*');
%         plot3(ad_motion(1, :, s), ad_motion(2,:,s) ,ad_motion(3,:,s),'*');
        set(gca,'DataAspectRatio',[1 1 1])
        %axis([0 400 0 400 0 400])
        for j=1:25
            if j == 11
                continue;
            end
            c1=parentid(j);
            c2=j;
            l = line([ad_motion(1, c1,s) ad_motion(1, c2,s)], [ad_motion(3, c1,s) ad_motion(3, c2,s)], ... 
            [ad_motion(2, c1,s) ad_motion(2, c2,s)], 'LineWidth', 3);
%             l = line([ad_motion(1, c1,s) ad_motion(1, c2,s)], [ad_motion(2, c1,s) ad_motion(2, c2,s)], ... 
%             [ad_motion(3, c1,s) ad_motion(3, c2,s)], 'LineWidth', 2);
        end
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
%         axis([-15 15 -10 20 -20 15]);
        axis([minx-margin maxx+margin miny-margin maxy+margin minz-margin maxz+margin]); %only for nturgbd60
        title(sprintf('After Attack: %s', classes{1}{ad_labels(motionIndex) + 1}), 'Units', 'normalized', 'Position', [0.5, -0.1, 0]);
        set( gca, 'xdir', 'reverse' );
        
        
        subplot(3,4, [9, 10, 11, 12]);
        

        bar(sqrt((motion(1, :, s) - ad_motion(1, :, s)).^2 + (motion(2, :, s) - ad_motion(2,:,s)).^2 + (motion(3, :, s) - ad_motion(3,:,s)).^2));
        title('Perturbations on Joints (Joint 1-25) as Euclidean distances');

        
        frame = getframe(gcf);
        writeVideo(myVideo, frame);

    %     pause(1/60)
%         k = waitforbuttonpress;
    end

    close(myVideo);
    
end