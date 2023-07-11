%**************************************************************************
%   利用对原图像和输出图像进行加权处理，
%   产生加权后的图像
%**************************************************************************

clear
close all;
clc
path = 'D:\Infrared_Target_Detection\Infrared target Detection\Contrast_Experiment\IRSTd\';
% path = 'E:\Project\Infrared target Detection\ExperimentResults\ExperimentResults(TopHat)\';
% path = 'E:\Project\infrared_data\target_detect_tophat_sample\';

image_path = [path 'images\'];
result_path = [path 'noThSe\inpainted\'];
mask_path = [path 'noThSe\mask\'];
target_result_path = [path 'noThSe\results\'];
target_mesh_path = [path 'noThSe\mesh\'];
target_seg_path = [path 'noThSe\target_segment\'];
mkdir(target_result_path)
mkdir(target_mesh_path)
mkdir(target_seg_path)
image_Files = dir(strcat(image_path, '*.png'));
result_Files = dir(strcat(result_path, '*.png'));
mask_Files = dir(strcat(mask_path, '*.png'));
len = length(image_Files);

for i = 1:len
    image = imread([image_path image_Files(i).name]);
    if size(image, 3) > 1
        image = rgb2gray(image);
    end
    result = rgb2gray(imread([result_path result_Files(i).name]));
    mask = imread([mask_path mask_Files(i).name]);
%     v = max(max(image));
%     [x, y] = find(v == image);
%     imwrite(uint8(target_results), [target_result_path, image_Files(i).name])

%     temp = image(133:143, 78:88);
%     imwrite(uint8(temp), ['C:\Users\Health\Desktop\', image_Files(i).name])


    image_splice = image;
    [m, n] = size(image); 
    
    image_splice = double(image_splice);
    image = double(image);
    for k = 1:m
        for j = 1:n
            if mask(k,j) == 0
                image_splice(k,j) = result(k,j);
            end
        end
    end
    
    target_results = image - image_splice;
%     temp = target_results(121:131, 88:98);
    imwrite(uint8(target_results), [target_result_path, image_Files(i).name])

    out = target_results;
    out(out<0)=0;
%     break
%     MAX = max(max(out));
%     MIN = min(min(out));
% %     %nomalize
%     for k = 1:m
%         for j = 1:n
%             out(k,j) = (out(k,j)-MIN)/(MAX-MIN);
%         end
%     end
%     
%     target_results = uint8(target_results);
%     imwrite(uint8(out*255), [target_result_path, image_Files(i).name])
%     temp = out(121:131, 88:98);
%     imwrite(uint8(temp*255), [path, image_Files(i).name])
%     out = target_results;
%     mesh(out)
%     view(-37.5, 30)
%     axis([0 256 0 256])
% %     set(gcf,'position',[0,0, 300, 300]);
%     saveas(gcf,[target_mesh_path strrep(image_Files(i).name,'png','tif')],'tiff');
%     saveas(gcf,['C:\Users\Health\Desktop\' strrep(image_Files(i).name,'png','tif')],'tiff');
%     %% adaptive threshold segmentation
%     target_results = double(target_results);
%     k = 500; % parametric 
%     v_min = 8;
%     mean = mean2(target_results);
%     std = std2(target_results);
%     t_adapt = max(v_min, mean * k + std);
%     target_seg = (target_results > t_adapt);
%     target_seg = target_seg * 255;
%     target_seg = uint8(target_seg);
%     imshow(target_seg)
%     imwrite(target_seg, [target_seg_path, num2str(i-1, '%04d') '.bmp']);
%     if i == 5
%         break;
%     end
end
