clear

path = 'D:\Infrared_Target_Detection\Infrared target Detection\Contrast_Experiment\SIRST\';

mask_path = [path 'mask\'];
image_path = [path 'images\'];
sub_path = 'mask/';
image_sub_path = 'images/';
% Files= dir(strcat(mask_path,'*.png'));
image_Files = dir(strcat(image_path,'*.png'));

for i=1:length(image_Files)    
    f = fopen([path 'name_list'],'a+');
    fprintf(f, '%s\t%s\n', ['../Datasets/SIRST/images/' image_Files(i).name], ['mask/' image_Files(i).name]);
    fclose(f); 
end