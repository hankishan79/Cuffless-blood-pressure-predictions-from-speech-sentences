clear all;
close all;
clc;

% addpath('F:\Haydar_Flash_Disk_25_Aralik_2014\flash\scii\databases/S_H_B_Database'); 
addpath('C:\Users\hankishan\Desktop\S_H_B_Database_10132022/Turkish Sentence'); 
addpath('C:\Users\hankishan\Desktop\S_H_B_Database_10132022/databases'); 

%% READ EXCEL 1
filename = 'Blood-Pressure-Data.xlsx';
% sheets = sheetnames(filename)
sheet = 'Sayfa1';
% xlRange = 'B2:C3';

[num,txt,raw] = xlsread(filename, sheet);
vect_1=[];
vect_2=[];
for i=1:length(txt)-2,
    if((num(i,26)+num(i,30))/2>=115)
       SBP=2;
    else
       SBP=1;
    end
    if((num(i,27)+num(i,31))/2>=72)
       DBP=2;
    else
       DBP=1;
    end
    if((num(i,27)+num(i,31))/2<80 && (num(i,26)+num(i,30))/2<120)
       BP=1;
    else
       BP=2;
    end
     vect_2=[vect_2;num(i,6) num(i,36) num(i,14) num(i,15) (num(i,26)+num(i,30))/2 ...
        (num(i,27)+num(i,31))/2 (num(i,28)+num(i,32))/2 ...
        SBP DBP BP];     
end
for i=3:length(txt),
    vect_1=[vect_1;raw{i,2}];    
end

fp=fopen('feature_train_all.txt','wt');   % txt dosya olusturur isimi biz veriyoruz

[ FList ] = ReadImageNames('databases');
nam=[];
for i=1:length(FList),
    file = string(FList(i));    
    [filepath,name,ext] = fileparts(file);
    nam=[nam;name];
end
ft_vect=[];
% size(vect_1,1)
for i=1:size(vect_1,1),         
        a=['C:\Users\hankishan\Desktop\S_H_B_Database_10132022\' char(FList(i,1))];
        [s,fs]=audioread(a);    
        s=s(:,1);
        % normalization
%         maks=max(abs(s));
%         s=s/maks;
        s=normalize(s);
        [sss,loc]=vowel_est(s,fs);
        signals=[];
%         j=1:length(loc)-1
        for j=2:length(loc)-1,
            signals=[signals;s(loc(j)*48000-4800:loc(j)*48000,1)];
        end
%         for j=1:4800:length(s)-4800,
%             signals=[signals;s(j:j+4800,1)];
%         end
        L=length(signals);
        inn=char(nam(i));
                %% Excel read        
        idxx=0;
        for jj=1:length(vect_1),    
            po = strcmpi(inn,vect_1(jj,:));% Compare user input string with entries in the Excel sheet
            if(po)
            idxx=jj;
            break;
            end
        end
        if(idxx>0)
            ft_vect=vect_2(idxx,:);
        end
        if(idxx==0)
            ft_vect=zeros(1,10);
        end
        
        for k=1:2400:L-2500,  %% 50 ms olarak boldugumuzde her bir bolut icin 9600 data girer
            %%%islem yap
            %% audioIn
            audioIn=signals(1+k:2400+k,1);
%             sound(audioIn,48000);           
%             pause(1);                        
            [entr,coeffs,delta,deltaDelta,loc]=features(audioIn,fs);
            size(entr)
            fprintf(fp,'%s',nam(i));
            fprintf(fp,'\t%f',i);
            fprintf(fp,'\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f',[entr(1:22) ft_vect(1:10)]);
            fprintf(fp,'\n');
        end
    
end

fclose(fp);


