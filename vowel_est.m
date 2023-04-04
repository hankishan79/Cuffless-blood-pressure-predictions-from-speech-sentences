function [s, loc]=vowel_est(b,Fs)

len=length(b)/Fs; %We get time of audio in seconds
sam_len = floor(((len*1000)- 10)/(10));
max = zeros(10);
getans=zeros(sam_len,1);

%calculating the spectral peaks corresponding to a signal with interval of
%20 ms.
for i=2:sam_len
      b_test=b(Fs*(i-1)*0.01 : Fs*(i+1)*0.01);
      b_test_fft = fft(b_test,256);
      b_test_fft2=b_test_fft(1:128);
      [sortedX,sortingIndices] = sort(b_test_fft2,'descend');
      maxValues = sortedX(1:10);
      for temp=1:10
        getans(i,1)=getans(i)+abs(maxValues(temp));
      end
end
%Enhancing the signal by using the normalisation technique
k=length(getans);
reference = getans(2);
reference_index = 2;
getans=smooth(getans);
temp=40;
down=0;
getans_new=zeros(k,1);
i=0;
while(i<k)
    if (down==0 && temp<(k-1))
    while(getans(temp)< getans(temp+1))
        temp=temp+1;
        if(temp==k)
            break;
        end
    end
    if((getans(temp)-getans(reference_index)) > 10)
    for now=reference_index:(temp)
        if (now < k)
        getans_new(now,1) = (getans(now)-getans(reference_index))/(getans(temp)-getans(reference_index));
        end
    end
    else
        for now=reference_index:(temp)
        if (now < k)
        getans_new(now,1) = (getans_new(reference_index-1));
        end
        end 
    end
    reference_index=temp+1;
    i = temp+1;
    down=1;
    end
    if(down==1 && temp<(k-1))
        while(getans(temp) > getans(temp+1))
            temp=temp+1;
            if(temp==k)
                break;
            end
        end
        if((getans(temp)-getans(reference_index)) < -10)
        for now=(reference_index):temp
            getans_new(now,1)=(getans(now)-getans(temp))/(getans(reference_index)-getans(temp));
        end
        else
        for now=reference_index:(temp)
        if (now < k)
        getans_new(now,1) = (getans_new(reference_index-1));
        end
    end
        end
        reference_index=temp+1;
        i = temp+1;
        down=0;
    end
end

% Further smoothing of the signal by convulution with that of the gaussian
% window
f_new = smooth(getans_new);
windowWidth = 10;
halfWidth = windowWidth / 2
gaussFilter = gausswin(5)
gaussFilter = gaussFilter / sum(gaussFilter);
y = conv(gaussFilter,f_new);

%Axis Alignment
x_axis=0:1/Fs:len;
if(length(x_axis)~=length(b))
    q=length(x_axis)-length(b);
    if(q>0)
        x_axis=0:1/Fs:len-q/Fs;
    else
        x_axis=0:1/Fs:len+q/Fs;
    end
end

x_axis2=0:len/length(f_new):len;
if(length(x_axis2)~=length(f_new))
    q=length(x_axis2)-length(f_new);
    if(q>0)
        x_axis2=0:len/length(f_new):len-(q*len/length(f_new));
    else
        x_axis2=0:len/length(f_new):len+(q*len/length(f_new));
    end
end

x_axis3=0:len/length(y):len;
if(length(x_axis3)~=length(y))
    q=length(x_axis3)-length(y);
    if(q>0)
        x_axis3=0:len/length(y):len-(q*len/length(y));
    else
        x_axis3=0:len/length(y):len+(q*len/length(y));
    end
end

f1=smooth(getans);

findpeaks(y,x_axis3)
[idx,loc]=findpeaks(y,x_axis3);
deger=0;
for i=1:length(loc),
    if(loc(i)>0.51)
        deger=loc(i);
        break;
    end
end
% s=b(deger*48000:deger*48000+4900,1);    onset point detection
s=b(deger*48000:deger*48000+2500,1);   % onset point detection-50ms 

