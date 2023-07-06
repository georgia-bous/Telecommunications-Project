%o kwdikas gia am 03119059
function thlep()

%ask 1

A = 4;
Ac=1;
fm = 5000;
Tm=1/fm;
fs1 = 150000;
Ts1=1/fs1;
fs2 = 250000;
Ts2=1/fs2;
fs3=4*fm;
Ts3=1/fs3;
L=1000;
Ad=1;


t = 0:0.0001:0.0008;
y = A*sawtooth(2*pi*fm*t,1/2);


t1=0:Ts1:4*Tm;
y1=A*sawtooth(2*pi*fm*t1,1/2);
figure (1)
stem(t1, y1)
title("Sample y at fs1")
xlabel("Time Sample");
ylabel("Amplitude");

t2=0:Ts2:4*Tm;
y2=A*sawtooth(2*pi*fm*t2,1/2);
figure (2)
stem(t2, y2)
title("Sample y at fs2")
xlabel("Time Sample");
ylabel("Amplitude");

figure(3)
stem(t1, y1)
hold on
stem(t2, y2)
title("Sample y at fs1 and fs2")
xlabel("Time Sample");
ylabel("Amplitude");
hold off

t3=0:Ts3:4*Tm;
y3=A*sawtooth(2*pi*fm*t3,1/2);
figure (4)
stem(t3, y3)
title("Sample y at 4*fm")
xlabel("Time Sample");
ylabel("Amplitude");


t = 0:0.00001:1-0.00001;
y = A*sawtooth(2*pi*fm*t,1/2);
Y=fft(y)/(length(y));
figure(5);
plot(abs(Y(2:end/2)))%abs->to metro, /2->einai symmetriko
title("Fourier Transform of y");
xlabel("Frequency(Hz)");
ylabel("Amplitude");

t=0:0.000001:0.0008;
z=Ac*sin(2*pi*fm*t);

t1=0:Ts1:4*Tm;
z1=Ac*sin(2*pi*fm*t1);
figure (6)
stem(t1, z1)
title("Sample z at fs1")
xlabel("Time Sample");
ylabel("Amplitude");

t2=0:Ts2:4*Tm;
z2=Ac*sin(2*pi*fm*t2);
figure (7)
stem(t2, z2)
title("Sample z at fs2")
xlabel("Time Sample");
ylabel("Amplitude");

figure(8)
stem(t1, z1)
hold on
stem(t2, z2)
title("Sample z at fs1 and fs2")
xlabel("Time Sample");
ylabel("Amplitude");
hold off

t3=0:Ts3:4*Tm;
z3=Ac*sin(2*pi*fm*t3);
figure (9)
stem(t3, z3)
title("Sample z at 4*fm")
xlabel("Time Sample");
ylabel("Amplitude");

t = 0:0.00001:1-0.00001;
z = Ac*sin(2*pi*fm*t);
Z=fft(z)/length(z);
figure(10);
plot(abs(Z(2:end/2)))
title("Fourier Transform of z");
xlabel("Frequency(Hz)");
ylabel("Amplitude");


t=0:0.000001:0.001;
help=Ad*sin(2*pi*(fm+L)*t);
z=Ac*sin(2*pi*fm*t);
q=help+z;

t1=0:Ts1:0.001;
help1=Ad*sin(2*pi*(fm+L)*t1);
z1=Ac*sin(2*pi*fm*t1);
q1=help1+z1;
figure (11)
stem(t1, q1)
title("Sample q at fs1")
xlabel("Time Sample");
ylabel("Amplitude");

t2=0:Ts2:0.001;
z2=Ac*sin(2*pi*fm*t2);
help2=Ad*sin(2*pi*(fm+L)*t2);
q2=help2+z2;
figure (12)
stem(t2, q2)
title("Sample q at fs2")
xlabel("Time Sample");
ylabel("Amplitude");

figure(13)
stem(t1, q1)
hold on
stem(t2, q2)
title("Sample q at fs1 and fs2")
xlabel("Time Sample");
ylabel("Amplitude");
hold off

t3=0:Ts3:0.001;
z3=Ac*sin(2*pi*fm*t3);
help3=Ad*sin(2*pi*(fm+L)*t3);
q3=help3+z3;
figure (14)
stem(t3, q3)
title("Sample q at 4*fm")
xlabel("Time Sample");
ylabel("Amplitude");


t = 0:0.00001:1-0.00001;
z = Ac*sin(2*pi*fm*t);
help=Ad*sin(2*pi*(fm+L)*t);
q=help+z;
Q=fft(q)/length(q);
figure(15);
plot(abs(Q(2:end/2)))
title("Fourier Transform of q");
xlabel("Frequency(Hz)");
ylabel("Amplitude");



%ask 2


R=5;
L=2^R;
ymax=4;
D=2*ymax/L;

t = 0:Ts1:4*Tm;
y = A*sawtooth(2*pi*fm*t,1/2);

partition=[-3.75:0.25:3.875];
codebook=[-3.875:0.25:4];
[index, quants]=quantiz(y, partition, codebook);


figure (16)
stem(t, quants)
title("y through mid riser")
xlabel("Time(s)")
ylabel("Amplitude")


h=uencode(quants, R, 4);
figure (17)
stem(t, h)
title("y through mid riser")
xlabel("Time(s)")
ylabel("Gray Code")
L = get(gca,'YTickLabel');
set(gca,'YTickLabel',cellfun(@(x) bin2gray(dec2bin(str2num(x))),L,'UniformOutput',false));

q=y-quants;
error_first10=q(1,1:10);
disp(error_first10)
error_first20=q(1, 1:20);
disp(error_first20)

P=rms(y);
disp(P);

%c
bitstream=[0 0 0 0 0,0 0 0 1 1,0 0 1 1 0,0 0 1 0 1,0 1 1 0 0,0 1 1 1 1,0 1 0 1 0,0 1 0 0 1,1 1 0 0 1,1 1 0 1 0,1 1 1 1 1,1 1 1 0 0,1 0 1 0 1,1 0 1 1 0,1 0 0 1 1,1 0 0 0 0,   1 0 0 1 1,1 0 1 1 0,1 0 1 0 1,1 1 1 0 0,1 1 1 1 1,1 1 0 1 0,1 1 0 0 1,0 1 0 0 1,0 1 0 1 0,0 1 1 1 1, 0 1 1 0 0, 0 0 1 0 1,0 0 1 1 0,0 0 0 1 1 ];
%vlepw apo to figure 17 gia thn prwth periodo se poio epipedo
%antistoixei kathe deigma kai metatrepw ayton ton arithmo ston antistoiho
%gray
figure(18)
PRZ(bitstream, fm/1000);%vrisketai sto telos tou programmatos
xlabel('Bit');
ylabel('Amplitude(V)');
hold off




%ask3
%a
x=randi([0 1], 36, 1);
disp(x');
N = length(x);
Tb = 0.25;   
nb = 100;   % Digital signal per bit
digit = []; 
for n = 1:1:N    
    if x(n) == 1;   
       sig = ones(1,nb);
    else x(n) == 0;
        sig = -ones(1,nb);
    end
     digit = [digit sig];
end
t1=Tb/nb:Tb/nb:N*(Tb);   % Time period 
figure(19);
plot(t1,digit,'lineWidth',2.5);
grid on;
axis([0 Tb*N -1.5 1.5]);
xlabel('Time(s)');
ylabel('Amplitude(V)');
title('Digital Input Signal');
 
Ac = 1;      
br = 1/Tb;    
Fc = 1; 


%BPSK
Pc1 = 0;      % Carrier phase for binary input '1'
Pc2 = pi;     % Carrier phase for binary input '0'
t2 = 0:Tb/nb:Tb-Tb/nb;   % Signal time                 
mod = [];
for (i = 1:1:N)
    if (x(i)==1)
        y = Ac*cos(2*pi*Fc*t2+Pc1);   % Modulation signal with carrier signal 1
    else
        y = Ac*cos(2*pi*Fc*t2+Pc2);   % Modulation signal with carrier signal 2
    end
    mod=[mod y];
end
t3=Tb/nb:Tb/nb:Tb*N;   % Time period
figure(20)
plot(t3,mod);
xlabel('Time(s)');
ylabel('Amplitude(V)');
title('BPSK Modulated Signal');



%QPSK
Pc1 = 0;     
Pc2 = pi/2;
Pc3=3*pi/2;
Pc4=pi;
t2 = 2*Tb/nb:2*Tb/(nb):2*Tb;   % Signal time                 
mod = [];
for (i = 1:2:N-1)
    if (x(i)==0&& x(i+1)==0)
        y = Ac*cos(2*pi*Fc*t2+Pc1);   
    elseif(x(i)==0&&x(i+1)==1)
        y = Ac*cos(2*pi*Fc*t2+Pc4);   
    elseif(x(i)==1&&x(i+1)==1)
        y = Ac*cos(2*pi*Fc*t2+Pc3); 
    else
        y = Ac*cos(2*pi*Fc*t2+Pc2); 
    end
    mod=[mod y];
end
t3=2*Tb/nb:2*Tb/nb:Tb*N;   % Time period
figure(21)
plot(t3,mod);
xlabel('Time(s)');
ylabel('Amplitude(V)');
title('QPSK Modulated Signal');




%8-PSK
Pc1 = 0;     
Pc2 = pi/4;
Pc3=2*pi/4;
Pc4=3*pi/4;
Pc5=pi;
Pc6=5*pi/4;
Pc7=6*pi/4;
Pc8=7*pi/4;
t2 = 3*Tb/(nb):3*Tb/(nb):3*Tb;   % Signal time                 
mod = [];
for (i = 1:3:N-2)
    if (x(i)==0&& x(i+1)==0&&x(i+2)==0)
        y = Ac*cos(2*pi*Fc*t2+Pc1);   
    elseif(x(i)==0&&x(i+1)==0&&x(i+2)==1)
        y = Ac*cos(2*pi*Fc*t2+Pc2);   
    elseif(x(i)==0&&x(i+1)==1&&x(i+2)==1)
        y = Ac*cos(2*pi*Fc*t2+Pc3); 
    elseif(x(i)==0&&x(i+1)==1&&x(i+2)==0)
        y = Ac*cos(2*pi*Fc*t2+Pc4); 
    elseif(x(i)==1&&x(i+1)==1&&x(i+2)==0)
        y = Ac*cos(2*pi*Fc*t2+Pc5); 
    elseif(x(i)==1&&x(i+1)==1&&x(i+2)==1)
        y = Ac*cos(2*pi*Fc*t2+Pc6); 
    elseif(x(i)==1&&x(i+1)==0&&x(i+2)==1)
        y = Ac*cos(2*pi*Fc*t2+Pc7);   
    else
        y = Ac*cos(2*pi*Fc*t2+Pc8); 
    end
    mod=[mod y];
end
t3=3*Tb/nb:3*Tb/nb:Tb*N;   % Time period
figure(22)
plot(t3,mod);
xlabel('Time(s)');
ylabel('Amplitude(V)');
title('8-PSK Modulated Signal');



%b
A=5;
ypam=A*digit;
t=Tb/nb:Tb/nb:N*Tb;
figure(23)
title('PAM Waveform');
plot(t,ypam,'lineWidth',2.5);
xlabel('Time');
ylabel('Amplitude');
ylim([-A-0.2 A+0.2])


%c

scatterplot(ypam);
grid on


%d
k=1;
EbN_ratio1=5;
EbN_ratio2=15;
snr1=EbN_ratio1+10*log(k);
snr2=EbN_ratio2+10*log(k);
awgnchan1 = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)','SNR', snr1);
out1=awgnchan1(ypam);
awgnchan2 = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)','SNR', snr2);
out2=awgnchan2(ypam);


figure(24)
subplot(3,1,1)
plot(t1, ypam)
xlabel('Time(s)');
ylabel('Amplitude(V)');
title('B- PAM without noise');
subplot(3, 1,2)
plot(t1, out1)
xlabel('Time(s)');
ylabel('Amplitude(V)');
title('B-PAM with Eb/N0=5(dB)');
subplot(3, 1, 3)
plot(t1, out2)
xlabel('Time(s)');
ylabel('Amplitude(V)');
title('B-PAM with Eb/N0=15(dB)');



%e
snr1=EbN_ratio1+3+10*log(k);
awgnchan1 = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)','SNR', snr1);
out1 = awgnchan1(ypam);
scatterplot(out1)
snr2=EbN_ratio2+3+10*log(k);
awgnchan2 = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)','SNR', snr2);
out2 = awgnchan2(ypam);
scatterplot(out2)


%st
%the estimated BER
N = 10^6; % number of bits or symbols
ip = rand(1,N)>0.5; % generating 0,1 with equal probability
s = 2*ip-1; % 2-PAM modulation 0 -> -1; 1 -> 1
Eb_N0_dB = [0:1:15]; 
for ii = 1:length(Eb_N0_dB)
awgn = 1/sqrt(2)*[randn(1,N) + 1i*randn(1,N)];% AWGN with variance (power) 1
y = s + 10^(-Eb_N0_dB(ii)/20)*awgn; %Pass the modulated signal through an AWGN channel
ipHat = real(y)>0;%Demodulate the received signal.
nErr(ii) = size(find([ip- ipHat]), 2); % counting the errors
end
simBer = nErr/N;

berTheory = berawgn(Eb_N0_dB,'pam',2);

figure(25)
semilogy(Eb_N0_dB,simBer,'*')
hold on
semilogy(Eb_N0_dB,berTheory)
grid
legend('Estimated BER','Theoretical BER')
xlabel('Eb/No (dB)')
ylabel('Bit Error Rate')



%ask4

%a
A=1;
Tb=0.25;
Ebit=Tb*A^2;
mod=comm.QPSKModulator('PhaseOffset',pi/4,'BitInput',true, ...
    'SymbolMapping', 'Gray'); % mod(x) to diamorfomeno kata qpsk
constDiagram0 = comm.ConstellationDiagram('Title','QPSK Constellation', 'ShowReferenceConstellation',...
 false, 'XLimits',[-1.5*sqrt(Ebit) 1.5*sqrt(Ebit)], ...
'YLimits',[-1.5*sqrt(Ebit) 1.5*sqrt(Ebit)]);


constDiagram0(mod(x))

    

%b prepei na trexei xehorista gia na fanoun ta diagrammata asterismon
k = 2; %bits per symbol - QPSK
EbN_ratio1=5; %Energy of bit to No ratio (dB)
EbN_ratio2=15; %Energy of bit to No ratio (dB)

awgnchan1 = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (Eb/No)','EbNo', EbN_ratio1,...
'BitsPerSymbol',k); %AWGN Noise1
awgnchan2 = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (Eb/No)','EbNo', EbN_ratio2,...
'BitsPerSymbol', k); %AWGN Noise2

out1 = awgnchan1(mod(x));
out2 = awgnchan2(mod(x));
constDiagram1 = comm.ConstellationDiagram('Title','QPSK Constellation with Eb/N0=5dB', 'ShowReferenceConstellation',...
 false);
constDiagram1(out1);
constDiagram2 = comm.ConstellationDiagram('Title','QPSK Constellation with Eb/N0=15dB', 'ShowReferenceConstellation',...
 false);
constDiagram2(out2);


%c

l=100000;
snrdb=1:1:15;
snrlin=10.^(snrdb/10);%den thelw se db
for n=1:1:15
    si=2*(round(rand(1,l))-0.5);                      
    sq=2*(round(rand(1,l))-0.5);                                    
    s=si+j*sq;      
    awgnchan = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)','SNR', n);
    w=awgnchan(s);
    r=w;                                           
    si_=sign(real(r));                                
    sq_=sign(imag(r));                               
    ber1=(l-sum(si==si_))/l;                          
    ber2=(l-sum(sq==sq_))/l;                          
    ber(n)=mean([ber1 ber2]);                         
end

tber=0.5.*erfc(sqrt(snrlin));
figure(26)
semilogy(snrdb,ber,'-bo',snrdb,tber,'-mh')
title('QPSK with awgn');
legend('Estimated BER','Theoretical BER')
xlabel('Signal to noise ratio');
ylabel('Bit error rate');      
grid on;




%ask5

%a
Fs=44100;
[y, Fs]=audioread('soundfile2_lab2.wav');
Ts=1/Fs;
ly=size(y);
dt=1./Fs;
t=(0:ly-1)*dt;
figure(27)
plot(t, y)
xlabel('Time(s)')
ylabel('Amplitude')
title('Audio Signal')


%b
R=8;
L=2^R;
max_value = max(y);
min_value = min(y);
D = (max_value - min_value)/L; %quantizing step
partition = min_value:D:max_value;
codebook = min_value-D:D:max_value;
[index, quants]=quantiz(y, partition, codebook);


figure (28)
stem(t, quants)
title("Audio signal through mid riser")
xlabel("Time(s)")
ylabel("Amplitude")


h=uencode(quants, R, 4);
figure (29)
stem(t, h)
title("Audio signal through mid riser")
xlabel("Time(s)")
ylabel("Gray Code")
L = get(gca,'YTickLabel');
set(gca,'YTickLabel',cellfun(@(x) bin2gray(dec2bin(str2num(x))),L,'UniformOutput',false));


end



function g=bin2gray(b)
g(1)=b(1);
for i=2: length(b);
    x=xor(str2num(b(i-1)), str2num(b(i)));
    g(i)=num2str(x);
end
end




function PRZ(h, fm)
clf;
n=1;
l=length(h);
h(l+1)=1;
while n<=length(h)-1;
    t=n-1:0.001:n;
if h(n) == 0
    if h(n+1)==0  
        y=fm*(-(t<n-0.5)-(t==n));
    else
        y=fm*(-(t<n-0.5)+(t==n));
    end
    plot(t,y,'b');grid on;
    title('Line code POLAR RZ');
    hold on;
    axis([0 length(h)-1 -(fm+1) (fm+1)]);
else
    if h(n+1)==0
        y=fm*((t<n-0.5)-1*(t==n));
    else
        y=fm*((t<n-0.5)+1*(t==n));
    end
    plot(t,y,'b');grid on;
     title('Line code POLAR RZ');
    hold on;
    axis([0 length(h)-1 -(fm+1) (fm+1)]);
end
n=n+1;
end
end



