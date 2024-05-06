clear
addpath('./functions');
addpath('./functions1');

load posterior

horizon1=46; %IRF horizon
scale=3; %scale of shock in SD
pos=2;  %equation number to shock 
fsize=1000; %number of gibbs
REPSx=100; %number of monte-ccarlo reps to compute GIRF
irfmat=zeros(fsize,horizon1,nfact*2+NN);
irfmatH=zeros(fsize,horizon1,nfact*2+NN);
irfmatL=zeros(fsize,horizon1,nfact*2+NN);
X=datetime(2001,6,30):calmonths(1):datetime(2023,12,31);
Financialization=0;

for jgibbs=1:fsize
    %factors
    pmat=squeeze(pmatsave(jgibbs,:,:));
    pmatw=squeeze(pmatwsave(jgibbs,:,:));
    fload=squeeze(floadsave(jgibbs,:,:,:));
    %world 
    hlastw=squeeze(hwsave(jgibbs,:,:));
    %beta2w=squeeze(beta2wesave(jgibbs,:,:));
    qw=squeeze(qsave(jgibbs,:,:));
    %idiosyncratic
    hlaste=squeeze(hesave(jgibbs,:,:)); 
    qe=squeeze(qsave(jgibbs,:,:));
    resLoading=squeeze(resloadingsave(jgibbs,:,:));
    % VAR
    iamat=squeeze(amatsave(jgibbs,:,:));
    beta2=squeeze(betasave(jgibbs,:,:));
    fbig=squeeze(gammasave(jgibbs,:,:));
    Qbig=squeeze(qsave(jgibbs,:,:))';
    hlast=squeeze(hvarsave(jgibbs,:,:));
    
    A0=iamat';
    betaf1=reshape(fbig,NW*LV+(NW+1),NW);
    Fmat=betaf1;
    Qmat=Qbig;
    
    FF=comp(beta2,NW,L,EX);
    %varcoef=reshape(betaVAR,NW*L+EX,NW);
    varcoef=reshape(beta2,NW*L+EX,NW);


    jgibbs
 [ ir,hir,zir] = getGIRF( L,LH,LV,NC,....
     horizon1,Fmat,Qmat,varcoef,iamat,A0,REPSx,yF,...
     hlast(1:end,:),FF,scale,pos,...
     pmatw,fload,NN,dataS,idc,indexc, ...
     hlastw, ...
     hlaste, resLoading, Financialization); %IR=response of level, hir=response of stochastic vol, vir=response of
 %unconditional volatility, dir=response of log determinant'
 tmp = [ir hir zir];

    irfmat(jgibbs,:,:)=tmp;
    
   
end


save IRFResults


