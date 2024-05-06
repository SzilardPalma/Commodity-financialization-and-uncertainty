function [ ir,hir,zir, irHigh, hirHigh, zirHigh, irLow, hirLow, zirLow  ] = ...
getGIRF( L,LH,LV,N,horizon1,...
    Fmat,Qmat,varcoef,iamat,A0,REPSx,Y,hlast,FF,scale,pos,pmatw,fload,NN,dataS, ...
    idc,indexc,hlastw, hlaste, resLoading, Financialization)

ir=0;
hir=0;
zir=0;
T=rows(Y);
%T=size(Financialization(Financialization==idx),1);
hh=L+1:12:T-horizon1; %computed every 12 months
%fload = 70*2*T
fload0=zeros(size(fload,1),N+1,size(fload,3));
fload0(:,1,:)=fload(:,1,:);
for i=1:N
    fload0(indexc==idc(i),i+1,:)=fload(indexc+1==idc(i)+1,2,:);
end    
%{
Y=Y(Financialization==idx,:);
hlast=hlast(Financialization==idx,:);
dataS=dataS(Financialization==idx,:);
fload0=fload0(:,:,Financialization==idx);
pmatw=pmatw(Financialization==idx);
resLoading=resLoading(Financialization==idx,:);
%}
for j=1:cols(hh)

    Y0=Y(hh(j)-L+1:hh(j),:);
    H0=hlast(hh(j)-LV+1:hh(j),1:N);
    %factor level
    Z0=dataS(hh(j)-L+1:hh(j),:);
    fload00=fload0(:,:,(hh(j)-L+1:hh(j)+horizon1));
    pmatw0=pmatw(hh(j)-L+1:hh(j)+horizon1);
    resLoading0=resLoading(hh(j)-L+1:hh(j)+horizon1,:);
   

    H0W=hlastw(hh(j)-L+1+1:hh(j));
    H0E=hlaste(hh(j)-L+1+1:hh(j),:);
 
 [ir1,hir1,zir1] =...
    volIRFWithMC( L,LH,LV,N,horizon1,Fmat,Qmat,...
    varcoef,iamat,A0,REPSx,Y0,H0,FF,scale,pos,...
    pmatw0, fload00, NN, Z0, H0W,H0E,resLoading0);


ir=ir+ir1;
hir=hir+hir1;
zir=zir+zir1;
end


hhx=cols(hh);
ir=ir/hhx;
hir=hir/hhx;
zir=zir/hhx;



