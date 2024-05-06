clear
addpath('./functions');
addpath('./functions1');
dfolder='./data/';
sfolder='./results/';
file=1;
%Load data and transform%%%%%%%
dataS=xlsread("Input2","Input","B2:BS312");
[~,titles]=xlsread("Input2","Input","B1:BS1");
dataS=standardise(dataS); %standardise data
%%estimation options%%%%
T0=40;   %training sample
L=2;     %lag for transition equation
Lx=1;    %lag for idiosyncratic component transition eq
REPS=20000; %Reps
BURN=10000; %burn-in
SKIP=10;  %every SKIP draw is kept after burn-in 
maxdraws=100; %max trys to find stable coefficients
nfact = 6; % number of factors
CHECK=1;
Sindex=BURN+1:SKIP:REPS;
fsize=length(Sindex);
NC=nfact; %number of countries
NN=cols(dataS); %number of series
%idc=vec(repmat(1:NC,nfact,1)); %index of countries
idw = [25 31 37]; %intervals between which the world factors are
indexc=[ones(1,5).*1 ones(1,6).*2 ones(1,19).*3  ones(1,25).*4 ...
    ones(1,11).*5 ones(1,4).*6 ]';
nfact = max(indexc); % number of factors
NC=nfact; %number of countries
idc= (1:NC)';%any number different from 0s
totalFactors=nfact; %world+ regional+idiosyncratic;
npart=40;  %number of particles
nw=1;

%% Starting Values and Priors

%initial estimate of the factors;

pmatw = extract(dataS,1);      %PC estimator of world factor
BB=pmatw\dataS;
dataS = dataS - pmatw*BB;


% extract country factor
pmat=zeros(rows(dataS),NC); %PC for countries
for i=1:NC % TODO: refactor NC to represent true value
    dataSS=dataS(:,indexc==idc(i));  
    %tmp=zeros(rows(dataSS), nfact);
    tmp=extract(dataSS,1); 
    pmat(:,i)=tmp;
    BB=pmat(:,i)\dataSS;
end


%Mumtaz2018 priors, l.331 in Mumtaz code
resLoading=zeros(rows(pmat),NN); %idiosyncratic
scaling=3.5e-04;
QB=cell(NN,1);
Q0B=QB;
FLOAD0=zeros(NN,totalFactors);
PFLOAD0=cell(NN,1);
for j=1:NN
    yy=dataS(:,j);
    xx=[pmatw pmat(:,idc==indexc(j))];
    BB=xx\yy;
    resLoading(:,j)=yy-xx*BB;
        %FLOAD0(j,1:cols(xx))=BB';
    SS=(resLoading(:,j)'*resLoading(:,j))/(rows(yy)-cols(xx));
    VV=SS*invpd(xx'*xx);
    FLOAD0(j,1:cols(xx))=BB';
    PFLOAD0{j}=VV;
    Q0B{j}=VV*scaling*(cols(VV)+1);
    QB{j}=VV*scaling*(cols(VV)+1);
end
VFLOAD0=eye(nfact).*10; %prior variance


%% VAR priors:
%%%%%%%%%%%%%%%%%%%
% World factor VAR:
YFactor = [pmat(1:L,:); pmat];
XFactor =[ lag0(YFactor,1) lag0(YFactor,2) ones(size(YFactor,1),1) ]; % 2 lags
% same dimensions
YFactor=YFactor(3:end,:);
XFactor=XFactor(3:end,:);
NW = cols(YFactor); %nfact
L=2; %lags of endogenous variables
LH=-1; %contemporaneous value plus lags of volatility in mean -1 to drop all regressors, 0 to drop lags only 
LV=1; %lags of data in vol equations 0 to drop completely
CHECK=1;
maxdraws=1000;

T0=40; %training sample
EX = NW*LH+1; %P: num of VAR coefs
EX=(NW*(LH+1))+1;


%step 1 set starting values
y0Factor=YFactor(1:T0,:);
x0Factor=XFactor(1:T0,:);
b0F=x0Factor\y0Factor;
e0F=y0Factor-x0Factor*b0F;
sigma0F=(e0F'*e0F)/T0;



%Minnesota type prior for the VAR coefficients in mean equation via dummy observations
LAMDAP=0.1;
TAUP=0;
EPSILON=1/1000;
EPSILONH=1; %closer to 0 tighter prior on lagged vol in mean equations
RW=0;
[yd,xd,BETA0,SIGMA0]=getdummies(LAMDAP,TAUP,EPSILON,...
    YFactor,L,EX,LH+1,EPSILONH,RW);



%starting value for the stochastic volatility
yF=YFactor(T0+1:end,:);
xF=XFactor(T0+1:end,:);
T=rows(xF);

[outh,outf,outg]=getinitialvol(YFactor,500,400,T0,L);

%define priors for variances of shock to transition eq
vg0=NW+1;             %prior df
g00=diag(outg*vg0);  %prior scale paramter for Q[i]~IG(g0,vg0)
h0=outh(2:end,:);


Qbig=diag(outg);


%priors for transition equation
LAMDAPV=0.1; 
TAUPV=0;
EPSILONV=1/100; %prior on constant
EPSILONVY=0.05; %prior on lagged data entering svol eqs closer to 0 is tighter, was 0.05
[ydv,xdv,bv0,sv0]=getdummiesVOLF(LAMDAPV,TAUPV,EPSILONV,log(h0),1,LV,EPSILONVY,outf(1:NW),outg);


hlast=zeros(T,NW*(LH+1));
hlast(:,1:NW)=h0;
i=1;
for j=NW+1:NW:cols(hlast)
    hlag=lag0(h0,i);
    hlag=packr(hlag);
    hlag=[repmat(hlag(1,:),i,1);hlag];
    hlast(:,j:j+NW-1)=hlag;
    i=i+1;
end
hnew=hlast;

xvar=[xF(:,1:cols(XFactor)-1) log(hlast(1:end,1:NW*(LH+1))) xF(:,end:end)];
xfix=log(hlast);


varcoef0=xvar\yF;
res=yF-xvar*varcoef0;
varcoef0=vec(varcoef0)'; %initial estimates of VAR coefficients

mu0=(diag(sigma0F))';
for j=1:LH
    mu0=[mu0 (diag(sigma0F))'];  %ln(H0)~N(mu0,s0)
end
s0=eye(cols(mu0))*0.1;
is0=inv(s0);
NS=cols(hlast);



% A matrix
A0 = eye(nfact);
VA0 = 1*ones(nfact);
p=L;
k = 1+NW*((LH+1+L));

%%%%%%%%%%%%%%%%%%%%%%
% END OF VAR PARAMS
%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%
% world factor AR
scale=3.5e-04;

b00c=cell(nw,1);
s00c=cell(nw,1);
p00c=cell(nw,1);
Q0c=cell(nw,1);
Qc=zeros(L+1,L+1,nw);
for j=1:nw
        [y0c,x0c]=preparex(pmat(1:T0,j),L,1);
        [b00c{j},s00c{j},p00c{j}]=getols(y0c,x0c);
        Q0c{j}=scale*p00c{j}*T0;  %Qc~IW(Q0c{j},T0) for j=1,2,...NC
        Qc(:,:,j)=scale*p00c{j}*T0; %starting value for Qc
end

%idiosyncratic
b00e=cell(NN,1);
s00e=cell(NN,1);
p00e=cell(NN,1);
Q0e=cell(NN,1);
Qe=zeros(Lx,Lx,NN);
for j=1:NN
    [y0e,x0e]=preparex(resLoading(1:T0,j),Lx,0);
    [b00e{j},s00e{j},p00e{j}]=getols(y0e,x0e);
    Q0e{j}=scale*p00e{j}*T0; %Qe~IW(Q0e{j},T0) for j=1,2,...NN
    Qe(:,:,j)=scale*p00e{j}*T0; %Starting values
end


% Idiosyncratic factor AR
%priors for TVP parameters
scale=3.5e-04;

b00e=cell(NN,1);
s00e=cell(NN,1);
p00e=cell(NN,1);
Q0e=cell(NN,1);
Qe=zeros(Lx,Lx,NN);
for j=1:NN
    [y0e,x0e]=preparex(resLoading(1:T0,j),Lx,0);
    [b00e{j},s00e{j},p00e{j}]=getols(y0e,x0e);
    Q0e{j}=scale*p00e{j}*T0; %Qe~IW(Q0e{j},T0) for j=1,2,...NN
    Qe(:,:,j)=scale*p00e{j}*T0; %Starting values
end

%remove training sample
dataS=dataS(T0+1:end,:);
pmat=pmat(T0+1:end,:);
pmatw=pmatw(T0+1:end,:);
resLoading=resLoading(T0+1:end,:);
T=rows(dataS);

%priors and starting values for stochastic volatilties as residual^2+small
% world
hlastc=zeros(T+1,nw);
% multiplying stochvol factor by the number of world factors
for j=1:nw
        [y0c,x0c]=preparex(pmat(:,j),L,1);
        [~,~,~,epsc]=getols(y0c,x0c);%regression of factor on lags
        hlastcc=epsc.^2+0.0001;%residual^2+small number
        hlastcc=[hlastcc(1:L+1,:);hlastcc];
        hlastc(:,j)=hlastcc;
end



%idiosyncratic
hlaste=zeros(T+1,NN);
for j=1:NN
    [y0e,x0e]=preparex(resLoading(:,j),Lx,0);
    [~,~,~,epse]=getols(y0e,x0e);%regression of factor on lags
    hlastee=epse.^2+0.0001;%residual^2+small number
    hlastee=[hlastee(1:Lx+1,:);hlastee];
    hlaste(:,j)=hlastee;
end

SS0=10;    %variance of initial condition of SVOL
g0=0.01^2;  %prior scale parameter for inverse gamma prior for g
Tg0=1;     %prior degrees of freedom
gc=ones(1,1).*g0;  %starting values
ge=ones(NN,1).*g0;   %starting values


% loadings
beta0c=zeros(T,L+1,nw);

for j=1:nw
    beta0c(:,:,j)=repmat(b00c{j}',T,1);
end


beta0e=zeros(T,Lx,NN);
for j=1:NN
    beta0e(:,:,j)=repmat(b00e{j}',T,1);
end

%initial conditions for the factors
pmat00=[pmatw(L,:) pmat(L,:)];
for j=1:L-1
    pmat00=[pmat00 [pmatw(L-j,:) pmat(L-j,:)]];
end
vmat00=eye(cols(pmat00))*1;


save priors

pmatsave=zeros(fsize,T,nfact);
pmatwsave=zeros(fsize,T,nw);

hwsave=zeros(fsize,T+1,nw);
hvarsave=zeros(fsize,T,nfact);

floadsave=zeros(fsize,NN,2,T);

amatsave=zeros(fsize,nfact,nfact);
betasave=zeros(fsize,1,NW*(EX+NW*L)); % NW*(EX+NW*L) = NW*NW*(1+LH+L+1)
gammasave=zeros(fsize,1,NW*(NW+NW*LV+1));

hesave=zeros(fsize,T+1,NN);
qsave=zeros(fsize,NC,nfact); 
beta2esave=zeros(fsize,T,NN);
qesave=zeros(fsize,1,NN);
gesave=zeros(fsize,NN,1);

resloadingsave=zeros(fsize,T,NN);

beta2wsave=zeros(fsize,T,L+1);
qwsave=zeros(fsize,L+1,L+1);
gcsave=zeros(fsize,1,1);
%this above should be just %NC+nfact

jgibbs=1;
igibbs=1;

%% GIBBS

while jgibbs<=fsize
%%%%%%%%%%%%%%%%% Gibbs Step 1: Draw TVP-VAR-SV %%%%%%%%%%%%%%%%%
%% 1.A: World Factor
% 1. Draw (A | y_t, h_t)
A = (xvar'*xvar + .01*speye(k))\(xvar'*yF);
U = yF-xvar*A;
alpha = A(:);
[T,n] = size(yF);

for i=1:nfact
    EiOhi = U'*sparse(1:T,1:T,1./hlast(1:end,i));       
    Kbi = sparse(1:n,1:n,1./VA0(i,:)) + EiOhi*U; 
    mui = Kbi\(A0(i,:)./VA0(i,:))'; 
    Ci = chol(Kbi,'lower')/sqrt(T);
    Gam_mi = A0([1:i-1 i+1:end],:)';
    Gam_miperp = null(Gam_mi');
        
   V = zeros(n,n); zeta = zeros(n,1);        
   for jj=1:n
       if jj==1                
           v1 = Ci\Gam_miperp; v1 = v1/norm(v1);
           V = [v1 null(v1')];
           zetaj_hat = mui'*(Ci*v1);
           zeta(1) = anormrnd(zetaj_hat,1/T);
       else
           zetaj_hat = mui'*(Ci*V(:,jj));
           zeta(jj) = zetaj_hat + 1/sqrt(T)*randn;
       end
   end
        phii = (Ci')\sum(V.*repmat(zeta',n,1),2);
        % B0(ii,:) = phii;
        A0(i,:) = phii*sign(phii(i)); % fix the sign of the i-th element to be positive
end   
iamat = inv(A0);


%% Mumtaz 2016

%Step 2 VAR coef
[betaVAR,res,roots,problem]=carterkohnvar(yF,xvar,0,iamat,[hlast(1,1:NW);hlast(:,1:NW)],BETA0',SIGMA0,L,CHECK,maxdraws,EX);
if problem
    betaVAR=varcoef0;
else
    varcoef0=betaVAR;
end



% Step 3: Transition equation

Y0=log(hlast(1:end,1:NW));
  
  X0=[lag0(Y0,1) xvar(:,1:NW*LV) ones(rows(Y0),1) ];
  Y0=Y0(2:end,:);
  X0=X0(2:end,:);
  Y0s=[Y0;ydv];
  X0s=[X0;xdv];
  iX0=invpd(X0s'*X0s);
  mstar=vec(iX0*(X0s'*Y0s));
  vstar=kron(Qbig,iX0);
  %draw beta but ensure stability
  chck=-1;
  while chck<0
   betaf=mstar+(randn(1,NW*(NW*LV+(NW+1)))*chol(vstar))';
   
   if ~stability(betaf,NW,1,NW*LV+1)
       chck=10;
   end
  end
  
  
  betaf1=reshape(betaf,NW*LV+(NW+1),NW);
  
%sample Qbig~variance of shocks to transition eq
resv=Y0-X0*betaf1;
scalev=resv'*resv+g0;
Qbig(1:end,1:end)=iwpq(T+vg0,invpd(scalev)); % TODO: to make Q diagonal


%Step 4 Stochastic vol%

Fmat=zeros(NS,NS);
Fmat(NW+1:NS,1:NS-NW)=eye(NS-NW);
Fmat(1:NW,1:NW)=betaf1(1:NW,:)';
fit=X0(:,NW+1:end)*betaf1(NW+1:end,:);
fit=[fit(1,:);fit];
Qmat=zeros(NS,NS);
Qmat(1:NW,1:NW)=Qbig;
iQmat=zeros(NS,NS);
iQmat(1:NW,1:NW)=invpd(Qbig);
cQmat=zeros(NS,NS);
cQmat(1:NW,1:NW)=chol(Qbig);
varcoef=reshape(betaVAR,NW*L+EX,NW); 

%Sample volatility via particle Gibbs step
[betax,BB,W]=pftest3F_0(Fmat,fit,cQmat,iQmat,iamat,xfix,yF,xF,npart,NS,log(mu0),varcoef,NW,LH);
index=discretesample(W(:,end),1);
xfix=betax(:,:,index);

%update
hlast=exp(xfix);
xvar=[xF(:,1:cols(XFactor)-1) log(hlast(1:end,1:NW*(LH+1))) xF(:,end:end)];


%% World factor

% TVP parameters
beta2c=zeros(T,L+1,nw);
errorc=zeros(T,nw);
problemC=zeros(nw,1);
for j=1:nw %should be NC*nfact if all country factors were present
    [yc,xc]=preparex([pmatw(1:L,j);pmatw(:,j)],L,1);
    [beta2c(:,:,j),errorc(:,j),rootsc,problemc]=...
    carterkohnAR(yc,xc,Qc(:,:,j),hlastc(:,j),b00c{j}',p00c{j},L,CHECK,maxdraws,1);
if problemc
    beta2c(:,:,j)=beta0c(:,:,j);
else
    beta0c(:,:,j)=beta2c(:,:,j);
end
%draw Qc
resbeta=diff(beta2c(:,:,j));
scaleQ=resbeta'*resbeta+Q0c{j};
Qc(:,:,j)=iwpq(T+T0,invpd(scaleQ));
problemC(j)=problemc;
end


%% 1.C: Idiosyncratic Factors
beta2e=zeros(T,Lx,NN);
errore=zeros(T,NN);
problemE=zeros(NN,1);
for j=1:NN
    [ye,xe]=preparex([resLoading(1:Lx,j);resLoading(:,j)],Lx,0);
    [beta2e(:,:,j),errore(:,j),rootse,probleme]=...
    carterkohnAR(ye,xe,Qe(:,:,j),hlaste(:,j),b00e{j}',p00e{j},Lx,CHECK,maxdraws,0);
if probleme
    beta2e(:,:,j)=beta0e(:,:,j);
else
    beta0e(:,:,j)=beta2e(:,:,j);
end
%  draw Qe
resbeta=diff(beta2e(:,:,j));
scaleQ=resbeta'*resbeta+Q0e{j};
Qe(:,:,j)=diag(diag(iwpq(T+T0,invpd(scaleQ))));
problemE(j)=probleme;
end


%% 2. Stochastic volatility of idiosyncratic and country factor
% Jacquier et al. (2004)
%   let r_it = e_it - beta2e * e_it
%   r_it = h_it^0.5 * eps_it
%   ln(h_it) = ln(h_it-1) + g_i^0.5 * u_it


%world
hnewc=zeros(T+1,nw);
for j=1:nw
    hnewc(:,j)=getsvol(hlastc(:,j),gc(j),log(s00c{j}),SS0,errorc(:,j));
    hlastc(:,j)=hnewc(:,j);
    gerrors=diff(log(hlastc(:,j)));
    gc(j)=IG(Tg0,g0,gerrors);
end

%idiosyncratic
hnewe=zeros(T+1,NN);
for j=1:NN
    hnewe(:,j)=getsvol(hlaste(:,j),ge(j),log(s00e{j}),SS0,errore(:,j));
    hlaste(:,j)=hnewe(:,j);
    gerrors=diff(log(hlaste(:,j)));
    ge(j)=IG(Tg0,g0,gerrors);
end  


%% 3. Factor loadings


% ERROR: no identification restriction per country factor
T=rows(pmat);
totalFactors=nfact;
fload=zeros(NN,1+nw,T);    %main eq. loading
resLoading=zeros(T,NN);
restrictions1 = eye(2);
for j = 1:cols(dataS)
    yy=dataS(:,j);
    tmpfload0=FLOAD0(j,:);
    tmpvfload0=PFLOAD0{j};
        xx = [pmatw pmat(:,indexc(j))];
        %remove serial correlation
        yys=transformrho(yy,beta2e(:,:,j));
        xxs=transformrho(xx,beta2e(:,:,j));
        yyss=yys./sqrt(hlaste(2:end,j));
        xxss=xxs./repmat(sqrt(hlaste(2:end,j)),1,cols(xxs));
        %draw from conditional posterior
        %[FL,~,~,~]=carterkohnAR(yyss,xxss,QB{j},hlaste(:,j),tmpfload0,tmpvfload0,1,CHECK,maxdraws,0);
        regidx= 2;
        [FL,~]= carterkohn1(tmpfload0(1:cols(xx)),tmpvfload0(1:cols(xx),1:cols(xx)),ones(T+1,1),QB{j},yyss,xxss);
        
    
  fload(j,:,:)=FL'; %save factor loadings for each country
   resLoading(:,j)=yy-sum(xx.*FL,2);
end


%% 4: Carter Kohn Algorithm to sample the factors
%  MODIFIED MANUALLY IF "L" CHANGES
dataF=zeros(T,NN);
for j=1:NN
    dataF(:,j)=remSC(dataS(:,j),beta2e(:,:,j));
end
dataF(1:Lx,:)=repmat(dataF(Lx+1,:),Lx,1);
%Carter and Kohn algorithm to draw the factor
ns=cols(pmat00);    % (nfact*NC + nfact)*L
beta_tt=zeros(T,ns);          %will hold the filtered state variable
ptt=zeros(T,ns,ns);    % will hold its variance
beta11=pmat00;
p11=vmat00;
totalFactors = NC+nw;
varSize = 1+NW*((LH+1+L));
varState=varcoef';
hlastState=hlast(1:end,1:NW);

for i=1:T    
    %build matrices of state space as they are time-varying
    xvarState=xvar(i,:);
    % OBSERVATION EQUATION: remain as in original example
    H1=zeros(NN,totalFactors);
    H2=H1;
    % factor loadings
    jj=nw+1;
    jjj=1;
    for j=1:nw
        H1(:,j)=fload(:,j,i);
        H2(:,j)=fload(:,j,i).*-squeeze(beta2e(i,:,j));
    end
    for j=1:NC
        floadc=fload(indexc==idc(j),(nw+1):end,i);    %country factor loadings
        tmpbeta2e=squeeze(beta2e(i,:,indexc==idc(j))); %AR coefficient at time t of idiosyncratic shock
     H1(jjj:jjj+rows(floadc)-1,jj+cols(indexc)-1)=floadc;   
     H2(jjj:jjj+rows(floadc)-1,jj+cols(indexc)-1)=floadc.*-tmpbeta2e;  
     jj=jj++cols(indexc);
     jjj=jjj+rows(floadc);
    end
    H=zeros(NN,totalFactors*2);
    H(:,1:totalFactors)=H1;
    H(:,(totalFactors+1):end)=H2;
    
    R=diag(hlaste(i+1,:));
    
    % TRANSITION EQUATION
    FWorld=zeros(nw, totalFactors*2);
    FWorld(1:nw, [1:nw totalFactors+1:totalFactors+nw])=[diag(squeeze(beta2c(i,1,:))) diag(squeeze(beta2c(i,2,:)))];
    F1=varState(:,1:NW*L); %only factor coefs
    FCountry= [zeros(NC,nw) F1(1:NC,1:NW) zeros(NC,nw) F1(1:NC,NW+1:end)];
    %AR 1 and AR2 coefficients
    F=[FWorld; FCountry; eye(totalFactors,totalFactors*2)];
    MU = [ squeeze(beta2c(i,L+1,:))' varState(:,end)' zeros(totalFactors,1)'];
    MU1 = [zeros(nw,1)' (varState(:,NW*L+1:end-1)*xvarState(NW*L+1:end-1)')' zeros(totalFactors,1)'];
    MU = MU + MU1;
    Q=zeros(totalFactors*2,totalFactors*2);
    Q(1,1)=diag(hlastc(i,:));
    Q(2:totalFactors,2:totalFactors)=iamat*diag(hlastState(i,:))*iamat';
    Q(totalFactors+1:totalFactors*2,:)=zeros(totalFactors,totalFactors*2);
%Prediction
x=H;
beta10=MU+beta11*F';
p10=F*p11*F'+Q;
yhat=(x*(beta10)')';                                               
eta= dataF(i,:)-yhat;
feta=(x*p10*x')+R;  
ifeta=invpd(feta);
%updating
K=(p10*x')*ifeta;
beta11=(beta10'+K*eta')';
p11=p10-K*(x*p10);
ptt(i,:,:)=p11;
beta_tt(i,:)=beta11;

end
% Backward recursion to calculate the mean and variance of the distribution of the state
%vector
beta2 = zeros(T,ns);   %this will hold the draw of the state variable
jv1=1:nfact; %index of state variables to extract
jv=jv1;
wa=randn(T,ns);

i=T;  %period t
p00=squeeze(ptt(i,jv1,jv1)); 
%beta2(i,:)=beta_tt(i,:);
beta2(i,jv1)=beta_tt(i:i,jv1)+(wa(i:i,jv1)*cholx(p00));   %draw for beta in period t from N(beta_tt,ptt)
%periods t-1..to .1

for i=T-1:-1:1
 %build matrices of transition equation
    Q=zeros(totalFactors*2,totalFactors*2);
    Q(1,1)=diag(hlastc(i,:));
    Q(2:totalFactors,2:totalFactors)=iamat*diag(hlastState(i+1,:))*iamat';
    Q(totalFactors+1:totalFactors*2,:)=zeros(totalFactors,totalFactors*2);

    FWorld=zeros(nw, totalFactors*2);
    FWorld(1:nw, [1:nw totalFactors+1:totalFactors+nw])=[diag(squeeze(beta2c(i+1,1,:))) diag(squeeze(beta2c(i,2,:)))];
    F1=varState(:,1:NW*L); %only factor coefs
    FCountry= [zeros(NC,nw) F1(1:NC,1:NW) zeros(NC,nw) F1(1:NC,NW+1:end)];
    %AR 1 and AR2 coefficients
    F=[FWorld; FCountry; eye(totalFactors,totalFactors*2)];
    MU = [ squeeze(beta2c(i+1,L+1,:))' varState(:,end)' zeros(totalFactors,1)'];
    MU1 = [zeros(nw,1)' (varState(:,NW*L+1:end-1)*xvarState(NW*L+1:end-1)')' zeros(totalFactors,1)'];
    MU = MU + MU1;


f=F(jv,:);
q=Q(jv,jv);
mu=MU(jv);
pt=squeeze(ptt(i,:,:));
ifptfq=invpd(f*pt*f'+q);
bm=beta_tt(i:i,:)+(pt*f'*ifptfq*(beta2(i+1:i+1,jv)-mu-beta_tt(i,:)*f')')';  
pm=pt-pt*f'*ifptfq*f*pt;  
%beta2(i,:)=bm;
beta2(i:i,jv1)=bm(jv1)+(wa(i:i,jv1)*cholx(pm(jv1,jv1)));  
end


pmatnew=beta2(:,jv1);   %update the factors
pmatw=pmatnew(:,1:nw);
pmat=pmatnew(:,nw:end);
YWorld = [pmat(1:L,:); pmat];
XWorld =[ lag0(YWorld,1) lag0(YWorld,2) ones(size(YWorld,1),1) ]; % 2 lags
yF=YWorld(3:end,:);
xF=XWorld(3:end,:);
xvar=[xF(:,1:cols(XFactor)-1) log(hlast(1:end,1:NW*(LH+1))) xF(:,end:end)];


vars = {'beta_tt','ptt','beta11','p11','p10','beta10','eta','x','yhat',...
    'feta','R','F','K','MU','p00','pt','pm','bm','wa'};
clear(vars{:})

if igibbs>BURN
    if sum(Sindex==igibbs)>0
%% save variables

  pmatsave(jgibbs,:,:)=pmat;
  pmatwsave(jgibbs,:,:)=pmatw;
  hwsave(jgibbs,:,:)=hlastc;
  hesave(jgibbs,:,:)=hlaste;
  hvarsave(jgibbs,:,:)=hlast(:,1:nfact);
  floadsave(jgibbs,:,:,:)=fload;
  amatsave(jgibbs,:,:)=iamat;
  betasave(jgibbs,:,:)=betaVAR;
  gammasave(jgibbs,:,:)=betaf;
  qsave(jgibbs,:,:)=Qbig;

  beta2wsave(jgibbs,:,:)=squeeze(beta2c);
  qwsave(jgibbs,:,:)=squeeze(Qc);
  gcsave(jgibbs,:,:)=gc;
  %idiosyncratic
  beta2esave(jgibbs,:,:)=squeeze(beta2e);
  qesave(jgibbs,:,:)=squeeze(Qe);
  gesave(jgibbs,:,:)=squeeze(gc);
  resloadingsave(jgibbs,:,:)=resLoading;

  jgibbs=jgibbs+1;
    end
end

disp(strcat('REPS=',num2str([igibbs ])));

igibbs=igibbs+1;
end

%% Save results

%save posteriors

