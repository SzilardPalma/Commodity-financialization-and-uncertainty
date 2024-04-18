clear
addpath('./functions');
dfolder='./data/';
sfolder='./results/';
file=1;
%Load data and transform
dataS=xlsread("Input1","Transformed","B2:AK312");
[~,titles]=xlsread("Input1","Transformed","B1:AK1");
dataS=standardise(dataS); %standardise data

% Estimation parameters
T0=40;   %training sample
L=2;     %lag for the factor eq and regional factors
Lx=1;    %lag for idiosyncratic component transition eq
REPS=20000;     % total reps = burn-in + keep
BURN=10;        % burn-in
SKIP=10;        % every SKIP draw is kept after burn-in 
maxdraws=100;   % max trys to find stable coefficients
nfact = 2;      % number of global factors
CHECK=1;
Sindex=BURN+1:SKIP:REPS;
fsize=length(Sindex);

% Regional factor indices
indexc=[ones(1,8).*1 ones(1,5).*2 ones(1,5).*3 ...
    ones(1,4).*4 ones(1,4).*5 ones(1,3).*6 ones(1,4).*7 ones(1,3).*8]';
NC=max(indexc); % number of regional factors
NN=cols(dataS); % number of idiosyncratic factors
idc= (1:NC)'; % number of regiona
totalFactors=nfact+cols(indexc); % world + regional;
indexcTotals=NC-4; %the number of country factor that run with 2 world factors (e.g.: agriculturals)
% Global factor indices
indexw(:,1)=[ones(29,1); zeros(7,1)].*1; % financial factor
indexw(:,2)=[ones(22,1); zeros(7,1); ones(7,1)].*2; % macro factor

%% Starting Values and Priors

%initial estimate of the factors
pmatw=zeros(rows(dataS),nfact);
dataSS=dataS;
for i=1:nfact   
    % commodity-financial factors
    if i==1
        dataW = dataS(:,1:29);
    % commodity-macroeconomic factor
    elseif i==2
        dataW = dataS(:,[1:22, 30:36]);
    end
    pmatw(:,i) = extract(dataW,1);      %PC estimator of world factor
    BB=pmatw(:,i)\dataW;
    if i==1
        dataSS(:,1:29) = dataW - pmatw(:,i)*BB;
    elseif i==2
        dataSS(:,[1:22, 30:36]) =dataW - pmatw(:,i)*BB;
    end
end


% extract regional factors
pmatc=zeros(rows(pmatw),NC); %PC for countries
for i=1:NC
    dataC=dataSS(:,indexc==idc(i));
    tmp=extract(dataC,1); 
    pmatc(:,i)=tmp;
    %for the country factors
    BB=pmatc(:,i)\dataC;
end


% factor loadings prior
res=zeros(rows(pmatw),NN); 
scaling=3.5e-04;
QB=cell(NN,1);
FLOAD0=zeros(NN,totalFactors);
PFLOAD0=cell(NN,1);
for j=1:NN
    yy=dataS(:,j);
    worldfactors = pmatw(:,nonzeros(indexw(j,:)));
    xx=[worldfactors pmatc(:,idc==indexc(j))];
    BB=xx\yy;

    res(:,j)=yy-xx*BB;
    SS=(res(:,j)'*res(:,j))/(rows(yy)-cols(xx));
    VV=SS*invpd(xx'*xx);
    FLOAD0(j,1:cols(xx))=BB';
    PFLOAD0{j}=VV;
    QB{j}=VV*scaling*(cols(VV)+1); % IW scaling factor
end


%% VAR priors:

YWorld = [pmatw(1:L,:); pmatw];
XWorld =[ lag0(YWorld,1) lag0(YWorld,2) ones(size(YWorld,1),1) ]; % 2 lags

% leave NaN values
YWorld=YWorld(3:end,:);
XWorld=XWorld(3:end,:);
% number of variables in the VAR model
NW = cols(YWorld); 
npart=20;  %number of particles in stochvol draw


% set starting values and priors using T0 observations:

y0World=YWorld(1:T0,:);
x0World=XWorld(1:T0,:);
b0World=x0World\y0World;
e0World=y0World-x0World*b0World;
sigma0World=(e0World'*e0World)/T0;

V0World=kron(sigma0World,inv(x0World'*x0World));
T01=T0;
%priors for the variance of the transition equation
Q0World=V0World*T0*3.5e-04;  %prior for the variance of the transition equation error
P00World=V0World;             % variance of the intial state vector  variance of state variable p[t-1/t-1]
beta0World=vec(b0World)';     % intial state vector   
QWorld = Q0World;


%starting value for the stochastic volatility
yworld=YWorld(T0+1:end,:);
xworld=XWorld(T0+1:end,:);
YWorld=YWorld(T0+1:end,:);
XWorld=XWorld(T0+1:end,:);
T=rows(xworld);

[outhw,outfw,outgw]=getinitialvol(pmatw,500,400,T0,L);

%define priors for variances of shock to transition eq
%priors and starting values for the stochastic vol
hlastw=(diff(YWorld).^2)+0.0001;
hlastw=[hlastw(1:2,:);hlastw];  % intial value for global svol
gWorld=ones(nfact,1);           % variance of the transition equation
g0=0.01^2;                      %scale parameter for IG
Tg0=1;
mubarWorld=log(diag(sigma0World));
sigmabar=10;
NS=cols(hlastw);


% A matrix priors
A0 = eye(nfact).*0.5;
VA0 = 1*ones(nfact);
p=L;
k = 1+NW*p;

% end of VAR params


% Priors for TVP parameters
% regional factor AR

%priors for TVP parameters
scale=3.5e-04;
b00c=cell(NC,1);
s00c=cell(NC,1);
p00c=cell(NC,1);
Q0c=cell(NC,1);
Qc=zeros(L+1,L+1,NC);
for j=1:NC
        [y0c,x0c]=preparex(pmatc(1:T0,j),L,1);
        [b00c{j},s00c{j},p00c{j}]=getols(y0c,x0c);
        Q0c{j}=scale*p00c{j}*T0;  %Qc~IW(Q0c{j},T0) for j=1,2,...NC
        Qc(:,:,j)=scale*p00c{j}*T0; %starting value for Qc
end


%idiosyncratic factor
b00e=cell(NN,1);
s00e=cell(NN,1);
p00e=cell(NN,1);
Q0e=cell(NN,1);
Qe=zeros(Lx,Lx,NN);
for j=1:NN
    [y0e,x0e]=preparex(res(1:T0,j),Lx,0);
    [b00e{j},s00e{j},p00e{j}]=getols(y0e,x0e);
    Q0e{j}=scale*p00e{j}*T0; %Qe~IW(Q0e{j},T0) for j=1,2,...NN
    Qe(:,:,j)=scale*p00e{j}*T0; %Starting values
end

%remove training sample
dataS=dataS(T0+1:end,:);
pmatw=pmatw(T0+1:end,:);
pmatc=pmatc(T0+1:end,:);
res=res(T0+1:end,:);
T=rows(dataS);


%priors and starting values for stochastic volatilties as residual^2+small
% regional factor stochvol
hlastc=zeros(T+1,NC);

for j=1:NC
        [y0c,x0c]=preparex(pmatc(:,j),L,1);
        [~,~,~,epsc]=getols(y0c,x0c);%regression of factor on lags
        hlastcc=epsc.^2+0.0001;%residual^2+small number
        hlastcc=[hlastcc(1:L+1,:);hlastcc];
        hlastc(:,j)=hlastcc;
end

%idiosyncratic factor stochvol
hlaste=zeros(T+1,NN);
for j=1:NN
    [y0e,x0e]=preparex(res(:,j),Lx,0);
    [~,~,~,epse]=getols(y0e,x0e);%regression of factor on lags
    hlastee=epse.^2+0.0001;%residual^2+small number
    hlastee=[hlastee(1:Lx+1,:);hlastee];
    hlaste(:,j)=hlastee;
end

SS0=10;    % intial variance of SVOL
g0=0.1^2;  % prior scale parameter for IG
Tg0=1;     % prior degrees of freedom for IG
gc=ones(NC,1).*g0;  % starting values
ge=ones(NN,1).*g0;  %s tarting values


% loadings
beta0c=zeros(T,L+1,NC);

for j=1:NC
    beta0c(:,:,j)=repmat(b00c{j}',T,1);
end
beta0e=zeros(T,Lx,NN);
for j=1:NN
    beta0e(:,:,j)=repmat(b00e{j}',T,1);
end

%initial conditions for the factors
pmat00=[pmatw(L,:) pmatc(L,:)];
for j=1:L-1
    pmat00=[pmat00 [pmatw(L-j,:) pmatc(L-j,:)]];
end
vmat00=eye(cols(pmat00))*1;


save priors

decompsave=zeros(fsize,T,NN);   %IMPORTANT
pmatsave=zeros(fsize,T,nfact+NC);
hsave=zeros(fsize,T+1,nfact+NC+NN);
iamatsave=zeros(fsize,T+1,NW*NW);
floadsave=zeros(fsize,NN,nfact+1,T);

jgibbs=1;
igibbs=1;

%% GIBBS

while jgibbs<=fsize
%%%%%%%%%%%%%%%%% Gibbs Step 1: Draw TVP-VAR-SV %%%%%%%%%%%%%%%%%
%% 1.A: World Factor

A = (XWorld'*XWorld + .01*speye(k))\(XWorld'*YWorld);
U = YWorld-XWorld*A;
alpha = A(:);
[T,n] = size(YWorld);

for i=1:nfact
    EiOhi = U'*sparse(1:T,1:T,1./hlastw(1:end-1,i));       
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
        A0(i,:) = phii*sign(phii(i)); % fix the sign of the i-th element to be positive
end   
iamat = inv(A0);


%% 1.B: Draw (beta | A,H,Q)  with Kalman-filter

ns=cols(beta0World);
T=rows(XWorld);
F=eye(ns);
mu=0;
beta_tt=[];            % holds the filtered state variable
ptt=zeros(T,ns,ns);    % holds its variance
beta11=beta0World;
p11=P00World;


for i=1:T
    x=kron(eye(nfact),XWorld(i,:));
    A = A0;
    H=diag(hlastw(i,:));
    R=inv(A)*H*inv(A)';

    %Prediction
    beta10=mu+beta11*F';
    p10=F*p11*F'+QWorld;
    yhat=(x*(beta10)')';                                               
    eta=YWorld(i,:)-yhat;
    feta=(x*p10*x')+R;

    %updating
    K=(p10*x')*inv(feta);
    beta11=(beta10'+K*eta')';
    p11=p10-K*(x*p10);
    ptt(i,:,:)=p11;
    beta_tt=[beta_tt;beta11];
end

% end of Kalman-filter

%s Backward recursion, Carter-Kohn

chck=-1;
while chck<0
    beta2World = zeros(T,ns);   %this will hold the draw
    wa=randn(T,ns);
    error=zeros(T,nfact);
    roots=zeros(T,nfact);

    i=T;  
    p00=squeeze(ptt(i,:,:)); 
    beta2World(i,:)=beta_tt(i:i,:)+(wa(i:i,:)*chol(p00));   %draw for beta in period t from N(beta_tt,ptt)
    error(i,:)=YWorld(i,:)-XWorld(i,:)*reshape(beta2World(i:i,:),nfact*L+1,nfact);  %var residuals
    roots(i)=stability(beta2World(i,:)',nfact,L,1);

    %periods t-1..to .1
    for i=T-1:-1:1
        pt=squeeze(ptt(i,:,:));
        bm=beta_tt(i:i,:)+(pt*F'*inv(F*pt*F'+QWorld)*(beta2World(i+1:i+1,:)-beta_tt(i,:)*F')')';  %update the filtered beta for information contained in beta[t+1]                                                                                 %i.e. beta2(i+1:i+1,:) eq 8.16 pp193 in Kim Nelson
        pm=pt-pt*F'*inv(F*pt*F'+QWorld)*F*pt;       
        beta2World(i:i,:)=bm+(wa(i:i,:)*chol(pm));  
        error(i,:)=YWorld(i,:)-XWorld(i,:)*reshape(beta2World(i:i,:),nfact*L+1,nfact); 
        roots(i)=stability(beta2World(i,:)',nfact,L,1);
    end

    if sum(roots)==0
        chck=1;
    end
end


%% 1.C:. Draw Q~IW() by calculating the residuals: b_t = b_t-1 + e_t, VAR(e_t) = Q
errorq=diff(beta2World);
scaleQ=(errorq'*errorq)+Q0World;
QWorld=iwwpQ(T+T0,inv(scaleQ));



%% 1.D: Draw h_i,t by eps = A_t * v_t
 epsilon=[];

 for i=1:T
     epsilon=[epsilon;error(i,:)*iamat];
 end

% sample stochastic vol for each epsilon using the MH algorithm
hnew=[];
for i=1:nfact
    htemp=getsvol(hlastw(:,i),gWorld(i),mubarWorld(i),sigmabar,epsilon(:,i));
    hnew=[hnew htemp];
end
hlastw=hnew;

clear htemp epsilon;


%% 1.E: Draw (g_i | h_it) ~ IG()
for i=1:nfact
    gerrors=diff(log(hnew(:,i)));
    gWorld(i)=IG(Tg0,g0,gerrors);  %draw from the IG distribution
end


%% 2 Regional Factors
% TVP parameters
beta2c=zeros(T,L+1,NC);
errorc=zeros(T,NC);
problemC=zeros(NC,1);

for j=1:NC 
    [yc,xc]=preparex([pmatc(1:L,j);pmatc(:,j)],L,1);
    [beta2c(:,:,j),errorc(:,j),rootsc,problemc]=...
    carterkohnAR(yc,xc,Qc(:,:,j),hlastc(:,j),b00c{j}',p00c{j},L,CHECK,maxdraws,1);
    if problemc
        beta2c(:,:,j)=beta0c(:,:,j);
    else
        beta0c(:,:,j)=beta2c(:,:,j);
    end

    % draw covariance from IW distribution
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
    [ye,xe]=preparex([res(1:Lx,j);res(:,j)],Lx,0);
    [beta2e(:,:,j),errore(:,j),rootse,probleme]=...
    carterkohnAR(ye,xe,Qe(:,:,j),hlaste(:,j),b00e{j}',p00e{j},Lx,CHECK,maxdraws,0);
    if probleme
        beta2e(:,:,j)=beta0e(:,:,j);
    else
        beta0e(:,:,j)=beta2e(:,:,j);
    end
    % draw covariance from IW distribution
resbeta=diff(beta2e(:,:,j));
scaleQ=resbeta'*resbeta+Q0e{j};
Qe(:,:,j)=iwpq(T+T0,invpd(scaleQ));
problemE(j)=probleme;
end


%% 3. Stochastic volatility of idiosyncratic and regional factor
% Jacquier et al. (2004)
%   let r_it = e_it - beta2e * e_it
%   r_it = h_it^0.5 * eps_it
%   ln(h_it) = ln(h_it-1) + g_i^0.5 * u_it

% regional
hnewc=zeros(T+1,NC);
for j=1:NC
    hnewc(:,j)=getsvol(hlastc(:,j),gc(j),log(s00c{j}),SS0,errorc(:,j));
    hlastc(:,j)=hnewc(:,j);
    gerrors=diff(log(hlastc(:,j)));
    gc(j)=IG(Tg0,g0,gerrors);
end

% idiosyncratic
hnewe=zeros(T+1,NN);
for j=1:NN
    hnewe(:,j)=getsvol(hlaste(:,j),ge(j),log(s00e{j}),SS0,errore(:,j));
    hlaste(:,j)=hnewe(:,j);
    gerrors=diff(log(hlaste(:,j)));
    ge(j)=IG(Tg0,g0,gerrors);
end  


%% 4. Factor loadings

T=rows(pmatw);
totalFactors=nfact+cols(indexc);
fload=zeros(NN,totalFactors,T);    %main eq. loading
res=zeros(T,NN);
restrictions1 = eye(totalFactors);

for j = 1:cols(dataS)
    yy=dataS(:,j);
    tmpfload0=FLOAD0(j,:);
    tmpvfload0=PFLOAD0{j};
    fload1=zeros(totalFactors,T);
    res1=zeros(T,cols(yy));
    if indexc(j)<=indexcTotals %if all 3 world factors are present
        xx=[pmatw pmatc(:,indexc(j))];

        % remove serial correlation
        yys=transformrho(yy,beta2e(:,:,j));
        xxs=transformrho(xx,beta2e(:,:,j));

        %r emove heteroscedasticity
        yyss=yys./sqrt(hlaste(2:end,j));
        xxss=xxs./repmat(sqrt(hlaste(2:end,j)),1,cols(xxs));

        regidx = totalFactors; 

        % draw TVP OLS with Carter and Kohn (1994)
        [FL,~]=carterkohn1(tmpfload0(1:regidx),tmpvfload0(1:regidx,1:regidx),ones(T+1,1),QB{j},yyss,xxss);
        fload1(1:regidx,:)=FL';

        % identification
        if sum(ismember([1 2 3],j))==1
            fload1(1:regidx,:)=(restrictions1(j,:).*ones(rows(FL),1))';
        end

        res1=yy-xx*FL'; 
    else
        worldFactorIndex = nonzeros(indexw(j,:));
        xx=[pmatw(:,worldFactorIndex) pmatc(:,indexc(j))];

        %remove serial correlation
        yys=transformrho(yy,beta2e(:,:,j));
        xxs=transformrho(xx,beta2e(:,:,j));
        yyss=yys./sqrt(hlaste(2:end,j));
        xxss=xxs./repmat(sqrt(hlaste(2:end,j)),1,cols(xxs));
        [FL,~]=carterkohn1(tmpfload0(1:cols(xx)),tmpvfload0(1:cols(xx),1:cols(xx)),ones(T+1,1),QB{j},yyss,xxss);
        fload1([worldFactorIndex totalFactors],:)=FL';
        res1=yy-xx*FL';
    end
   fload(j,:,:)=fload1; %save factor loadings
   res(:,j)=yy-sum(xx.*FL,2);
end


%% 5: Carter Kohn Algorithm to sample the factors
%  MODIFIED MANUALLY IF "L" CHANGES

dataF=zeros(T,NN);
for j=1:NN
    dataF(:,j)=remSC(dataS(:,j),beta2e(:,:,j));
end
dataF(1:Lx,:)=repmat(dataF(Lx+1,:),Lx,1);
ns=cols(pmat00);    % (nfact*NC + nfact)*L
beta_tt=zeros(T,ns);   % will hold the filtered state variable
ptt=zeros(T,ns,ns);    % will hold its variance
beta11=pmat00;
p11=vmat00;
totalFactors = cols(pmatc)+nfact;

for i=1:T    
    %build matrices of state space as they are time-varying

    H1=zeros(NN,totalFactors);
    H2=H1;

    %world factor loadings
    for j=1:nfact
        H1(:,j)=fload(:,j,i);
        H2(:,j)=fload(:,j,i).*-squeeze(beta2e(i,:,j));
    end

    % regional factor loadings
    jj=nfact+1;
    jjj=1;
    for j=1:NC
        floadc=fload(indexc==idc(j),(nfact+1):end,i);    % regional  factor loadings
        tmpbeta2e=squeeze(beta2e(i,:,indexc==idc(j)));  % AR coefficient at time t of idiosyncratic shock
     H1(jjj:jjj+rows(floadc)-1,jj+cols(indexc)-1)=floadc;   
     H2(jjj:jjj+rows(floadc)-1,jj+cols(indexc)-1)=floadc.*-tmpbeta2e;  
     jj=jj+cols(indexc);
     jjj=jjj+rows(floadc);
    end
    H=zeros(NN,(cols(pmatc)+nfact)*2);
    H(:,1:(cols(pmatc)+nfact))=H1;
    H(:,(cols(pmatc)+nfact+1):end)=H2;
    
    R=diag(hlaste(i+1,:));
    
    % TRANSITION EQUATION
    bWorldMat=reshape(beta2World(i,:),nfact*L+1,nfact)';
    FWorld=zeros(nfact,(totalFactors)*2);
    FCountry= zeros(NC,(totalFactors)*2);
    FWorld(1:nfact, [1:nfact (totalFactors+1):(totalFactors+nfact)]) = bWorldMat(:,1:nfact*2);
    % AR1 and AR2 coefficients
    FCountry(1:NC, [nfact+1:totalFactors (totalFactors+nfact+1):(totalFactors*2)])=[diag(squeeze(beta2c(i,1,:))) diag(squeeze(beta2c(i,2,:)))];
    F=[FWorld; FCountry; eye(totalFactors,totalFactors*2)];
    MU = [bWorldMat(:,nfact+1)' squeeze(beta2c(i,L+1,:))' zeros(totalFactors,1)'];
    % Covariance matrix
    Q=zeros(totalFactors*2,totalFactors*2);
    Q(1:nfact,1:nfact)=iamat*diag(hlastw(i,:))*iamat';
    Q(nfact+1:totalFactors,nfact+1:totalFactors)=diag(hlastc(i,:));
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
% Carter-Kohn (1994):
% Backward recursion to calculate the mean and variance of the distribution of the state vector
beta2 = zeros(T,ns);  
jv1=1:totalFactors; 
jv=jv1;
wa=randn(T,ns);

i=T; 
p00=squeeze(ptt(i,jv1,jv1)); 
beta2(i,jv1)=beta_tt(i:i,jv1)+(wa(i:i,jv1)*cholx(p00));   %draw for beta in period t from N(beta_tt,ptt)

for i=T-1:-1:1
    %build matrices of transition equation

    Q=zeros(totalFactors*2,totalFactors*2);
    Q(1:nfact,1:nfact)=iamat*diag(hlastw(i+1,:))*iamat';
    Q(nfact+1:totalFactors,nfact+1:totalFactors)=diag(hlastc(i+1,:));
    Q(totalFactors+1:totalFactors*2,:)=zeros(totalFactors,totalFactors*2);

    bWorldMat=reshape(beta2World(i+1,:),nfact*L+1,nfact)';
    FWorld=zeros(nfact,(totalFactors)*2);
    FCountry= zeros(NC,(totalFactors)*2);
    FWorld(1:nfact, [1:nfact (totalFactors+1):(totalFactors+nfact)]) = bWorldMat(:,1:nfact*2);
    FCountry(1:NC, [nfact+1:totalFactors (totalFactors+nfact+1):(totalFactors*2)])=[diag(squeeze(beta2c(i+1,1,:))) diag(squeeze(beta2c(i+1,2,:)))];
    F=[FWorld; FCountry; eye(totalFactors,totalFactors*2)];
    MU = [bWorldMat(:,nfact+1)' squeeze(beta2c(i+1,L+1,:))' zeros(totalFactors,1)'];

f=F(jv,:);
q=Q(jv,jv);
mu=MU(jv);
pt=squeeze(ptt(i,:,:));
ifptfq=invpd(f*pt*f'+q);
bm=beta_tt(i:i,:)+(pt*f'*ifptfq*(beta2(i+1:i+1,jv)-mu-beta_tt(i,:)*f')')';  
pm=pt-pt*f'*ifptfq*f*pt;  
beta2(i:i,jv1)=bm(jv1)+(wa(i:i,jv1)*cholx(pm(jv1,jv1)));  
end


pmat=beta2(:,jv1);      %update the factors
pmatw=pmat(:,1:nfact);  % global factors
pmatc=pmat(:,(nfact+1):end);    % regional factors

YWorld = [pmatw(1:L,:); pmatw];
XWorld =[ lag0(YWorld,1) lag0(YWorld,2) ones(size(YWorld,1),1) ]; % 2 lags
YWorld=YWorld(3:end,:);
XWorld=XWorld(3:end,:);


vars = {'beta_tt','ptt','beta11','p11','p10','beta10','eta','x','yhat',...
    'feta','R','F','K','MU','p00','pt','pm','bm','wa'};
clear(vars{:})

if igibbs>BURN
    if sum(Sindex==igibbs)>0

%% calculate variance decomposition


VOLW=zeros(T,nfact);
VOLC=zeros(T,NC);
VOLE=zeros(T,NN);
for i=1:T
    sigmaW=iamat*diag(hlastw(i,:))*iamat';
    tmp=volatilityTVP(beta2World(i,:)',sigmaW,nfact,L);
    VOLW(i,:)=diag(tmp);

    %Volatility of the country factor
    H1=zeros(NN,totalFactors);
    jj=nfact+1;
    jjj=1;
    for j=1:NC
        VOLC(i,j)=volatility(beta2c(i,:,j)',hlastc(i+1,j),1,L);
        floadcc=fload(indexc==idc(j),(nfact+1):end,i);    %country factor loadings
        H1(jjj:jjj+rows(floadcc)-1,jj+cols(indexc)-1)=floadcc; 
        jj=jj+cols(indexc);
        jjj=jjj+rows(floadcc);
    end

    for j=1:NN
        VOLE(i,j)=volatility([beta2e(i,:,j)';0],hlaste(i+1,j),1,Lx);
    end

   
   for j=1:nfact
        H1(:,j)=fload(:,j,i);
   end
  
   floadsquared=H1.^2;

 totalvol=[VOLW VOLC]*floadsquared'+VOLE;   
 totalvolw(i,:)=VOLW(i,:)*floadsquared(:,1:nfact)';
 totalvolw1(i,:)=VOLW(i,1)*floadsquared(:,1)';
 totalvolw2(i,:)=VOLW(i,2)*floadsquared(:,2)';
 totalvolc(i,:)=VOLC(i,:)*floadsquared(:,nfact+1:(NC+nfact))';
 TOTALVOL=[VOLW zeros(rows(VOLC),cols(VOLC))]*floadsquared';  
end

   
 volwdecompsave(jgibbs,:,:)=totalvolw./totalvol;
 volw1decompsave(jgibbs,:,:)=totalvolw1./totalvol;
 volw2decompsave(jgibbs,:,:)=totalvolw2./totalvol;
 volcdecompsave(jgibbs,:,:)=totalvolc./totalvol;
 volidiosdecompsave(jgibbs,:,:) = VOLE./totalvol;


  pmatsave(jgibbs,:,:)=[pmatw pmatc];
  hsave(jgibbs,:,:)=[hlastw hlastc hlaste];
  floadsave(jgibbs,:,:,:)=fload;
   
  %VAR
  beta2wsave(jgibbs,:,:)=beta2World;
  AWorldsave(jgibbs,:,:)=iamat;
  Qsave(jgibbs,:,:)=QWorld;



  jgibbs=jgibbs+1;
    end
end

disp(strcat('REPS=',num2str([igibbs ])));

igibbs=igibbs+1;
end

%% Plot results

%save posteriors2factor
%load posteriors2factor
%save posterior2factors
save posteriors




tmp=prctile((pmatsave),[50 38 68]);
pmatTitle=({'Macro','Financial','Agricultural',...
    'Energy','Industrial metal','Precious metal',...
    'Economic activity', 'Macro expectations','Financial stress',...
    'Financial activity'});


avgvolw1=squeeze(mean(volw1decompsave,1));
avgvolw2=squeeze(mean(volw2decompsave,1));
avgvolc=squeeze(mean(volcdecompsave,1));

% Figure 1.
X=datetime(2001,5,31):calmonths(1):datetime(2023,12,29);
years=datetime(2001,5,31):calyears(3):datetime(2023,12,29); % xticks
figure(1)
index = reshape(1:20, 4,5).';   % re-order subplot

set(gcf,'papertype','A4');   
fig = gcf;
fig.PaperSize=[21 29.7];
fig.PaperPosition = [1 1 20 25];
print(fig,'test','-dbmp' , '-r300')

grid on
for j=4:22
    subplot(6,4,index(j-3)+4);
    Y=[avgvolw1(:,j) avgvolw2(:,j) avgvolc(:,j)];
    area(X,Y)
    ylabel('Variance decomp.','fontweight','bold')
    if sum(ismember([8 13 18 22],j))==1
        xlabel('Date','fontweight','bold')
    end
    title(char(titles(j)),'fontweight','bold','FontSize',12)
    ylim([0 1])
    newcolors = ["#540202", "#4B791E" "#E0C99A"];
    colororder(newcolors);
    ax = gca;
    grid("on");
    xticks(years)
    xtickangle(45)
    ax.XAxis.TickLabelInterpreter='none';
    ax.XAxis.TickLabelFormat='yyyy';
    ax.GridColor = [.7 .7 .7];
    ax.GridLineStyle = '-.';
    ax.GridAlpha = 0.3;
end

lgnd= legend({ 'Financial','Macro', 'Regional'}, 'Position',[0.765 0.127 0.095 0.075]);
set(lgnd,'color','none');
fontsize(lgnd,11,'points');

