function omega = volatilityTVP(betaCoef,covmat,n,l )
[FF,mu]=comp(betaCoef,n,l,1);
   
    OMEGA=zeros(rows(FF),rows(FF));
    OMEGA(1:n,1:n)=covmat;
    %unconditional variance
    uvar=doublej(FF,OMEGA);
    omega=uvar(1:n,1:n);
     
end