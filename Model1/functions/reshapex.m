function varcoef= reshapex(beta,N,L)
varcoef=zeros(N*L+1,N);
varcoef(1,1)=beta(1);
varcoef(6,1)=beta(2);
varcoef(end,1)=beta(3);
tmp=reshape(beta(4:end),N*L+1,N-1);
varcoef(:,2:end)=tmp;