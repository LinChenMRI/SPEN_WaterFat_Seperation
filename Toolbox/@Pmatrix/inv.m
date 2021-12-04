function res = inv(a)
% res = inv(Pmatrix)

if a.adjoint
    res=inv(a.P');
else
    res=inv(a.P);
end

