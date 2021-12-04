function res = mtimes(a,b)

if a.adjoint
    res=pinv(a.keymatrix_w)*pinv(a.P_matrix_w)*b;


else
    res=a.P_matrix_w*a.keymatrix_w*b;

end



    
