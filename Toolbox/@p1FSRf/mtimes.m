function res = mtimes(a,b)

if a.adjoint
    res=pinv(a.keymatrix_f)*pinv(a.P_matrix_f)*b;


else
    res=a.P_matrix_f*a.keymatrix_f*b;

end



    
