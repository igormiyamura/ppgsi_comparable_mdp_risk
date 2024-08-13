pr = 0.7;
cr = 2.3;

psi = 'u'

if psi == 'u'

%EUM
L = 0.3;

g = @( L,c ) sign(L)*exp(L*c);
f = @( L,c,p ) sign(L)*exp(L*c)*p / ( 1 - exp(L*c)*(1-p));
c = @( L,cr,pr, p ) log (  exp(L*cr)*pr / (p - exp(L*cr)*p + exp(L*cr)*pr)  ) *1/L;  


P = 1;
C = 0;
S = 0;
for i=1:200
  C = C + cr;
  S = S + g(L,C)*P*pr;
  P = P*(1-pr);
endfor



disp(S)
disp(f(L,cr,pr))
p =0.5;
disp(f(L,   c(L,cr,pr,p)   ,p))

end

if psi == 'l'

%PLT
k = 0.7; %positive

X = @( k,x )  (1-sign(x))/2*(1-k)*x  + (1+sign(x))/2*(1+k)*x;
f = @( k,c,p ) -c*(2*k*p - k - 1) / ( (1-k)*p ) ;
c = @( k,cr,pr, p ) cr*p*(2*k*pr - k - 1) / ( pr*(2*k*p - k - 1) )

V = 0;
for i=1:1000
  V = V +  0.5*(pr* X( k, cr - V) + (1-pr)* X( k, cr + V - V));
endfor


disp(V +  0.01*(pr* X( k, cr - V) + (1-pr)* X( k, cr + V - V)))
disp(V)
disp(f(k,cr,pr))
p =0.5;
disp(f(k,   c(k,cr,pr,p)   ,p))

end


if psi == 'v'

%VaR
a = 0.027; %positive

f = @( a,c,p ) c*ceil(log(a)/log(1-p)) ;
c = @( a,cr,pr, p ) cr*ceil(log(a)/log(1-pr))/ceil(log(a)/log(1-p)) ;

P = 0;
C = 0;
S = 0;
for i=1:200
  C = i*cr;
  P = P + (1-pr)^(i-1)*pr; 
  if a >= 1-P
    break;
  end
    
endfor


disp([(1-P) a])
disp(C)
disp(f(a,cr,pr))
p =0.5;
disp(f(a,   c(a,cr,pr,p)   ,p))

end


if psi == 'v'

%VaR
a = 0.027; %positive

f = @( a,c,p ) c*ceil(log(a)/log(1-p)) ;
c = @( a,cr,pr, p ) cr*ceil(log(a)/log(1-pr))/ceil(log(a)/log(1-p)) ;

P = 0;
C = 0;
S = 0;
for i=1:200
  C = i*cr;
  P = P + (1-pr)^(i-1)*pr; 
  if a >= 1-P
    break;
  end
    
endfor


disp([(1-P) a])
disp(C)
disp(f(a,cr,pr))
p =0.5;
disp(f(a,   c(a,cr,pr,p)   ,p))

end



if psi == 'c'

%CVaR
a = 0.125; %positive

%f = @( a,c,p ) c*ceil(log(a)/log(1-p)) ;
%c = @( a,cr,pr, p ) cr*ceil(log(a)/log(1-pr))/ceil(log(a)/log(1-p)) ;

P = 0;
var = 0;
S = 0;
for i=1:2000
  v = i*cr;
  P = P + (1-pr)^(i-1)*pr; 
  if a >= 1-P
    break;
  end
    
endfor

disp([(1-P) a])

S = 0;
P = pr;
for i=1:2000
  S = S + P*(v + max( 0, i*cr - v ));
  P = P*(1-pr);  
endfor



disp(S)
disp(f(a,cr,pr))
p =0.5;
disp(f(a,   c(a,cr,pr,p)   ,p))

end