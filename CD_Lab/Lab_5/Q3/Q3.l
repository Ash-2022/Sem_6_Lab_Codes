%{ 
int posInt = 0; 
int negInt = 0; 
int posFloat = 0; 
int negFloat = 0; 
%} 

%% 
[0-9]+ { posInt++; } 
[-][0-9]+ { negInt++; } 
[0-9]*[.][0-9]+ { posFloat++; } 
[-][0-9]*[.][0-9]+ { negFloat++;}
%% 
int main(int argc, char **argv) 
{ 
yylex(); 
printf(" posInt : %d\n negInt : %d\n posFloat : %d\n negFloat : %d\n", posInt, negInt, posFloat, negFloat); 
} 

int yywrap() 
{ 
return 1; 
}