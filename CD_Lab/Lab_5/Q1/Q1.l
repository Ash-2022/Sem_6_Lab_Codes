%{ 
int consonants = 0; 
int vowels = 0; 
%} 
%% 
[aeiouAEIOU] {vowels++;;} 
[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z] {consonants++;}
%% 
int main(int argc, char **argv) 
{ 
yylex(); 
printf(" Vowels : %d\n Consonants : %d\n ", vowels , consonants); 
} 

int yywrap() 
{ 
return 1; 
}