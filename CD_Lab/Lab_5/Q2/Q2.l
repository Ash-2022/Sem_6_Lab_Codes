%{ 
int chars = 0; 
int words = 0; 
int lines = 0; 
int blanks = 0;
%} 

%% 
[a-zA-Z]+ { words++; chars += strlen(yytext); } 
\n { chars++; lines++; } 
[' ']  { blanks++; chars++; } 
. { chars++;}
%% 
int main(int argc, char **argv) 
{ 
yylex(); 
printf(" Lines : %d\n Words : %d\n Blanks : %d\n Chars : %d\n", lines, words, blanks, chars); 
} 

int yywrap() 
{ 
return 1; 
}