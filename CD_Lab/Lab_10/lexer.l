%{
#include <string.h>
#include "parser.tab.h"
%}

%option noyywrap

%%

"declare"   { return DECLARE; }
"if"        { return IF; }
"then"      { return THEN; }
"else"      { return ELSE; }
"endif"     { return ENDIF; }

[0-9]+      { 
    yylval.ival = atoi(yytext); 
    return NUMBER; 
}

[a-zA-Z_][a-zA-Z0-9_]* { 
    yylval.sval = strdup(yytext); 
    return ID; 
}

"+"         { return PLUS; }
"-"         { return MINUS; }
"*"         { return MULT; }
"/"         { return DIV; }
"^"         { return POWER; }
"n"         { return 'n'; }

"\n"        { return '\n'; }
[ \t]       ; // Ignore whitespace
.           { return yytext[0]; }

%%
