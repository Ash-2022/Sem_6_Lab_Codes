%{
#include <stdio.h>
#include <ctype.h>

FILE *outFile;

int toUpper(int c) {
    return toupper(c);
}

%}

%%

"//".*                  { fprintf(outFile, "%s", yytext); } 
"/*"(.|\n)*"*/"          { fprintf(outFile, "%s", yytext); } 
[a-z]                   { fprintf(outFile, "%c", toUpper(yytext[0])); }
[A-Z]                   { fprintf(outFile, "%s", yytext); }
[0-9]                   { fprintf(outFile, "%s", yytext); }
[(){}<>=+-/*&%$#@!,. ]   { fprintf(outFile, "%s", yytext); }
\n                      { fprintf(outFile, "%s", yytext); }
%%

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    FILE *inputFile = fopen(argv[1], "r");
    if (!inputFile) {
        perror("Error opening input file");
        return 1;
    }

    outFile = fopen(argv[2], "w");
    if (!outFile) {
        perror("Error opening output file");
        fclose(inputFile);
        return 1;
    }

    yyin = inputFile;
    yyout = outFile;
    yylex();
    fclose(inputFile);
    fclose(outFile);
    return 0;
}

int yywrap() {
    return 1;
}
