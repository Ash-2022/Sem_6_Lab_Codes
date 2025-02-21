%{
    #include <stdio.h>
    FILE * outFile;
    int scanf_count = 0;
    int printf_count = 0;
%}

%%
"scanf"   { fprintf(outFile, "READ"); scanf_count++; }
"printf"  { fprintf(outFile, "WRITE"); printf_count++; }
.       {fprintf(outFile, yytext);}
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

    printf("Number of scanf statements: %d\n", scanf_count);
    printf("Number of printf statements: %d\n", printf_count);

    return 0;
}

int yywrap() {
    return 1;
}
