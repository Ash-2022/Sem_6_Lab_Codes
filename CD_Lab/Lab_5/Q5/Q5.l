%{
#include <stdio.h>
#include <stdlib.h>

char* decimalToHex(int decimal) {
    char* hexadecimal = (char*)malloc(100 * sizeof(char)); 
    sprintf(hexadecimal, "%X", decimal); 
    return hexadecimal;
}

%}

%%

[0-9]+  {
            int decimal = atoi(yytext); 
            char* hex = decimalToHex(decimal); 
            printf("0x%s",hex); 
        }
.       { ECHO; }
%%

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE *inputFile = fopen(argv[1], "r");
    if (!inputFile) {
        perror("Error opening input file");
        return 1;
    }

    yyin = inputFile;
    yylex();
    fclose(inputFile);

    return 0;
}

int yywrap() {
    return 1;
}
